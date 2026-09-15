"""Nano original-weight step/CFG latency budget, not a distilled quality result."""
import argparse
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import random
import statistics
import sys
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--steps', type=int, choices=[1,2], required=True)
p.add_argument('--guidance', type=float, choices=[1.,3.], required=True)
p.add_argument('output', type=Path)
p.add_argument('--attention', choices=['native', 'cudnn'], default='native')
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--iterations', type=int, default=10)
a = p.parse_args()
if a.iterations < 10 or a.output.exists() or a.output.with_suffix('.npz').exists():
    p.error('Use a fresh output and at least ten measured requests')
assert torch.cuda.get_device_capability() == (11, 0)
for key in list(os.environ):
    if key.startswith('IFL_COSMOS3_'):
        del os.environ[key]
os.environ.update(IFL_COSMOS3_EXACT_POINTWISE='1', IFL_COSMOS3_LAYER_GRAPHS='1', IFL_COSMOS3_ATTENTION='native')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
from instinctflash import Runtime, register
from instinctflash.planners.planner import PassResult, Tier
report = dict(ok=False, category='OPERATING-POINT', task_quality_certified=False, attention=a.attention,
              checkpoint='nvidia/Cosmos3-Nano-Policy-DROID', trained_student=False, requested_steps=a.steps, requested_guidance=a.guidance, torch=torch.__version__, cuda=torch.version.cuda,
              cudnn=torch.backends.cudnn.version(), fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
              benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), calls=[])
api = None
try:
    from huggingface_hub import snapshot_download
    model_id = 'nvidia/Cosmos3-Nano-Policy-DROID'
    snapshot = Path(snapshot_download(model_id, local_files_only=True))
    report['revision'] = snapshot.name
    report['checkpoint_index_sha256'] = hashlib.sha256((snapshot/'model.safetensors.index.json').read_bytes()).hexdigest()
    with np.load(a.fixture, allow_pickle=True) as data:
        frames = [np.asarray(Image.open(io.BytesIO(bytes(row[0]))).convert('RGB').resize((640, 540))) for row in data['jpeg_0'][:12]]
    api = Runtime.from_pretrained(model_id, revision=snapshot.name, nfe={'action': a.steps}, precision='native', tier_ceiling='numeric', placement='in_process')
    api.reset(prompt='pick up the object')
    service = api._backend._impl._service
    # Explicit experimental serving override, recorded separately from checkpoint facts.
    service.cfg = dataclasses.replace(service.cfg, guidance=a.guidance)
    branches = []
    original_velocity = service.model._get_velocity
    def velocity(**kwargs):
        branches.append(bool(kwargs.get('skip_text_tokens', False)))
        return original_velocity(**kwargs)
    service.model._get_velocity = velocity
    if a.attention == 'cudnn':
        # Qualification experiment: Fixed-step/student admission is not enabled in production.
        import cosmos_framework
        import cosmos_framework.model.generator.mot.attention as mot
        from cosmos3_iwm.conditioning_cache import SOURCE_HASHES
        from cosmos3_iwm.numeric_attention import NumericAttention
        root = Path(cosmos_framework.__file__).parent
        for name, digest in SOURCE_HASHES.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
        assert torch.backends.cudnn.version() == 91501
        owners = [layer.self_attn for layer in service.model.net.language_model.model.layers]
        assert len(owners) == 36 and all(owner.dispatch_attention_fn is mot.dispatch_attention for owner in owners)
        service._ifl_numeric_attention = NumericAttention(mot, owners)
        api.plan.results.append(PassResult('experimental_student_cudnn', True, Tier.NUMERIC,
            'Nano step/guidance latency budget; original weights, no task-quality certificate'))
    actions = []
    for i in range(6 + a.iterations):
        others = subprocess.check_output(['/usr/sbin/nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
        assert not [pid for pid in others if int(pid) != os.getpid()], 'GPU contention'
        prompt = ['pick up the object', 'place the object down'][(i//3)%2]
        if i < 6 or i % 5 == 0:
            api.reset(prompt=prompt)
        torch.manual_seed(707+i); np.random.seed(707+i); random.seed(707+i)
        torch.cuda.synchronize(); start = time.perf_counter()
        action = np.asarray(api.predict(dict(image=frames[i%len(frames)].copy(), state=np.full(8, .01*(i%3), np.float32), prompt=prompt))['action'])
        torch.cuda.synchronize(); elapsed = 1000*(time.perf_counter()-start)
        assert action.shape == (32, 8) and np.isfinite(action).all()
        actions.append(action.copy())
        report['calls'].append(dict(i=i, phase='warmup' if i<6 else 'measured', ms=elapsed))
        print(json.dumps(report['calls'][-1]), flush=True)
    stats = api._backend._impl.backend_stats()
    expected_branches = len(actions)*a.steps*(1 if a.guidance == 1 else 2)
    assert len(branches) == expected_branches, (len(branches), expected_branches)
    report['actual_velocity_branches'] = len(branches)
    report['experimental_override'] = {'guidance': a.guidance, 'action_steps': a.steps, 'sampler': 'native UniPC'}
    if a.attention == 'cudnn':
        assert stats['numeric_attention']['eligible_python_calls'] > 0
    report.update(backend_stats=stats, execution_policy=api.execution_policy,
                  p50_ms=statistics.median(c['ms'] for c in report['calls'][6:]),
                  p95_ms=float(np.percentile([c['ms'] for c in report['calls'][6:]],95)), ok=True)
    np.savez_compressed(a.output.with_suffix('.npz'), actions=np.stack(actions))
    report['actions_sha256'] = hashlib.sha256(a.output.with_suffix('.npz').read_bytes()).hexdigest()
except Exception:
    report['error'] = traceback.format_exc()
    traceback.print_exc()
finally:
    report['sources'] = {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for name, m in list(sys.modules.items())
        if name.startswith(('instinctflash', 'instinct_compress', 'cosmos3_iwm', 'cosmos_framework'))
        and getattr(m, '__file__', None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}
    if api is not None:
        api.close()
    a.output.write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['ok'] else 1)
