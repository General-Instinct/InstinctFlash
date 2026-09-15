"""Thor original Nano SDE1/SDE2 cost screen; not a trained student or quality candidate."""
import argparse
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
p.add_argument('checkpoint', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--attention', choices=['native', 'cudnn'], default='native')
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--cache-mode',choices=['off','verify','reuse'],default='off')
p.add_argument('--swiglu', action='store_true')
p.add_argument('--iterations', type=int, default=24)
p.add_argument('--warmup', type=int, default=12)
p.add_argument('--allow-unqualified', action='store_true')
a = p.parse_args()
if a.warmup < 12 or a.iterations < 24 or a.output.exists() or a.output.with_suffix('.npz').exists():
    p.error('Use a fresh output and at least ten measured requests')
assert torch.cuda.get_device_capability() == (11, 0)
for key in list(os.environ):
    if key.startswith('IFL_COSMOS3_'):
        del os.environ[key]
os.environ.update(IFL_COSMOS3_EXACT_POINTWISE='1', IFL_COSMOS3_LAYER_GRAPHS='1', IFL_COSMOS3_ATTENTION='native')
if a.swiglu:
    os.environ['IFL_COSMOS3_SWIGLU']='1'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
from instinctflash import Runtime, register
from instinctflash.planners.planner import PassResult, Tier
from realtime_adapter import BACKBONE, Cosmos3RealtimeAdapter
def verify_artifact(directory):
    manifest = json.loads((directory/'budget_manifest.json').read_text())
    assert manifest['kind'] == 'untrained_original_nano_cost_only'
    for item in manifest['files']:
        path = directory/item['relative_path']
        assert path.stat().st_size == item['bytes']
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
        assert value.hexdigest() == item['sha256'], path
register(BACKBONE, Cosmos3RealtimeAdapter)
report = dict(ok=False, category='SCREEN', task_quality_certified=False, attention=a.attention,
              checkpoint=str(a.checkpoint), trained_student=False, experiment='original Nano SDE1/SDE2 CFG1 cost only', torch=torch.__version__, cuda=torch.version.cuda,
              cudnn=torch.backends.cudnn.version(), fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
              benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), calls=[])
report['swiglu_requested']=a.swiglu
report['cache_mode'] = a.cache_mode
report['warmup_requests']=a.warmup
report['measured_requests']=a.iterations
report['cache_helper_sha256'] = hashlib.sha256(Path(__file__).with_name('persistent_cache.py').read_bytes()).hexdigest()
cache = None
api = None
try:
    verify_artifact(a.checkpoint)
    report['qualification_only'] = a.allow_unqualified
    report['checkpoint_manifest_sha256'] = hashlib.sha256((a.checkpoint/'budget_manifest.json').read_bytes()).hexdigest()
    with np.load(a.fixture, allow_pickle=True) as data:
        frames = [np.asarray(Image.open(io.BytesIO(bytes(row[0]))).convert('RGB').resize((640, 540))) for row in data['jpeg_0'][:12]]
    from runtime_assets import audit_vae_load
    with audit_vae_load() as assets:
        api = Runtime.from_pretrained(a.checkpoint, precision='native', tier_ceiling='numeric' if a.attention == 'cudnn' else 'bitexact', placement='in_process', strict=not a.allow_unqualified)
        api.reset(prompt='pick up the object')
    report['external_runtime_assets'] = assets
    service = api._backend._impl._native_loop._service
    if a.swiglu:
        assert sum(name.endswith('.swiglu') for name in service._ifl_exact_pointwise['installed']) == 72
    initial_stats = api._backend._impl.backend_stats()
    sigmas = initial_stats['sampling']['sigmas']
    steps = len(sigmas) - 1
    guidance = service.cfg.guidance
    assert sigmas in ([1., 0.], [1., .5, 0.]) and guidance == 1.
    from prompt_contract import verify_plain_prompt
    report['prompt_contract'] = verify_plain_prompt(service)
    assert len(service.model.net.language_model.model.layers) == 36
    report['format_prompt_as_json'] = False
    branches_per_callback = 1 if guidance == 1. else 2
    report['declared_execution'] = dict(sigmas=sigmas, steps=steps, guidance=guidance,
        branches_per_callback=branches_per_callback, action_padding='zero')
    recorded_clocks = []
    record_clock = True
    velocity_owned = '_get_velocity' in vars(service.model)
    original_velocity = service.model._get_velocity
    def clocked_velocity(**kwargs):
        if record_clock:
            recorded_clocks.extend(kwargs['timestep'].detach().cpu().reshape(-1).tolist())
        return original_velocity(**kwargs)
    service.model._get_velocity = clocked_velocity
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
        assert len(owners) in (28, 36) and all(owner.dispatch_attention_fn is mot.dispatch_attention for owner in owners)
        service._ifl_numeric_attention = NumericAttention(mot, owners)
        api.plan.results.append(PassResult('experimental_student_cudnn', True, Tier.NUMERIC,
            'Exported-student qualification screen; no quality certificate'))
    if a.cache_mode != 'off':
        import cosmos_framework
        from cosmos3_iwm.conditioning_cache import SOURCE_HASHES
        from persistent_cache import PersistentConditioningCache
        root = Path(cosmos_framework.__file__).parent
        for name, digest in SOURCE_HASHES.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
        assert steps == 1 and guidance == 1.
        cache = PersistentConditioningCache(service.model, verify=a.cache_mode == 'verify')
        generate, velocity = service.model.generate_samples_from_batch, service.model._get_velocity
        cache.patch(service.model, 'generate_samples_from_batch', lambda *args, **kw:cache.generate(generate,*args,**kw))
        cache.patch(service.model, '_get_velocity', lambda **kw:cache.velocity(velocity,**kw))
        api.plan.results.append(PassResult('experimental_persistent_text',True,Tier.NUMERIC,
            'Fixed-geometry original Nano cross-request text cache diagnostic'))
    actions = []
    for i in range(a.warmup + a.iterations):
        others = subprocess.check_output(['/usr/sbin/nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
        assert not [pid for pid in others if int(pid) != os.getpid()], 'GPU contention'
        prompt = ['pick up the object', 'place the object down'][(i//3)%2]
        if i < a.warmup or i % 5 == 0:
            api.reset(prompt=prompt)
        torch.manual_seed(707+i); np.random.seed(707+i); random.seed(707+i)
        torch.cuda.synchronize(); start = time.perf_counter()
        action = np.asarray(api.predict(dict(image=frames[i%len(frames)].copy(), state=np.full(8, .01*(i%3), np.float32), prompt=prompt))['action'])
        torch.cuda.synchronize(); elapsed = 1000*(time.perf_counter()-start)
        assert action.shape == (32, 8) and np.isfinite(action).all()
        actions.append(action.copy())
        record_clock = False
        report['calls'].append(dict(i=i, phase='warmup' if i<a.warmup else 'measured', ms=elapsed))
        if cache is not None and a.cache_mode == 'reuse':
            counters={key:sum(g[key] for g in cache.graph_history) for key in ('captures','checks','replays')}
            report['calls'][-1]['graph_counters']=counters
            if i >= a.warmup:
                previous=report['calls'][-2]['graph_counters']
                assert counters['captures']==previous['captures'] and counters['checks']==previous['checks'], 'Graph preparation inside steady timing'
                assert counters['replays']-previous['replays']==36, 'Expected all36 layers to replay'
        print(json.dumps(report['calls'][-1]), flush=True)
    expected_clocks = [1000.*sigma for sigma in sigmas[:-1] for _ in range(branches_per_callback)]
    assert len(recorded_clocks) == len(expected_clocks)
    assert all(abs(actual-expected) <= float(np.spacing(np.float32(expected)))
               for actual, expected in zip(recorded_clocks, expected_clocks))
    report['first_request_native_branch_clocks'] = recorded_clocks
    report['clock_check'] = 'Complete declared SDE grid and literal CFG branches; at most one native FP32 ULP'
    stats = api._backend._impl.backend_stats()
    steps = len(stats['sampling']['sigmas']) - 1
    assert stats['velocity_evaluations'] == steps*len(actions) and stats['sampler_calls'] == len(actions)
    assert stats['action_padding'] == 'zero'
    assert stats['padding_projection_calls'] == len(actions)
    assert stats['padding_projected_velocity_branches'] == branches_per_callback*steps*len(actions)
    assert stats['padding_hooks_restored']
    if a.attention == 'cudnn':
        assert stats['numeric_attention']['eligible_python_calls'] > 0
    if cache is not None:
        report['persistent_cache'] = cache.report()
        assert not cache.disabled and not cache.stats['rejected']
        assert cache.stats['prefills'] < 16 and cache.stats['decodes'] > 0
        assert cache.stats['admitted_branches'] >= 2 and len(cache.published) == 2
        if a.cache_mode == 'reuse':
            history=cache.report()['graph_lifetime_stats']
            assert history and not any(g['rejected'] for g in history)
            assert sum(g['captures'] for g in history) >= 36
            assert sum(g['checks'] for g in history) >= 72
    report.update(backend_stats=stats, execution_policy=api.execution_policy,
                  p50_ms=statistics.median(c['ms'] for c in report['calls'][a.warmup:]),
                  p95_ms=float(np.percentile([c['ms'] for c in report['calls'][a.warmup:]],95)), ok=True)
    verify_artifact(a.checkpoint)
    np.savez_compressed(a.output.with_suffix('.npz'), actions=np.stack(actions))
    report['actions_sha256'] = hashlib.sha256(a.output.with_suffix('.npz').read_bytes()).hexdigest()
except Exception:
    report['error'] = traceback.format_exc()
    traceback.print_exc()
finally:
    report['sources'] = {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for name, m in list(sys.modules.items())
        if name.startswith(('instinctflash', 'instinct_compress', 'cosmos3_iwm', 'cosmos_framework', 'realtime_adapter', 'runtime_assets'))
        and getattr(m, '__file__', None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}
    if cache is not None:
        report['persistent_cache'] = cache.report()
        cache.close()
    if api is not None:
        if 'clocked_velocity' in globals():
            assert service.model._get_velocity is clocked_velocity
            if velocity_owned:
                service.model._get_velocity = original_velocity
            else:
                del service.model._get_velocity
        api.close()
    a.output.write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['ok'] else 1)
