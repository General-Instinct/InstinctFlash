"""Diagnostic native Runtime profile, with no model-forward monkeypatches."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
from instinctflash import Runtime

p = argparse.ArgumentParser()
p.add_argument('family', choices=('edge', 'nano'))
p.add_argument('output', type=Path)
p.add_argument('--fixture', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
assert torch.cuda.get_device_capability() == (11, 0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
others = [int(x) for x in subprocess.check_output(['/usr/sbin/nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines() if x.strip() and int(x) != os.getpid()]
assert not others, others
with np.load(a.fixture, allow_pickle=True) as data:
    frame = np.asarray(Image.open(io.BytesIO(bytes(data['jpeg_0'][0][0]))).convert('RGB').resize((640, 540)))
api = Runtime.from_pretrained(f'nvidia/Cosmos3-{a.family.title()}-Policy-DROID', precision='native', tier_ceiling='bitexact')
api.reset(prompt='pick up the object')
obs = {'image': frame, 'state': np.zeros(8, np.float32), 'prompt': 'pick up the object'}
def predict():
    torch.manual_seed(707); np.random.seed(707); random.seed(707)
    return np.asarray(api.predict(obs)['action']).copy()
for _ in range(6):
    predict()
latencies = []
for _ in range(2):
    torch.cuda.synchronize()
    start = time.perf_counter()
    reference = predict()
    torch.cuda.synchronize()
    latencies.append((time.perf_counter() - start) * 1000)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
    action = predict()
    torch.cuda.synchronize()
a.output.parent.mkdir(parents=True, exist_ok=True)
prof.export_chrome_trace(str(a.output.with_suffix('.trace.json')))
events = []
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        events.append({'name': e.name, 'us': e.device_time_total})
report = {'family': a.family, 'diagnostic_only': True, 'unprofiled_ms': latencies,
          'profile_action_byte_equal': reference.tobytes() == action.tobytes(),
          'finite_actions': bool(np.isfinite(action).all()), 'cuda_events': events,
          'stats': api._backend._impl.backend_stats(), 'execution_policy': api.execution_policy,
          'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
          'fixture_sha256': hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'sources': {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
              for name, m in list(sys.modules.items()) if name.startswith(('instinctflash', 'cosmos3_iwm', 'cosmos_framework', 'natten'))
              and getattr(m, '__file__', None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}}
a.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('sources','cuda_events','stats')}), flush=True)
api.close()
