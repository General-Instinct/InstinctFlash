"""Real DROID request phase profiling; timings here are diagnostic, not speed claims."""
import argparse
import collections
import functools
import json
from pathlib import Path
import time
import sys
import numpy as np
import torch
from instinctflash import Runtime

p = argparse.ArgumentParser()
p.add_argument('family', choices=['edge', 'nano'])
p.add_argument('output', type=Path)
p.add_argument('--upstream-compiled', action='store_true', help='Profile the upstream compiled BF16 reference')
a = p.parse_args()
assert not a.output.exists()
assert torch.cuda.get_device_capability() == (11, 0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
api = Runtime.from_pretrained('nvidia/Cosmos3-' + ('Edge' if a.family == 'edge' else 'Nano') + '-Policy-DROID', precision='native', tier_ceiling='bitexact')
if a.upstream_compiled:
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService
    original_init = RobolabPolicyService.__init__
    original_setup = RobolabPolicyService._build_setup_args
    def initialize(self, args):
        self._build_setup_args = lambda args: original_setup(self, args).model_copy(update={
            'guardrails': False, 'use_torch_compile': True, 'diffusion_cache': False})
        original_init(self, args)
    RobolabPolicyService.__init__ = initialize
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native_total_2026-09-10'))
    from upstream import build
    try:
        api = build(a.family, api)
    finally:
        RobolabPolicyService.__init__ = original_init
api.reset(prompt='pick up the object')
service = api._backend._impl._service
rows = []
active = False

def wrap(obj, name, label):
    fn = getattr(obj, name)
    @functools.wraps(fn)
    def measured(*args, **kwargs):
        if not active:
            return fn(*args, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.profiler.record_function(label):
            result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        rows.append({'phase': label, 'ms': 1000 * (time.perf_counter() - start)})
        return result
    setattr(obj, name, measured)

for name in ('_build_sample',):
    wrap(service, name, 'service.' + name)
for name in ('generate_samples_from_batch', '_prepare_inference_data', 'get_data_and_condition', 'encode', '_get_velocity', 'denoise'):
    wrap(service.model, name, 'model.' + name)
obs = {'image': np.random.default_rng(55).integers(0, 256, (540, 640, 3), dtype=np.uint8), 'state': np.zeros(8, np.float32)}
for _ in range(2):
    api.predict(obs)
active = True
for _ in range(2):
    api.predict(obs)
active = False
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
    api.predict(obs)
    torch.cuda.synchronize()
a.output.parent.mkdir(parents=True, exist_ok=True)
prof.export_chrome_trace(str(a.output.with_suffix('.trace.json')))
keys = sorted(prof.key_averages(group_by_input_shape=True), key=lambda x: x.self_device_time_total, reverse=True)
report = {'family': a.family, 'upstream_compiled': a.upstream_compiled, 'phases': rows, 'kernels': [{'name': x.key, 'count': x.count, 'cuda_us': x.self_device_time_total, 'cpu_us': x.self_cpu_time_total, 'shapes': x.input_shapes} for x in keys[:70]], 'stats': api._backend._impl.backend_stats()}
a.output.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str), flush=True)
api.close()
