"""Actual DreamZero checkpoint + native history, with selected Q/K/V FP8.

Recorded RoboTwin cameras and synthetic DROID states test execution, not task
quality. The same original grid/mask and processors serve native and FP8 arms.
"""
import hashlib
import random
import re
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

os.environ['DREAMZERO_ROOT'] = '/home/guanming/thorcol/dreamzero'
os.environ['DYNAMIC_CACHE_SCHEDULE'] = 'false'
os.environ['NUM_DIT_STEPS'] = '8'
os.environ['ENABLE_TENSORRT'] = 'false'
from dreamzero_iwm.adapter import DreamZeroAdapter
from instinctflash.runtime.dreamzero_fp8 import install_dreamzero_fp8
from instinctflash.runtime.torch_fp8_linear import ThorFP8Linear

root = Path(__file__).parent
# The previous native FP32 construction exhausted shared device memory. Stop
# this owned diagnostic before available memory falls below a 12-GiB reserve.
import threading
import time
import signal
def memory_guard():
    minimum = None
    while True:
        info = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        available_kib = int(info['MemAvailable'].split()[0])
        minimum = available_kib if minimum is None else min(minimum, available_kib)
        if available_kib < 12*1024*1024:
            (root/'dreamzero-seeded_construction-memory-stop.json').write_text(json.dumps({
                'reason':'shared-memory reserve reached', 'available_kib':available_kib,
                'minimum_available_kib':minimum, 'pid':os.getpid()})+'\n')
            os.kill(os.getpid(),signal.SIGTERM)
            return
        time.sleep(.1)
precision = sys.argv[1]
assert precision in ('native', 'fp8')
run_id = sys.argv[2]
assert re.fullmatch(r'[a-zA-Z0-9_-]+', run_id)
root = root / f'dreamzero-seeded-construction-{precision}-{run_id}'
root.mkdir(exist_ok=False)
threading.Thread(target=memory_guard,daemon=True).start()
checkpoint = '/home/guanming/.cache/huggingface/hub/models--GEAR-Dreams--DreamZero-DROID/snapshots/96ad344138c66e82536422432ad742f015784942'
from instinctflash import Runtime
def source_hashes():
    import instinctflash
    roots = [Path(instinctflash.__file__).resolve().parent.parent,
             Path(os.environ['DREAMZERO_ROOT'])]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for base in roots for path in sorted(base.rglob('*.py'))
            if not any(part in ('.git', '.venv', '__pycache__') for part in path.parts)}
source_before = source_hashes()
construction_seed = 9173
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
random.seed(construction_seed)
np.random.seed(construction_seed)
torch.manual_seed(construction_seed)
construction_rng = {'cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    'cuda': [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()]}
runtime = Runtime.from_pretrained(
    'GEAR-Dreams/DreamZero-DROID', revision='96ad344138c66e82536422432ad742f015784942',
    precision=precision, device='cuda:0', placement='in_process')
runtime.reset(prompt='pick up the object')
loop = runtime._backend._loop if precision == 'fp8' else runtime._backend._impl
head = loop._wrapper._policy.trained_model.action_head
load_receipt = loop.backend_stats['loading']
# Record all persistent model state, not only the DiT tensors checked by loading.
# Hash one tensor at a time to bound host memory use; this is not latency evidence.
model_state = {}
for name, tensor in loop._wrapper._policy.trained_model.state_dict().items():
    value = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    model_state[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                         'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}
(root/f'dreamzero-{precision}-seeded_construction-state.json').write_text(json.dumps({
    'construction_seed': construction_seed, 'construction_rng': construction_rng,
    'model_state': model_state,
    'scope': 'Full persistent state audit after construction; does not cover nonpersistent caches.'}, indent=2)+'\n')
del value

assert load_receipt['verified_dit_tensors'] == 1317
assert loop.declaration()['precision'] == precision
before_frame = head.current_start_frame
try:
    runtime.predict({}, executed_action=[1])
except ValueError as error:
    assert 'executed-action' in str(error)
else:
    raise AssertionError('Unsupported feedback was silently accepted')
assert head.current_start_frame == before_frame
original_mask = list(head.dit_step_mask)
assert head.num_inference_steps == 16 and sum(original_mask) == 8
receipt = loop._fp8_recipe
assert list(head.dit_step_mask) == original_mask
counts = {}
handles = []
for name, module in head.model.named_modules():
    if isinstance(module, ThorFP8Linear):
        counts[name] = 0
        def record(mod, args, out, name=name):
            if args[0].numel(): counts[name] += 1
        handles.append(module.register_forward_hook(record))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
data = np.load('/home/guanming/thor_va_engine/va_eval_obs.npz', allow_pickle=True)
keys = ('observation/exterior_image_0_left', 'observation/exterior_image_1_left',
        'observation/wrist_image_left')
def decode(values):
    return [np.asarray(Image.open(io.BytesIO(bytes(v))).convert('RGB')) for v in values]
frames = [decode(data['frame0_0'])] + [decode(x) for x in data['jpeg_0'][:8]]
outputs, positions = [], []
for episode in range(2):
    runtime.reset(prompt='pick up the object')
    assert head.current_start_frame == 0
    for cycle, indices in enumerate(([0], list(range(1, 5)), list(range(5, 9)))):
        torch.manual_seed(1300+cycle)
        np.random.seed(1300+cycle)
        obs = {key:np.stack([frames[i][view] for i in indices]) for view,key in enumerate(keys)}
        obs['observation/joint_position'] = np.zeros(7, dtype=np.float32)
        obs['observation/gripper_position'] = np.zeros(1, dtype=np.float32)
        action = np.asarray(runtime.predict(obs)['action'])
        assert action.shape == (24,8) and np.isfinite(action).all(), action.shape
        outputs.append(action.copy())
        positions.append(int(head.current_start_frame))
        print('episode',episode,'cycle',cycle,'frame',positions[-1],flush=True)
assert np.array_equal(np.stack(outputs[:3]), np.stack(outputs[3:]))
assert positions[:3] == positions[3:] and 0 < positions[0] < positions[1] < positions[2]
if precision == 'fp8': assert counts and min(counts.values()) > 0
else: assert not counts
for handle in handles: handle.remove()
owned_path = Path(loop._checkpoint_view.name)
assert owned_path.is_dir()
runtime.close()
assert not owned_path.exists()
np.savez(root/f'dreamzero-{precision}-seeded_construction-actions.npz', actions=np.stack(outputs))
source_after = source_hashes()
assert source_before == source_after, 'Python sources changed during diagnostic'
(root/'source.json').write_text(json.dumps(source_before,sort_keys=True,indent=2)+'\n')
report = dict(source_sha256=hashlib.sha256((root/'source.json').read_bytes()).hexdigest(), construction_seed=construction_seed, run_id=run_id, pid=os.getpid(),
              probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              input_sha256=hashlib.sha256(Path('/home/guanming/thor_va_engine/va_eval_obs.npz').read_bytes()).hexdigest(),
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              torch_version=str(torch.__version__),
              device_name=torch.cuda.get_device_name(),
              matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
              cudnn_tf32=torch.backends.cudnn.allow_tf32,
              cudnn_benchmark=torch.backends.cudnn.benchmark,
              state_sha256=hashlib.sha256((root/f'dreamzero-{precision}-seeded_construction-state.json').read_bytes()).hexdigest(),
              actions_sha256=hashlib.sha256((root/f'dreamzero-{precision}-seeded_construction-actions.npz').read_bytes()).hexdigest(),
              feedback_refused_before_prediction=True, owned_checkpoint_view_released=True, load_receipt=load_receipt,ok=True, precision=precision, checkpoint=checkpoint, recipe=receipt,
              executed_fp8_projections=counts, frame_positions=positions, repeated_episodes_equal=True,
              scheduler_steps=16, dit_step_mask=original_mask,
              scope='Actual public Runtime native/FP8 dispatch with native checkpoint processors and owned direct-BF16 view; recorded RoboTwin cameras and synthetic DROID state. Not speed or task-quality certification.')
(root/f'dreamzero-{precision}-seeded_construction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
