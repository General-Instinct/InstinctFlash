"""VLA4 staged three-arm comparison; synthetic inputs, no task-quality claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--arm', choices=['stock', 'capture', 'engine8', 'stock_vendor', 'capture_vendor'], required=True)
p.add_argument('--repeat', type=int, required=True)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
out = a.root / 'results' / f'vla4.{a.arm}.{a.repeat}.json'
if out.exists():
    raise RuntimeError('refusing overwrite')
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1701+a.repeat)
np.random.seed(1701+a.repeat)
data = torch.load(a.root / 'vla4-inputs.pt', weights_only=False)
samples = data['samples']
checkpoint = Path.home() / '.cache/huggingface/hub/models--robbyant--lingbot-vla-4b-posttrain-robotwin/snapshots/fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e'
start = time.perf_counter()
if a.arm == 'engine8':
    from flash_rt.frontends.torch.vla4b_thor import Vla4bTorchFrontendThor
    fe = Vla4bTorchFrontendThor(checkpoint)
    fe.set_prompt(samples[0]['lang_tokens'][:samples[0]['n_real']].numpy())
    def predict(s):
        return fe.infer_staged(s['pixel_values'].numpy(), s['state'].numpy(), s['noise'].numpy())['actions'].copy()
    predict(samples[0])  # Explicit calibration sample excluded from evaluation.
else:
    import sys
    native = a.root / 'native-vla4'
    sys.path.insert(0, str(native))
    os.environ['QWEN25_PATH'] = str(Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3')
    from deploy.lingbot_vla_policy import LingbotVLAServer
    server = LingbotVLAServer(str(checkpoint), use_length=25, robot_norm_path=str(native / 'assets/norm_stats/robotwin_50.json'), num_denoising_step=10)
    model = server.vla.model
    if a.arm.startswith('capture'):
        from lingbot_vla_iwm.static_capture import install_static_capture
        den = install_static_capture(model)
    def predict(s):
        with torch.no_grad():
            return model.sample_actions(
                s['pixel_values'].to('cuda', dtype=torch.bfloat16),
                torch.ones(3, dtype=torch.bool, device='cuda'),
                s['lang_tokens'][None].to('cuda'), s['lang_masks'][None].to('cuda'),
                s['state'][None].to('cuda', dtype=torch.bfloat16),
                noise=s['noise'][None].to('cuda', dtype=torch.bfloat16), num_steps=10,
            )[0].float().cpu().numpy().copy()
# The upstream policy constructor sets matmul precision to "high". Reassert
# the declared comparison settings after loading, before any measured/warmup call.
if not a.arm.endswith('_vendor'):
    torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1701+a.repeat)
np.random.seed(1701+a.repeat)
torch.cuda.synchronize()
load_s = time.perf_counter()-start
start = time.perf_counter()
for i in range(8):
    predict(samples[1+i%11])
torch.cuda.synchronize()
warmup_s = time.perf_counter()-start
torch.cuda.reset_peak_memory_stats()
lat, actions = [], []
for i in range(128):
    torch.cuda.synchronize()
    start = time.perf_counter()
    action = predict(samples[1+i%11])
    torch.cuda.synchronize()
    lat.append((time.perf_counter()-start)*1000)
    if not np.isfinite(action).all():
        raise RuntimeError('nonfinite actions')
    actions.append(action)
null = [predict(samples[1]) for _ in range(3)]
np.savez(out.with_suffix('.npz'), actions=np.stack(actions), null=np.stack(null))
result = dict(arm=a.arm, repeat=a.repeat, model='vla4', chunk=50, returned_scope='normalized 50x75 before controller slicing to 25x14',
              steps=10, views=3, synthetic=True,
              scope='CPU staged patches/state/actual-noise to CPU normalized actions; vision and prefill recomputed each call; excludes raw-image processing and controller decoding',
              common_inputs='All float inputs rounded to BF16 then stored FP32, exactly representable in both BF16 and FP16',
              calibration_sample=0, evaluation_samples=list(range(1,12)),
              input_sha256=hashlib.sha256((a.root/'vla4-inputs.pt').read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              actions_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
              torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
              numeric_environment=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic),
              load_calibrate_s=load_s, warmup_s=warmup_s, iterations=128,
              captured=bool(fe.graph_captured) if a.arm=='engine8' else bool(a.arm.startswith('capture') and den._graph is not None),
              samples_ms=lat, latency_ms={f'p{q}':float(np.percentile(lat,q)) for q in (50,95,99)},
              torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              memory_caveat='PyTorch allocator only; external engine allocations excluded')
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('arm','repeat','latency_ms','captured')}), flush=True)
