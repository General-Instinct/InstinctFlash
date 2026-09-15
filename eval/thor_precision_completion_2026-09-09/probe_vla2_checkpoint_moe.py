"""Build every VLA2 MoE layer from checkpoint weights and execute real FP8 kernels."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.vla2.moe_engine import Vla2MoeEngine

out=Path(sys.argv[1])
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
checkpoint=Path.home()/'.cache/huggingface/hub/models--robbyant--lingbot-vla-v2-6b-robotwin/snapshots/0451855729ec904f970600e0aec8b84661423afe/checkpoints/global_step_50000/hf_ckpt'
t0=time.perf_counter()
moe=Vla2MoeEngine.from_checkpoint(fvk,checkpoint,mode='batched',schedule='batched',router_source='fp16')
load_s=time.perf_counter()-t0
assert moe.L==36
assert all(x.dtype==torch.float8_e4m3fn for x in moe._gu_fp8+moe._dn_fp8)
assert all(bool(torch.isfinite(x).all()) for x in [moe._gu_scale,moe._dn_scale])
torch.manual_seed(817)
x=torch.randn(51,768,device='cuda',dtype=torch.float16)*0.1
scale=float(x.abs().max())/448.0
x8=(x.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn)
moe._own_ffn.fill_(scale);moe.bind_router_fp16(x.data_ptr())
y=torch.empty_like(x);outputs=[];deltas=[]
for layer in range(36):
    moe.calibrate=True
    moe.routed_moe_fn(layer,0,x8.data_ptr(),y.data_ptr(),0)
    torch.cuda.synchronize();a=y.clone()
    assert bool(torch.isfinite(a).all()),f'nonfinite layer {layer}'
    moe.calibrate=False
    moe.routed_moe_fn(layer,0,x8.data_ptr(),y.data_ptr(),0)
    torch.cuda.synchronize()
    delta=float((a.float()-y.float()).abs().max());deltas.append(delta)
    # Dynamic quantization and static quantization kernels may round differently;
    # record that delta, then require deterministic static execution itself.
    b=y.clone();moe.routed_moe_fn(layer,0,x8.data_ptr(),y.data_ptr(),0)
    torch.cuda.synchronize();assert torch.equal(b,y),f'nonrepeatable layer {layer}'
    outputs.append(y.float().cpu().numpy().copy())
np.savez(out.with_suffix('.npz'),outputs=np.stack(outputs),down_scales=moe.dn_act.cpu().numpy())
result=dict(layers=36,source='checkpoint safetensors; no prepacked/calibration artifact loaded',
            weight_grid='batched shared scale, direct from source weights',router='fp16 input, fp32 weights',
            load_seconds=load_s,all_outputs_finite=True,static_repeat_byte_equal=True,
            dynamic_vs_static_maxabs=deltas,actions_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
            engine_source_sha256=hashlib.sha256(Path(inspect.getfile(Vla2MoeEngine)).read_bytes()).hexdigest(),
            scope='Real MoE kernels on synthetic hidden states; not complete VLA2 Runtime or task-quality evidence')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
