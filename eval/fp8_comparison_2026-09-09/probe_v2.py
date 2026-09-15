"""V2 staged native/capture/full-engine comparison on two recorded real inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--arm',choices=['stock','capture','engine8','stock_vendor','capture_vendor'],required=True)
p.add_argument('--repeat',type=int,required=True)
a=p.parse_args()
out=a.root/'results'/f'v2.{a.arm}.{a.repeat}.json'
if out.exists():raise RuntimeError('refusing overwrite')
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.manual_seed(1701+a.repeat)
np.random.seed(1701+a.repeat)
import p1_backbone_harness as h
# Helpers are frozen copies with their path declarations bound to this campaign.
ds=[h.load_dump(i) for i in (0,8)]
assert torch.equal(ds[0]['eng_ids'],ds[1]['eng_ids'])
for d in ds:
    for key in ('patches','state55'):
        d[key]=d[key].to(torch.bfloat16).float()
rng=np.random.default_rng(2917)
noise=torch.from_numpy(rng.standard_normal((136,50,55)).astype(np.float32)).to(torch.bfloat16).float().numpy()
start=time.perf_counter()
verdicts=[]
if a.arm=='engine8':
    import flash_rt.flash_rt_kernels as fvk
    from flash_rt.models.vla2.moe_engine import Vla2MoeEngine
    fr=h.build_frontend()
    fr.lm_prefill_precision='fp16'
    fr.load_lm_fp16_stack(h.m2h.MODEL_PATH)
    moe=torch.load(h.ART/'moe.pt',map_location='cpu',weights_only=False)
    eng=Vla2MoeEngine.from_artifacts(fvk,moe,mode='batched',router_source='fp16')
    fr._routed_moe_fn=eng.routed_moe_fn
    fr._ffn_xn_fp16=True
    fr.set_prompt(ds[0]['eng_ids'].numpy())
    h.post_prompt(fr)
    fr._exp_act_scales.copy_(h.calib()['exp_act_scales'].float().cuda())
    eng.bind_ffn_slots(fr._exp_act_scales.data_ptr()+2*4,16)
    eng.set_down_act_scales(h.calib()['moe_down_act_amax'].float())
    eng.bind_router_fp16(fr._s_xn.data_ptr())
    def stage(d,n):
        fr._patches.copy_(d['patches'].reshape(768,1536).to('cuda',dtype=torch.float16))
        fr._state_in.copy_(d['state55'].reshape(1,55).to('cuda',dtype=torch.float16))
        fr._x_t.copy_(torch.from_numpy(n).to('cuda',dtype=torch.float16))
    with torch.no_grad():
        stage(ds[0],noise[0])
        for _ in range(2):
            fr._run_vis(0);fr._run_lm_prefill_only(0);fr._run_expert_only(0)
        torch.cuda.synchronize()
        stream=torch.cuda.Stream()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph,stream=stream):
                fr._run_vis(stream.cuda_stream)
                fr._run_lm_prefill_only(stream.cuda_stream)
                fr._run_expert_only(stream.cuda_stream)
        torch.cuda.synchronize()
    def predict(d,n):
        stage(d,n);graph.replay()
        return fr._x_t.float().cpu().numpy().copy()
else:
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    server=LingbotVLAv2Server(h.m2h.MODEL_PATH,use_length=50,chunk_ret=True,use_bf16=True,use_fp32=False,use_compile=False)
    server.reset('robotwin')
    model=server.vla.model
    # Use the native processor to establish masks/grid; cross-check frozen pixels/IDs.
    raw=h.m2h.load_eval_obs()
    preps=[]
    for i,d in zip((0,8),ds):
        prep=server._prepare_model_input(raw[i])
        assert torch.equal(prep['images'].cpu().to(torch.bfloat16).float(),d['patches'])
        mask=prep['lang_masks'].cpu().bool()
        assert torch.equal(prep['lang_tokens'].cpu()[mask],d['eng_ids'])
        preps.append(prep)
    if a.arm.startswith('capture'):
        from lingbot_vla_v2_iwm.static_capture import install_static_capture
        from lingbot_vla_v2_iwm.prefix_capture import install_prefix_capture
        from lingbot_vla_v2_iwm.adapter import _release_prefix_graphs_on_fail
        den=install_static_capture(model,on_self_check=_release_prefix_graphs_on_fail(verdicts.append,server))
        server._instinctflash_prefix_capture=install_prefix_capture(model)
    def predict(d,n):
        index=0 if d is ds[0] else 1
        prep=preps[index]
        with torch.no_grad():
            return model.sample_actions(
                d['patches'].to('cuda',dtype=torch.bfloat16),prep['img_masks'].to('cuda'),
                prep['lang_tokens'][None].to('cuda'),prep['lang_masks'][None].to('cuda'),
                d['state55'][None].to('cuda',dtype=torch.bfloat16),
                noise=torch.from_numpy(n)[None].to('cuda',dtype=torch.bfloat16),
                image_grid_thw=prep['image_grid_thw'].to('cuda',dtype=torch.long),
            )[0].float().cpu().numpy().copy()
if not a.arm.endswith('_vendor'):
    torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.manual_seed(1701+a.repeat)
np.random.seed(1701+a.repeat)
torch.cuda.synchronize()
load_s=time.perf_counter()-start
start=time.perf_counter()
for i in range(8):predict(ds[i%2],noise[i])
torch.cuda.synchronize()
warmup_s=time.perf_counter()-start
torch.cuda.reset_peak_memory_stats()
lat,actions=[],[]
for i in range(128):
    torch.cuda.synchronize();start=time.perf_counter()
    value=predict(ds[i%2],noise[8+i])
    torch.cuda.synchronize();lat.append((time.perf_counter()-start)*1000)
    if not np.isfinite(value).all():raise RuntimeError('nonfinite action')
    actions.append(value)
null=[predict(ds[0],noise[8]) for _ in range(3)]
np.savez(out.with_suffix('.npz'),actions=np.stack(actions),null=np.stack(null),noises=noise[8:])
prefix=None if not a.arm.startswith('capture') else getattr(server,'_instinctflash_prefix_capture',None)
result=dict(arm=a.arm,repeat=a.repeat,model='v2',chunk=50,steps=10,views=3,synthetic=False,
    scope='CPU staged patches/state/actual-noise to CPU normalized 50x55 actions; vision and prefill every call; excludes raw camera preprocessing and controller decoding',
    evaluation_recorded_indices=[0,8],calibration='Retained historical repack and calibration artifacts; no held-out quality claim',
    load_calibrate_s=load_s,warmup_s=warmup_s,iterations=128,
    captured=bool(a.arm=='engine8' or (a.arm.startswith('capture') and den.graph is not None)),
    vision_graph=bool(prefix and prefix.vision.graph is not None),prefill_graph=bool(prefix and prefix.prefill.graph is not None),self_checks=verdicts,
    torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
    numeric_environment=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic),
    input_sha256={str(i):hashlib.sha256((h.DUMPS/f'obs{i}.pt').read_bytes()).hexdigest() for i in (0,8)},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    actions_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
    latency_ms={f'p{q}':float(np.percentile(lat,q)) for q in (50,95,99)},samples_ms=lat,
    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),memory_caveat='PyTorch allocator only; external engine allocations excluded')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('arm','repeat','latency_ms','captured')}),flush=True)
