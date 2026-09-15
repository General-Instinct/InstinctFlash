"""Thor execution check for checkpoint-sized pi05 normalized action chunks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

p=argparse.ArgumentParser()
p.add_argument('--chunk',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise RuntimeError('refusing to overwrite evidence')
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
home=Path.home()
checkpoint=home/'.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/8e174154ef5f6c60a8da12ae99c303d8963138c1'
frames=np.load(home/'ifl/t3_assets/calib_obs.npz')
obs=[dict(image=frames['image'][i],wrist_image=frames['wrist_image'][i]) for i in range(5)]
ids=json.loads((home/'ifl/t3_assets/thor_assets.json').read_text())['synth_ids']['48']
fe=Pi05TorchFrontendThor(str(checkpoint),num_views=2,use_fp8=True,autotune=0,
                         action_chunk=a.chunk,action_output='normalized')
assert not hasattr(fe,'norm_stats'),'normalized engine must delegate decoding to checkpoint processor'
fe.set_prompt(ids)
fe.calibrate(obs[:3],percentile=99.9)
outputs=[];times=[]
for i in range(8):
    np.random.seed(813)
    torch.cuda.synchronize();start=time.perf_counter()
    value=fe.infer(obs[3+i%2])['actions']
    torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    assert value.shape==(a.chunk,32),value.shape
    assert np.isfinite(value).all()
    outputs.append(value)
assert all(np.array_equal(outputs[i],outputs[i+2]) for i in range(2,6))
assert not np.array_equal(outputs[-1],outputs[-2]),'vision refresh did not change outputs'
assert fe.graph_captured
np.savez(a.output.with_suffix('.npz'),actions=np.stack(outputs))
result=dict(chunk=a.chunk,shape=list(outputs[-1].shape),steps=10,captured=fe.graph_captured,
    output='normalized full 32-dimensional model output; native decoding not yet integrated',
    repeated_fixed_inputs_equal=True,changed_observation_changes_actions=True,
    samples_ms=times,p50_ms=float(np.median(times[2:])),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    frontend_sha256=hashlib.sha256(Path(__import__('inspect').getfile(Pi05TorchFrontendThor)).read_bytes()).hexdigest(),
    actions_sha256=hashlib.sha256(a.output.with_suffix('.npz').read_bytes()).hexdigest(),
    scope='Geometry/replay execution test on two recorded inputs, not quality certification or a latency sweep')
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
