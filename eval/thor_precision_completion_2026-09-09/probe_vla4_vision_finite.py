"""Find the first nonfinite vision GEMM on retained native-processed patches."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from flash_rt.frontends.torch.vla4b_thor import Vla4bTorchFrontendThor
import flash_rt.flash_rt_kernels as fvk
root=Path.home()/'ifl_eval/thor_precision_completion_20260909'
out=Path(sys.argv[1])
if out.exists():raise RuntimeError('refusing overwrite')
checkpoint=Path.home()/'.cache/huggingface/hub/models--robbyant--lingbot-vla-4b-posttrain-robotwin/snapshots/fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e'
fe=Vla4bTorchFrontendThor(checkpoint,use_cuda_graph=False)
fe.set_prompt([1,2,3])
arrays=np.load(root/'results/vla4-engine-loop-vision-kv-diagnostic.debug.npz')
fe.stage_inputs(arrays['patches'][0],np.zeros(75,dtype=np.float16),np.zeros((50,75),dtype=np.float16))
pointers={}
def collect(name,value):
    if torch.is_tensor(value):pointers[value.data_ptr()]=(name,value)
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):collect(f'{name}[{i}]',v)
for name,value in vars(fe).items():collect(name,value)
records=[];original=fvk.gmm_fp16

def summary(ptr,n):
    name,t=pointers[ptr];x=t.reshape(-1)[:n].float()
    return dict(name=name,finite=bool(torch.isfinite(x).all()),min=float(x.min()),max=float(x.max()))
def traced(ctx,a,b,c,m,n,k,beta,stream):
    original(ctx,a,b,c,m,n,k,beta,stream)
    torch.cuda.synchronize()
    record=dict(call=len(records),shape=[m,n,k],a=summary(a,m*k),b=summary(b,k*n),c=summary(c,m*n))
    records.append(record)
    if not record['c']['finite']:
        out.write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record,indent=2),flush=True)
        raise RuntimeError('nonfinite vision GEMM')
fvk.gmm_fp16=traced
try:
    fe._run_vis(0)
    out.write_text(json.dumps(records,indent=2)+'\n')
    print('All vision GEMMs finite')
finally:
    fvk.gmm_fp16=original
