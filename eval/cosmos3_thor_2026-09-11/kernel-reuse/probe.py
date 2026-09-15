"""Synthetic BF16 contract/timing screen of unchanged engine CUDA kernels.
Run under /tmp/thor_gpu.lock; not full-model qualification.
"""
import ctypes as c
import hashlib
import json
from pathlib import Path
import sys
import torch

assert torch.cuda.get_device_capability() == (11, 0)
torch.manual_seed(707)
root=Path(__file__).resolve().parent
lib=c.CDLL(str(root/'kernels.so'))
P=c.c_void_p; I=c.c_int
lib.probe_norm.argtypes=[P,P,P,I,I,c.c_float,P]
lib.probe_activation.argtypes=[P,P,P,I,P]
lib.probe_swiglu.argtypes=[P,P,P,I,P]
lib.probe_residual.argtypes=[P,P,I,P]
lib.probe_gate.argtypes=[P,P,P,I,P]
for name in ['norm','activation','residual','gate','swiglu']: getattr(lib,'probe_'+name).restype=None
stream=torch.cuda.current_stream().cuda_stream
rows=[]
def check(name, a, b):
    rows.append({'name':name,'shape':list(a.shape),'finite':bool(torch.isfinite(a).all()),
                 'byte_equal':torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),
                 'max_abs':float((a.float()-b.float()).abs().max()),
                 'mismatched_values':int((a!=b).sum())})
def timing(fn):
    for _ in range(10):fn()
    torch.cuda.synchronize()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(100):fn()
    b.record();b.synchronize()
    return a.elapsed_time(b)*10 # microseconds / call
for s,d in [(32,128),(1024,128),(1024,2048),(1024,4096)]:
    x=torch.randn(s,d,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(d,device='cuda',dtype=torch.bfloat16)
    y=torch.empty_like(x); r=torch.randn_like(x); g=torch.randn_like(x)
    def run_norm():lib.probe_norm(x.data_ptr(),w.data_ptr(),y.data_ptr(),s,d,1e-6,stream)
    run_norm()
    z=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
    check('rms_nemotron',y,(z*w).to(x.dtype));check('rms_qwen',y,z.to(x.dtype)*w)
    rows[-1]['cuda_us']=timing(run_norm)
    rows[-1]['torch_us']=timing(lambda:(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)*w)
    lib.probe_activation(x.data_ptr(),g.data_ptr(),y.data_ptr(),x.numel(),stream)
    check('named_silu_vs_actual_silu',y,torch.nn.functional.silu(x)*g)
    def run_swiglu():lib.probe_swiglu(x.data_ptr(),g.data_ptr(),y.data_ptr(),x.numel(),stream)
    run_swiglu();check('qwen36_swiglu',y,torch.nn.functional.silu(x)*g)
    rows[-1]['cuda_us']=timing(run_swiglu);rows[-1]['torch_us']=timing(lambda:torch.nn.functional.silu(x)*g)
    out=r.clone();lib.probe_residual(out.data_ptr(),x.data_ptr(),x.numel(),stream)
    check('residual_add',out,r+x)
    # Keep benchmarks non-mutating across iterations by restoring destination.
    def run_add():
        out.copy_(r);lib.probe_residual(out.data_ptr(),x.data_ptr(),x.numel(),stream)
    def eager_add():out.copy_(r);out.add_(x)
    rows[-1]['cuda_us']=timing(run_add);rows[-1]['torch_us']=timing(eager_add)
    out.copy_(r);lib.probe_gate(out.data_ptr(),x.data_ptr(),g.data_ptr(),x.numel(),stream)
    check('gate_bf16_intermediate',out,r+x*g)
    check('gate_fp32_intermediate',out,(r.float()+x.float()*g.float()).to(x.dtype))
    def run_gate():
        out.copy_(r);lib.probe_gate(out.data_ptr(),x.data_ptr(),g.data_ptr(),x.numel(),stream)
    rows[-1]['cuda_us']=timing(run_gate)
    rows[-1]['torch_us']=timing(lambda:(r.float()+x.float()*g.float()).to(x.dtype))
report={'kind':'synthetic contract screen, not Cosmos action qualification','torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'rows':rows,
        'library_sha256':hashlib.sha256((root/'kernels.so').read_bytes()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
