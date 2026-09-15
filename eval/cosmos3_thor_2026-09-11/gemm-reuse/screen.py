"""Real Cosmos gen-token GEMM shapes, synthetic tensors, unchanged engine runner."""
import ctypes as c
import hashlib
import json
from pathlib import Path
import sys
import torch

root=Path(sys.argv[1]);out=root/'screen.json';assert not out.exists()
assert torch.cuda.get_device_capability()==(11,0)
torch.manual_seed(9173);torch.backends.cuda.matmul.allow_tf32=False
lib=c.CDLL(str(root/'bf16_gemm.so'))
lib.gemm_create.restype=c.c_void_p
lib.gemm_destroy.argtypes=[c.c_void_p]
lib.gemm_error.restype=c.c_char_p
lib.gemm_run.argtypes=[c.c_void_p]*4+[c.c_int]*3+[c.c_void_p]
lib.gemm_tune.argtypes=[c.c_void_p]*4+[c.c_int]*3
ctx=lib.gemm_create();assert ctx,lib.gemm_error()

def timed(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):result=fn()
    for _ in range(5):graph.replay()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(30):graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)/30

rows=[]
try:
 with torch.inference_mode():
  for family,k,n in [('edge',2048,2048),('edge',2048,1024),('edge',2048,9216),('edge',9216,2048),
                     ('nano',4096,4096),('nano',4096,1024),('nano',4096,12288),('nano',12288,4096)]:
    m=3094
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
    packed=w.t().contiguous();d=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
    ref=torch.nn.functional.linear(x,w)
    def run():
      code=lib.gemm_run(ctx,x.data_ptr(),packed.data_ptr(),d.data_ptr(),m,n,k,torch.cuda.current_stream().cuda_stream)
      if code:raise RuntimeError(lib.gemm_error().decode())
      return d
    def difference(a):
      return dict(byte_equal=a.view(torch.uint8).equal(ref.view(torch.uint8)),max_abs=float((a.float()-ref.float()).abs().max()))
    row=dict(family=family,m=m,n=n,k=k,torch_native_ms=timed(lambda:torch.nn.functional.linear(x,w)),
        torch_prepacked_ms=timed(lambda:torch.mm(x,packed)),prepacked_difference=difference(torch.mm(x,packed)),
        engine_default_ms=timed(run),engine_default_difference=difference(run()))
    if lib.gemm_tune(ctx,x.data_ptr(),packed.data_ptr(),d.data_ptr(),m,n,k):raise RuntimeError(lib.gemm_error().decode())
    row.update(engine_tuned_ms=timed(run),engine_tuned_difference=difference(run()),
        torch_native_repeat_ms=timed(lambda:torch.nn.functional.linear(x,w)))
    rows.append(row);print(json.dumps(row),flush=True)
 out.write_text(json.dumps(dict(rows=rows,torch=torch.__version__,
    allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    library_sha256=hashlib.sha256((root/'bf16_gemm.so').read_bytes()).hexdigest(),
    qualification='Synthetic GEMM screen under CUDA Graph; excludes one-time weight transpose/tuning. No model action certificate.'),indent=2)+'\n')
finally:
 torch.cuda.synchronize();lib.gemm_destroy(ctx)
