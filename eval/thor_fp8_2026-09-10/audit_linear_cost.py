"""Thor operator diagnostic: graph replay isolates the dynamic conversion tax."""
import fcntl,json,statistics,torch
from pathlib import Path
from instinctflash.runtime.torch_fp8_linear import ThorFP8Linear
from instinctflash.runtime.fp8_pack import pack_bf16_e4m3
lock=open('/tmp/thor_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
assert torch.cuda.get_device_capability()==(11,0)
torch.manual_seed(9317);rows=[]
def bench(fn):
 for _ in range(4):fn()
 torch.cuda.synchronize();stream=torch.cuda.Stream()
 with torch.cuda.stream(stream):
  for _ in range(3):fn()
 stream.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=stream):fn()
 torch.cuda.synchronize();samples=[]
 for _ in range(5):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(20):graph.replay()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)/20)
 return statistics.median(samples)
for m,k,n in [(1,2048,2048),(40,2048,2048),(256,2048,2048),(2048,2048,2048),(256,5120,5120)]:
 native=torch.nn.Linear(k,n,bias=False,device='cuda',dtype=torch.bfloat16).eval();fp=ThorFP8Linear(native)
 x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
 def pack():
  scale=x.abs().amax().float().clamp_min(1e-12).reshape(1)/448.
  return pack_bf16_e4m3(x,scale),scale
 packed,scale=pack()
 def mm():return torch._scaled_mm(packed,fp.weight_fp8.t(),scale_a=scale,scale_b=fp.weight_scale,out_dtype=torch.bfloat16,use_fast_accum=False)
 with torch.no_grad():
  row={'M':m,'K':k,'N':n,'bf16_ms':bench(lambda:native(x)),'fp8_full_ms':bench(lambda:fp(x)),'fp8_prepacked_mm_ms':bench(mm),'dynamic_pack_ms':bench(pack)}
 rows.append(row);print(row,flush=True)
Path('/home/guanming/ifl_eval/thor_fp8_audit_20260910/linear-cost.json').write_text(json.dumps({'device':torch.cuda.get_device_name(),'torch':torch.__version__,'rows':rows,'scope':'Representative synthetic projection shapes, CUDA graph replay; not a model speed or task-quality claim'},indent=2))
