"""P009-A5 pinned bitexact BF16 cuBLASLt tactics for two hot Wan Linear shapes."""
from __future__ import annotations
import ctypes,os,types,weakref
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from instinctflash.backends.sm120_wan_stage2 import CERTIFIED_CUDA_VERSION,CERTIFIED_TORCH_VERSION,REQUIRED_ALIGNMENT

LIBRARY_ENV="IFL_SM120_GEMM_LIBRARY"; LIBRARY_NAME="libinstinctflash_sm120_wan_gemm.so"; ABI_VERSION=1
CERTIFIED_CUBLASLT_VERSION=120804
CERTIFIED_CONFIGS={(480,3072,3072):(21,15,25,0,0),(64,14336,3072):(21,18,12,0,0)}
CERTIFIED_MODULES=274

def library_candidates():
 configured=os.environ.get(LIBRARY_ENV); packaged=Path(__file__).resolve().parents[1]/"native"/LIBRARY_NAME
 return tuple([Path(configured)] if configured else [])+(packaged,)
def _library_abi(path):
 if not path.is_file():return None
 try:
  lib=ctypes.CDLL(str(path)); fn=lib.instinctflash_sm120_wan_gemm_abi_version;fn.argtypes=[];fn.restype=ctypes.c_int;return int(fn())
 except (OSError,AttributeError):return None
def _library_cublaslt_version(path):
 if not path.is_file():return None
 try:
  lib=ctypes.CDLL(str(path));fn=lib.wan_gemm_cublaslt_version;fn.argtypes=[];fn.restype=ctypes.c_uint64;return int(fn())
 except (OSError,AttributeError):return None
def available():return any(_library_abi(p)==ABI_VERSION and _library_cublaslt_version(p)==CERTIFIED_CUBLASLT_VERSION for p in library_candidates())
def resolve_library():
 for p in library_candidates():
  if _library_abi(p)==ABI_VERSION and _library_cublaslt_version(p)==CERTIFIED_CUBLASLT_VERSION:return p
 raise RuntimeError(f"SM120 Wan GEMM ABI v{ABI_VERSION} unavailable; searched {', '.join(map(str,library_candidates()))}. Build native/CMakeLists.txt or set {LIBRARY_ENV}. P009-A5 refuses instead of changing tactics implicitly.")

@dataclass(frozen=True)
class PinnedPlan:
 index:int;m:int;n:int;k:int;weight_ptr:int;bias_ptr:int;weight_version:int;bias_version:int

class SM120WanGEMMKernels:
 def __init__(self,library=None):
  import torch
  if not torch.cuda.is_available():raise RuntimeError("P009-A5 requires CUDA")
  i=torch.cuda.current_device()
  if torch.cuda.get_device_capability(i)!=(12,0):raise RuntimeError("P009-A5 is certified only on SM120")
  if torch.__version__!=CERTIFIED_TORCH_VERSION or torch.version.cuda!=CERTIFIED_CUDA_VERSION:raise RuntimeError(f"P009-A5 requires torch={CERTIFIED_TORCH_VERSION}, CUDA={CERTIFIED_CUDA_VERSION}; got torch={torch.__version__}, CUDA={torch.version.cuda}")
  path=Path(library) if library is not None else resolve_library();abi=_library_abi(path);lt_version=_library_cublaslt_version(path)
  if abi!=ABI_VERSION:raise RuntimeError(f"expected A5 ABI v{ABI_VERSION}, got {abi} from {path}")
  if lt_version!=CERTIFIED_CUBLASLT_VERSION:raise RuntimeError(f"P009-A5 requires cuBLASLt {CERTIFIED_CUBLASLT_VERSION}, got {lt_version} from {path}")
  self.path=path;self.device=torch.device("cuda",i);self.lib=ctypes.CDLL(str(path));u=ctypes.c_uint64
  self._create=self.lib.wan_gemm_context_create;self._create.argtypes=[];self._create.restype=u
  self._register=self.lib.wan_gemm_plan_register;self._register.argtypes=[u,ctypes.c_int,ctypes.c_int,ctypes.c_int,u,u,ctypes.c_int,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32];self._register.restype=ctypes.c_int
  self._run=self.lib.wan_gemm_bf16;self._run.argtypes=[u,ctypes.c_int,u,u,u,u,u];self._run.restype=ctypes.c_int
  self._last=self.lib.wan_gemm_context_last_status;self._last.argtypes=[u];self._last.restype=ctypes.c_int
  self._destroy=self.lib.wan_gemm_context_destroy;self._destroy.argtypes=[u];self._destroy.restype=None
  self.context=int(self._create())
  if not self.context:raise RuntimeError("P009-A5 could not create cuBLASLt context")
  self._finalizer=weakref.finalize(self,self._destroy,self.context);self._stream=None;self.calls=Counter()
 def _check_tensor(self,name,t,dtype):
  import torch
  if not isinstance(t,torch.Tensor):raise TypeError(f"{name} must be Tensor")
  if not t.is_cuda or t.device!=self.device:raise ValueError(f"{name} must be on {self.device}")
  if t.dtype is not dtype:raise TypeError(f"{name} must be {dtype}, got {t.dtype}")
  if t.data_ptr()%REQUIRED_ALIGNMENT:raise ValueError(f"{name} must be {REQUIRED_ALIGNMENT}-byte aligned")
 def register_linear(self,linear,m):
  import torch
  if not isinstance(linear,torch.nn.Linear):raise TypeError("linear must be nn.Linear")
  n,k=map(int,linear.weight.shape);key=(int(m),n,k);cfg=CERTIFIED_CONFIGS.get(key)
  if cfg is None:raise ValueError(f"uncertified A5 GEMM shape {key}")
  if linear.bias is None:raise ValueError("A5 requires a BF16 bias epilogue")
  self._check_tensor("weight",linear.weight,torch.bfloat16);self._check_tensor("bias",linear.bias,torch.bfloat16)
  if not linear.weight.is_contiguous() or not linear.bias.is_contiguous():raise ValueError("weight and bias must be contiguous")
  index=self._register(self.context,*key,linear.weight.data_ptr(),linear.bias.data_ptr(),*cfg)
  if index<0:raise RuntimeError(f"A5 tactic registration failed for {key}: cuBLASLt status {self._last(self.context)}")
  return PinnedPlan(index,*key,linear.weight.data_ptr(),linear.bias.data_ptr(),linear.weight._version,linear.bias._version)
 def _stream_for(self,x):
  import torch
  stream=torch.cuda.current_stream(x.device).cuda_stream
  if self._stream is None:self._stream=stream
  elif self._stream!=stream:raise RuntimeError("P009-A5 requires one CUDA stream per Runtime")
  return stream
 def linear_into(self,plan,x,weight,bias,out):
  import torch
  self._check_tensor("x",x,torch.bfloat16);self._check_tensor("out",out,torch.bfloat16)
  if not x.is_contiguous() or x.shape[-1]!=plan.k or x.numel()!=plan.m*plan.k:raise ValueError(f"x must be contiguous with certified shape M={plan.m},K={plan.k}")
  if not out.is_contiguous() or out.numel()!=plan.m*plan.n or out.shape[:-1]!=x.shape[:-1] or out.shape[-1]!=plan.n:raise ValueError("out shape must equal x prefix plus certified N")
  if weight.data_ptr()!=plan.weight_ptr or bias.data_ptr()!=plan.bias_ptr or weight._version!=plan.weight_version or bias._version!=plan.bias_version:raise RuntimeError("A5 pinned weight/bias changed after registration")
  status=self._run(self.context,plan.index,x.data_ptr(),weight.data_ptr(),bias.data_ptr(),out.data_ptr(),self._stream_for(x))
  if status:raise RuntimeError(f"A5 cuBLASLt launch failed with status {status}")
  self.calls[(plan.m,plan.n,plan.k)]+=1;return out

def install_wan_gemm(transformer,kernels=None):
 import torch
 if getattr(transformer,"_ifl_wan_qk_rope_kernels",None) is None:raise RuntimeError("P009-A5 requires installed A1+A2+A3+A4 chain")
 blocks=list(transformer.blocks)
 if len(blocks)!=30 or not all(getattr(b.attn1,"_ifl_wan_qk_rope_installed",False) for b in blocks):raise RuntimeError("P009-A5 requires the full 30-block P009-A4 install")
 kernels=kernels or SM120WanGEMMKernels();sites=[]
 for name,module in transformer.named_modules():
  if not isinstance(module,torch.nn.Linear) or module.bias is None:continue
  n,k=map(int,module.weight.shape)
  for m,nn,kk in CERTIFIED_CONFIGS:
   if (n,k)==(nn,kk):sites.append((name,module,m));break
 if len(sites)!=CERTIFIED_MODULES:raise RuntimeError(f"P009-A5 certified module graph has {CERTIFIED_MODULES} Linear sites; found {len(sites)}")
 for name,module,m in sites:
  if getattr(module,"_ifl_wan_gemm_installed",False):raise RuntimeError(f"P009-A5 duplicate install at {name}")
  plan=kernels.register_linear(module,m);original=module.forward;buffers={}
  def forward(self,x,_plan=plan,_original=original,_buffers=buffers):
   if (not x.is_cuda or x.device!=kernels.device or x.dtype is not torch.bfloat16 or not x.is_contiguous() or x.shape[-1]!=_plan.k or x.numel()!=_plan.m*_plan.k or self.weight.data_ptr()!=_plan.weight_ptr or self.bias.data_ptr()!=_plan.bias_ptr or self.weight._version!=_plan.weight_version or self.bias._version!=_plan.bias_version):return _original(x)
   shape=(*x.shape[:-1],_plan.n);key=(shape,x.device,x.dtype);out=_buffers.get(key)
   if out is None:out=_buffers[key]=torch.empty(shape,device=x.device,dtype=x.dtype)
   return kernels.linear_into(_plan,x,self.weight,self.bias,out)
  module._ifl_wan_gemm_original_forward=original;module._ifl_wan_gemm_buffers=buffers;module._ifl_wan_gemm_plan=plan;module.forward=types.MethodType(forward,module);module._ifl_wan_gemm_installed=True
 transformer._ifl_wan_gemm_kernels=kernels;transformer._ifl_wan_gemm_sites=tuple(name for name,_,_ in sites);return kernels

__all__=["ABI_VERSION","CERTIFIED_CUBLASLT_VERSION","CERTIFIED_CONFIGS","LIBRARY_ENV","LIBRARY_NAME","PinnedPlan","SM120WanGEMMKernels","available","install_wan_gemm","resolve_library"]
