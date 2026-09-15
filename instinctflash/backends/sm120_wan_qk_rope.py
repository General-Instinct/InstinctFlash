"""P009-A4 exact BF16 RMSNorm + FP64-complex RoPE fusion for SM120 Wan self-attention."""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import os
import types
from pathlib import Path

from instinctflash.backends.sm120_wan_stage2 import (
    CERTIFIED_CUDA_VERSION,
    CERTIFIED_DIM,
    CERTIFIED_EPS,
    CERTIFIED_ROWS,
    CERTIFIED_TORCH_VERSION,
    REQUIRED_ALIGNMENT,
    _assert_no_alias,
)

LIBRARY_ENV = "IFL_SM120_QK_ROPE_LIBRARY"
LIBRARY_NAME = "libinstinctflash_sm120_wan_qk_rope.so"
ABI_VERSION = 1
CERTIFIED_HEADS = 24
CERTIFIED_HEAD_DIM = 128
CERTIFIED_PAIRS = CERTIFIED_HEAD_DIM // 2
CERTIFIED_RING_FORWARD_SHA256 = "deb31380c8b6cf33d7f11516a11d4d679da70e1446fae760ae7217b99a347b57"


def library_candidates() -> tuple[Path, ...]:
    configured = os.environ.get(LIBRARY_ENV)
    packaged = Path(__file__).resolve().parents[1] / "native" / LIBRARY_NAME
    return tuple([Path(configured)] if configured else []) + (packaged,)


def _library_abi(path: Path) -> int | None:
    if not path.is_file(): return None
    try:
        library=ctypes.CDLL(str(path))
        version=library.instinctflash_sm120_wan_qk_rope_abi_version
        version.argtypes=[]; version.restype=ctypes.c_int
        return int(version())
    except (OSError,AttributeError): return None


def available() -> bool:
    return any(_library_abi(path)==ABI_VERSION for path in library_candidates())


def resolve_library() -> Path:
    for path in library_candidates():
        if _library_abi(path)==ABI_VERSION: return path
    raise RuntimeError(
        f"SM120 Wan QK-RoPE ABI v{ABI_VERSION} unavailable; searched "
        f"{', '.join(str(path) for path in library_candidates())}. Build native/CMakeLists.txt "
        f"or set {LIBRARY_ENV}. P009-A4 refuses instead of silently using eager.")


class SM120WanQKRoPEKernels:
    def __init__(self, library: str|Path|None=None):
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("P009-A4 requires CUDA")
        index=torch.cuda.current_device()
        if torch.cuda.get_device_capability(index)!=(12,0):
            raise RuntimeError("P009-A4 is certified only on SM120")
        if torch.__version__!=CERTIFIED_TORCH_VERSION or torch.version.cuda!=CERTIFIED_CUDA_VERSION:
            raise RuntimeError(
                f"P009-A4 requires torch={CERTIFIED_TORCH_VERSION}, CUDA={CERTIFIED_CUDA_VERSION}; "
                f"got torch={torch.__version__}, CUDA={torch.version.cuda}")
        path=Path(library) if library is not None else resolve_library()
        abi=_library_abi(path)
        if abi!=ABI_VERSION: raise RuntimeError(f"expected A4 ABI v{ABI_VERSION}, got {abi} from {path}")
        self.path=path; self.device=torch.device("cuda",index); self.lib=ctypes.CDLL(str(path))
        u64,i32,f32=ctypes.c_uint64,ctypes.c_int,ctypes.c_float
        self._launch_raw=self.lib.wan_qk_rms_rope_bf16
        self._launch_raw.argtypes=[u64,u64,u64,u64,u64,i32,i32,f32,u64]
        self._launch_raw.restype=i32
        self.calls=0; self._stream=None; self._validated=set()

    def _check(self,name,tensor,dtype):
        import torch
        if not isinstance(tensor,torch.Tensor): raise TypeError(f"{name} must be Tensor")
        if not tensor.is_cuda or tensor.device!=self.device: raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype is not dtype: raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
        if tensor.data_ptr()%REQUIRED_ALIGNMENT: raise ValueError(f"{name} is not {REQUIRED_ALIGNMENT}-byte aligned")

    def _validate(self,x,weight,freqs,out,rstds):
        import torch
        self._check("x",x,torch.bfloat16); self._check("weight",weight,torch.bfloat16)
        self._check("freqs",freqs,torch.complex64); self._check("out",out,torch.bfloat16)
        self._check("rstds",rstds,torch.float32)
        if x.ndim!=3 or not x.is_contiguous() or x.shape[-1]!=CERTIFIED_DIM:
            raise ValueError("x must be contiguous [B,L,3072]")
        rows=x.numel()//CERTIFIED_DIM
        if rows not in CERTIFIED_ROWS: raise ValueError(f"uncertified A4 rows={rows}")
        if weight.shape!=(CERTIFIED_DIM,) or not weight.is_contiguous(): raise ValueError("weight must be contiguous [3072]")
        expected=(*x.shape[:2],CERTIFIED_HEADS,CERTIFIED_HEAD_DIM)
        if out.shape!=expected or not out.is_contiguous(): raise ValueError(f"out must be contiguous {expected}")
        if (freqs.ndim!=4 or freqs.shape[0:2]!=x.shape[0:2] or freqs.shape[2:]!=(1,CERTIFIED_PAIRS)
                or not freqs.is_contiguous()):
            raise ValueError("freqs must be contiguous complex64 [B,L,1,64]")
        if rstds.shape!=(rows,) or not rstds.is_contiguous(): raise ValueError(f"rstds must be [{rows}]")
        _assert_no_alias({"x":x,"weight":weight,"freqs":freqs,"out":out,"rstds":rstds})
        return rows

    def _stream_for(self,x):
        import torch
        stream=torch.cuda.current_stream(x.device).cuda_stream
        if self._stream is None: self._stream=stream
        elif stream!=self._stream: raise RuntimeError("P009-A4 requires one CUDA stream per Runtime")
        return stream

    def _launch(self,x,weight,freqs,out,rstds,rows):
        error=self._launch_raw(x.data_ptr(),weight.data_ptr(),freqs.data_ptr(),out.data_ptr(),
            rstds.data_ptr(),rows,CERTIFIED_DIM,CERTIFIED_EPS,self._stream_for(x))
        if error: raise RuntimeError(f"P009-A4 launch failed with CUDA error {error}")
        self.calls+=1

    def qk_rope_into(self,x,weight,freqs,out,rstds):
        self._launch(x,weight,freqs,out,rstds,self._validate(x,weight,freqs,out,rstds))

    def qk_rope_certified(self,x,weight,freqs,out,rstds):
        key=(out.data_ptr(),rstds.data_ptr(),tuple(x.shape),weight.data_ptr(),freqs.data_ptr())
        if key not in self._validated:
            rows=self._validate(x,weight,freqs,out,rstds); self._validated.add(key)
        else: rows=x.numel()//CERTIFIED_DIM
        self._launch(x,weight,freqs,out,rstds,rows)


def install_wan_qk_rope(transformer,kernels:SM120WanQKRoPEKernels|None=None):
    import torch
    if getattr(transformer,"_ifl_wan_stage3_kernels",None) is None:
        raise RuntimeError("P009-A4 requires installed A1+A2+A3 chain")
    kernels=kernels or SM120WanQKRoPEKernels(); blocks=list(transformer.blocks)
    if len(blocks)!=30 or not all(getattr(block,"_ifl_wan_stage3_installed",False) for block in blocks):
        raise RuntimeError("P009-A4 requires the full 30-block P009-A3 install")
    attentions=[block.attn1 for block in blocks]
    if len({type(attn) for attn in attentions})!=1: raise RuntimeError("P009-A4 requires one WanAttention class")
    try: source=inspect.getsource(type(attentions[0]).forward)
    except (OSError,TypeError) as error: raise RuntimeError("P009-A4 cannot certify ring forward") from error
    digest=hashlib.sha256(source.encode()).hexdigest()
    if digest!=CERTIFIED_RING_FORWARD_SHA256:
        raise RuntimeError(f"P003 ring forward changed; expected {CERTIFIED_RING_FORWARD_SHA256}, got {digest}")
    for attn in attentions:
        if (attn.heads!=CERTIFIED_HEADS or attn.inner_dim!=CERTIFIED_DIM
                or attn.cross_attention_dim_head is not None):
            raise RuntimeError("P009-A4 requires self-attention H24,D128,inner3072")
        for name,norm in (("norm_q",attn.norm_q),("norm_k",attn.norm_k)):
            if (tuple(norm.normalized_shape)!=(CERTIFIED_DIM,) or norm.eps!=CERTIFIED_EPS
                    or norm.weight is None or norm.weight.dtype is not torch.bfloat16
                    or not norm.weight.is_contiguous() or norm.weight.device!=kernels.device):
                raise RuntimeError(f"P009-A4 uncertified {name}")
        attn._ifl_qk_rope_original_forward=attn.forward; attn._ifl_qk_rope_buffers={}
        def a4_forward(self,q,k,v,rotary_emb,update_cache=0,cache_name="pos"):
            kv_cache=(self.attn_caches[cache_name] if self.attn_caches is not None and cache_name in self.attn_caches else None)
            r=None if kv_cache is None else kv_cache.get("_ring")
            if r is None or kv_cache.get("k") is None or rotary_emb is None:
                return self._ifl_qk_rope_original_forward(q,k,v,rotary_emb,update_cache,cache_name)
            key_size=k.shape[1]
            query,key,value=self.to_q(q),self.to_k(k),self.to_v(v)
            sig=(tuple(query.shape),query.device,query.dtype,tuple(rotary_emb.shape))
            bufs=self._ifl_qk_rope_buffers.get(sig)
            rows=query.numel()//CERTIFIED_DIM
            if bufs is None:
                bufs=self._ifl_qk_rope_buffers[sig]=(
                    torch.empty(*query.shape[:2],CERTIFIED_HEADS,CERTIFIED_HEAD_DIM,device=query.device,dtype=query.dtype),
                    torch.empty(*key.shape[:2],CERTIFIED_HEADS,CERTIFIED_HEAD_DIM,device=key.device,dtype=key.dtype),
                    torch.empty(rows,device=query.device,dtype=torch.float32),
                    torch.empty(rows,device=query.device,dtype=torch.float32))
            query_out,key_out,qr,kr=bufs
            kernels.qk_rope_certified(query,self.norm_q.weight,rotary_emb,query_out,qr)
            kernels.qk_rope_certified(key,self.norm_k.weight,rotary_emb,key_out,kr)
            query,key=query_out,key_out; value=value.unflatten(2,(self.heads,-1))
            total=r["total"]; head=(r["start"]+r["count"])%total
            assert head+key_size<=total,"allocation wrapped; ring model violated"
            sl=slice(head,head+key_size); kvc=kv_cache
            if self._iwm_use_plan_buffer and r["count"]>=total:
                idx=self._iwm_plan_buffer(cache_name,key_size); kvc["k"].index_copy_(1,idx,key); kvc["v"].index_copy_(1,idx,value)
            else: kvc["k"][:,sl]=key; kvc["v"][:,sl]=value
            if not self._iwm_defer_commit:
                kvc["mask"][sl]=True; kvc["id"][sl]=r["next_id"]; kvc["is_pred"][sl]=(update_cache==1); r["next_id"]+=1
            kp,vp=kvc["k"],kvc["v"]; start,count=r["start"],r["count"]+key_size
            if count>=total: key_all,value_all=kp,vp
            elif start+count<=total: key_all,value_all=kp[:,start:start+count],vp[:,start:start+count]
            else:
                end=start+count-total; ring_concat=getattr(self,"_iwm_ring_concat",None)
                if ring_concat is None:
                    key_all=torch.cat([kp[:,:end],kp[:,start:]],dim=1); value_all=torch.cat([vp[:,:end],vp[:,start:]],dim=1)
                else:
                    key_all,value_all=ring_concat(kp,vp,start=start,count=count,total=total)
            hidden=self.attn_op(query,key_all,value_all)
            if not self._iwm_defer_commit: self._iwm_commit(cache_name,key_size,update_cache)
            hidden=hidden.flatten(2,3).type_as(query)
            return self.to_out[1](self.to_out[0](hidden))
        attn.forward=types.MethodType(a4_forward,attn); attn._ifl_wan_qk_rope_installed=True
    transformer._ifl_wan_qk_rope_kernels=kernels
    return kernels

__all__=["ABI_VERSION","LIBRARY_ENV","LIBRARY_NAME","SM120WanQKRoPEKernels","available","install_wan_qk_rope","resolve_library"]
