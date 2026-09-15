"""Experimental native-BF16 numeric candidate. Never installed by Runtime defaults."""
from pathlib import Path
import hashlib
import torch
from probe import FMHA


def install(service, library):
    op=FMHA(library)
    stats={'calls':0,'shapes':{},'kind':'native BF16 numeric candidate; not BITEXACT',
           'installer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'library_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest()}

    @torch.library.custom_op('ifl_cosmos_probe::bf16_attention',mutates_args=())
    def fused(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,scale:float)->torch.Tensor:
        result=op(q,k,v,scale)
        stats['calls']+=1
        shape=str((tuple(q.shape),tuple(k.shape)))
        stats['shapes'][shape]=stats['shapes'].get(shape,0)+1
        return result

    @fused.register_fake
    def fake(q,k,v,scale):
        return torch.empty(q.shape,device=q.device,dtype=q.dtype)

    import cosmos_framework.model.generator.mot.attention as mot_attention
    import cosmos_framework.model.generator.mot.inference_text_kv_memory as memory_attention
    originals=[]
    def make_dispatch(original):
        def dispatch(query,key,value,*args,**kwargs):
            # Never drop causal, varlen, backend, LSE or other requested semantics.
            supported_kwargs={'is_causal','return_lse'}
            eligible=(not args and set(kwargs).issubset(supported_kwargs)
                and not kwargs.get('is_causal',False) and not kwargs.get('return_lse',False)
                and query.ndim==key.ndim==value.ndim==4
                and query.dtype==key.dtype==value.dtype==torch.bfloat16
                and query.device==key.device==value.device and query.is_cuda
                and query.shape[0]==key.shape[0]==value.shape[0]==1
                and query.shape[-1]==key.shape[-1]==value.shape[-1]==128
                and key.shape==value.shape and key.shape[2]>0 and query.shape[2]%key.shape[2]==0
                and 0<query.shape[1]<=4096 and query.shape[2]<=64
                and all(t.stride(-1)==1 and t.stride(-2)==128 for t in (query,key,value))
                and key.stride(1)==value.stride(1))
            if eligible:
                return fused(query,key,value,128**-0.5)
            return original(query,key,value,*args,**kwargs)
        return dispatch
    for module in (mot_attention,memory_attention):
        original=module.attention
        originals.append((module,original))
        module.attention=make_dispatch(original)
    service._ifl_attention_probe=(op,originals)
    return stats
