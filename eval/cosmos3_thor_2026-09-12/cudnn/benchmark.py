"""Offline cuDNN numeric override; never a production BITEXACT registration."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from cosmos3_iwm import conditioning_cache
from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService
from benchmarks.regression import cosmos

stats=dict(eligible_python_calls=0, fallback_python_calls=0, shapes={})
patches=[]
original_init=RobolabPolicyService.__init__
def initialize(self,args):
    original_init(self,args)
    import cosmos_framework.model.generator.mot.attention as mot
    original=mot.attention
    def attention(q,k,v,*args,**kwargs):
        eligible=(not args and set(kwargs).issubset({'is_causal','return_lse'})
            and not kwargs.get('is_causal',False) and not kwargs.get('return_lse',False)
            and q.ndim==k.ndim==v.ndim==4 and q.shape[0]==k.shape[0]==v.shape[0]==1
            and q.dtype==k.dtype==v.dtype==torch.bfloat16 and q.device==k.device==v.device and q.is_cuda
            and k.shape==v.shape and q.shape[-1]==k.shape[-1]==128
            and k.shape[2]>0 and q.shape[2]%k.shape[2]==0)
        if not eligible:
            stats['fallback_python_calls']+=1
            return original(q,k,v,*args,**kwargs)
        stats['eligible_python_calls']+=1
        key=str((tuple(q.shape),tuple(k.shape)))
        stats['shapes'][key]=stats['shapes'].get(key,0)+1
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=0.0,is_causal=False,scale=128**-0.5,enable_gqa=True).transpose(1,2)
        assert out.dtype==q.dtype and out.shape==q.shape
        return out
    patches.append((mot,original))
    mot.attention=attention
RobolabPolicyService.__init__=initialize
output=Path(sys.argv[3])
try:
    code=cosmos.main()
finally:
    RobolabPolicyService.__init__=original_init
    for module,original in reversed(patches):module.attention=original
r=json.loads(output.read_text())
r['runtime_declared_execution_policy']=r.pop('execution_policy',None)
r['execution_policy']=dict(category='SCREEN',precision='native',experimental_override='Forced cuDNN BF16 dense generation attention',bitexact=False,task_quality_certified=False)
r['cudnn_experiment']=dict(stats,cudnn_version=torch.backends.cudnn.version(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),note='Python call counters exclude graph replays; scope is an offline numeric candidate')
if r.get('ok') and not stats['eligible_python_calls']:
    r.update(ok=False,error='Requested attention replacement executed zero calls');code=1
output.write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(code)
