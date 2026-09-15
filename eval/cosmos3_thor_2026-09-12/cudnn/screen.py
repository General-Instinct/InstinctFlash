"""Forced cuDNN BF16 screen on captured native Cosmos attention operands."""
import argparse, hashlib, io, json, statistics, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from instinctflash import Runtime

p=argparse.ArgumentParser();p.add_argument('family',choices=('edge','nano'));p.add_argument('output',type=Path);p.add_argument('--fixture',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
assert torch.cuda.get_device_capability()==(11,0)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
api=Runtime.from_pretrained(f'nvidia/Cosmos3-{a.family.title()}-Policy-DROID',precision='native',tier_ceiling='bitexact')
api.reset(prompt='pick up the object')
import cosmos_framework.model.generator.mot.attention as mot
original=mot.attention
layers=28 if a.family=='edge' else 36
selected={1,layers//2,layers,layers+1,layers+layers//2,2*layers}
captures=[];count=0

def observe(q,k,v,*args,**kwargs):
    global count
    out=original(q,k,v,*args,**kwargs)
    if q.shape[1]>1000 and not kwargs.get('is_causal',False):
        count+=1
        if count in selected:
            assert not args and not kwargs, kwargs
            captures.append((count,*[t.detach().clone() for t in (q,k,v,out)]))
    return out
mot.attention=observe
try:
    with np.load(a.fixture,allow_pickle=True) as d:
        frame=np.asarray(Image.open(io.BytesIO(bytes(d['jpeg_0'][0][0]))).convert('RGB').resize((640,540)))
    action=np.asarray(api.predict({'image':frame,'state':np.zeros(8,np.float32),'prompt':'pick up the object'})['action'])
    assert np.isfinite(action).all() and len(captures)==6
    cache_status=api._backend._impl.backend_stats()['conditioning_cache_status']
finally:
    mot.attention=original
    api.close()

def measure(fn):
    for _ in range(3):fn()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(5):fn()
    times=[]
    for _ in range(5):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record();g.replay();e.record();e.synchronize();times.append(s.elapsed_time(e)/5)
    return statistics.median(times)
rows=[]
with torch.inference_mode():
    for index,q,k,v,ref in captures:
        baseline=lambda:original(q,k,v)
        assert torch.equal(baseline().contiguous().view(torch.uint8),ref.contiguous().view(torch.uint8))
        baseline_ms=measure(baseline)
        for variant in ('gqa','expanded'):
            row=dict(call=index,q_shape=list(q.shape),k_shape=list(k.shape),q_stride=list(q.stride()),variant=variant,baseline_ms=baseline_ms)
            def candidate():
                kk,vv=k,v
                if variant=='expanded':
                    kk=k.repeat_interleave(q.shape[2]//k.shape[2],dim=2)
                    vv=v.repeat_interleave(q.shape[2]//v.shape[2],dim=2)
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),kk.transpose(1,2),vv.transpose(1,2),dropout_p=0.0,is_causal=False,scale=q.shape[-1]**-0.5,enable_gqa=variant=='gqa').transpose(1,2)
            try:
                out=candidate();delta=(out.float()-ref.float()).abs()
                row.update(finite=bool(torch.isfinite(out).all()),byte_equal=torch.equal(out.contiguous().view(torch.uint8),ref.contiguous().view(torch.uint8)),max_abs_delta=float(delta.max()),mean_abs_delta=float(delta.mean()),candidate_ms=measure(candidate))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    candidate();torch.cuda.synchronize()
                row['operators']=[e.key for e in prof.key_averages() if 'attention' in e.key.lower() or 'cudnn' in e.key.lower()]
                assert any('_scaled_dot_product_cudnn_attention' in n for n in row['operators']),row['operators']
                row['speedup']=baseline_ms/row['candidate_ms']
            except Exception as error:
                row['error']=repr(error)
            rows.append(row);print(json.dumps(row),flush=True)
report=dict(family=a.family,diagnostic_only=True,scope='Real tensors: first/middle/last layer, conditional/unconditional first-step branches; not full-model speed or quality',torch=torch.__version__,cudnn=torch.backends.cudnn.version(),cache_status=cache_status,rows=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(),sources={str(Path(m.__file__).resolve()):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for name,m in list(sys.modules.items()) if name.startswith(('instinctflash','cosmos3_iwm','cosmos_framework','natten')) and getattr(m,'__file__',None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()})
a.output.write_text(json.dumps(report,indent=2)+'\n')
