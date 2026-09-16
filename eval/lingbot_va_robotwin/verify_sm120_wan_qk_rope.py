#!/usr/bin/env python3
"""RTX 5090 operator gate for P009-A4 exact Q/K RMSNorm + FP64-complex RoPE."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import torch
from instinctflash.backends.sm120_wan_qk_rope import SM120WanQKRoPEKernels
D,H,HD,P=3072,24,128,64

def eager(x,w,freqs):
    normed,rstd=torch.ops.aten._fused_rms_norm.default(x,[D],w,1e-6)
    value=torch.view_as_complex(normed.to(torch.float64).reshape(*x.shape[:2],H,P,2))
    output=torch.view_as_real(value*freqs).flatten(3).to(x.dtype)
    return output,rstd.reshape(-1)

def timed(fn,warm=20,iters=200):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); out=[]
    for _ in range(iters):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); out.append(a.elapsed_time(b))
    return statistics.median(out)

def one(kernels,rows,pattern,exponent):
    gen=torch.Generator(device='cuda').manual_seed(100+rows+exponent)
    if pattern=='random': x=torch.randn(1,rows,D,device='cuda',dtype=torch.bfloat16,generator=gen)
    elif pattern=='constant': x=torch.full((1,rows,D),2.0**exponent,device='cuda',dtype=torch.bfloat16)
    else:
        values=torch.empty(rows*D,device='cuda'); values[0::2]=2.0**exponent; values[1::2]=-(2.0**exponent)
        x=values.reshape(1,rows,D).to(torch.bfloat16)
    weight=torch.randn(D,device='cuda',dtype=torch.bfloat16,generator=gen)
    angles=torch.randn(1,rows,1,P,device='cuda',dtype=torch.float32,generator=gen)
    freqs=torch.polar(torch.ones_like(angles),angles).contiguous()
    out=torch.empty(1,rows,H,HD,device='cuda',dtype=torch.bfloat16); rstd=torch.empty(rows,device='cuda')
    ref,ref_rstd=eager(x,weight,freqs); kernels.qk_rope_into(x,weight,freqs,out,rstd); torch.cuda.synchronize()
    record={'rows':rows,'pattern':pattern,'exponent':exponent,
      'output':{'exact':torch.equal(ref,out),'differing_words':int(torch.count_nonzero(ref.view(torch.int16)!=out.view(torch.int16))),'max_abs':float((ref.float()-out.float()).abs().max())},
      'rstd':{'exact':torch.equal(ref_rstd,rstd),'differing_words':int(torch.count_nonzero(ref_rstd.view(torch.int32)!=rstd.view(torch.int32))),'max_abs':float((ref_rstd-rstd).abs().max())}}
    if pattern=='random' and exponent==0:
        em=timed(lambda:eager(x,weight,freqs)); fm=timed(lambda:kernels.qk_rope_certified(x,weight,freqs,out,rstd)); record['timing']={'eager_ms':em,'fused_ms':fm,'speedup':em/fm}
    return record

def guards(kernels):
    x=torch.zeros(1,64,D,device='cuda',dtype=torch.bfloat16); w=torch.ones(D,device='cuda',dtype=torch.bfloat16)
    f=torch.ones(1,64,1,P,device='cuda',dtype=torch.complex64); out=torch.empty(1,64,H,HD,device='cuda',dtype=torch.bfloat16); r=torch.empty(64,device='cuda')
    cases=[('dtype',(x,w,f.to(torch.complex128),out,r)),('shape',(x[:,:63],w,f[:,:63],out[:,:63],r[:63])),('alias',(x,w,f,x.view(1,64,H,HD),r))]
    ans=[]
    for label,args in cases:
        try: kernels.qk_rope_into(*args)
        except (TypeError,ValueError,RuntimeError) as error: ans.append({'label':label,'rejected':True,'error':str(error)})
        else: ans.append({'label':label,'rejected':False})
    return ans

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--library',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    kernels=SM120WanQKRoPEKernels(a.library); records=[one(kernels,rows,pattern,exponent) for rows in (64,480) for pattern in ('random','constant','alternating') for exponent in (-4,0,4)]; invalid=guards(kernels)
    exact=all(r[n]['exact'] for r in records for n in ('output','rstd')); result={'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'torch':torch.__version__,'cuda':torch.version.cuda,'library':str(a.library.resolve()),'records':records,'guards':invalid,'kernel_calls':kernels.calls,'all_bitexact':exact,'all_invalid_inputs_rejected':all(x['rejected'] for x in invalid)}; result['status']='pass' if result['all_bitexact'] and result['all_invalid_inputs_rejected'] else 'fail'; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); return 0 if result['status']=='pass' else 1
if __name__=='__main__': raise SystemExit(main())
