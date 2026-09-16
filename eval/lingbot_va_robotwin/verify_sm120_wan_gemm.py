#!/usr/bin/env python3
"""Formal operator gate for P009-A5 pinned BF16 cuBLASLt tactics."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from instinctflash.backends.sm120_wan_gemm import CERTIFIED_CONFIGS,SM120WanGEMMKernels

def fill(pattern,exponent,m,n,k,seed,linear):
 scale=2.0**exponent;g=torch.Generator(device="cuda").manual_seed(seed)
 if pattern=="random":x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16,generator=g)*scale;w=torch.randn(n,k,device="cuda",dtype=torch.bfloat16,generator=g)*0.125
 elif pattern=="constant":x=torch.full((m,k),0.25*scale,device="cuda",dtype=torch.bfloat16);w=torch.full((n,k),-0.125,device="cuda",dtype=torch.bfloat16)
 else:
  signs=(torch.arange(k,device="cuda")%2*2-1).to(torch.bfloat16);x=signs.expand(m,k).clone().mul_(scale);w=torch.randn(n,k,device="cuda",dtype=torch.bfloat16,generator=g)*0.125
 b=torch.randn(n,device="cuda",dtype=torch.bfloat16,generator=g)
 with torch.no_grad():linear.weight.copy_(w);linear.bias.copy_(b)
 return x

def bench(reference,candidate):
 for _ in range(12):reference();candidate()
 torch.cuda.synchronize();begin,end=torch.cuda.Event(True),torch.cuda.Event(True)
 def arm(fn):
  values=[]
  for _ in range(31):
   begin.record()
   for _ in range(20):fn()
   end.record();end.synchronize();values.append(begin.elapsed_time(end)*50.0)
  return {"median_us":statistics.median(values),"min_us":min(values),"spread_percent":(max(values)-min(values))/statistics.mean(values)*100}
 b1=arm(reference);c1=arm(candidate);c2=arm(candidate);b2=arm(reference);bm=(b1["median_us"]+b2["median_us"])/2;cm=(c1["median_us"]+c2["median_us"])/2
 return {"baseline_arms":[b1,b2],"candidate_arms":[c1,c2],"baseline_mean_us":bm,"candidate_mean_us":cm,"delta_us":bm-cm,"speedup":bm/cm}

def guards(kernels,linear,plan):
 good_x=torch.empty(plan.m,plan.k,device="cuda",dtype=torch.bfloat16);good_out=torch.empty(plan.m,plan.n,device="cuda",dtype=torch.bfloat16);rows=[]
 cases=(("wrong_dtype",good_x.float(),good_out),("wrong_m",good_x[:-1],good_out),("noncontiguous",good_x.t(),good_out),("wrong_out",good_x,good_out[:,:-1]))
 for label,x,out in cases:
  try:kernels.linear_into(plan,x,linear.weight,linear.bias,out)
  except (TypeError,ValueError,RuntimeError) as error:rows.append({"label":label,"rejected":True,"error":str(error)})
  else:rows.append({"label":label,"rejected":False})
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--library",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args();kernels=SM120WanGEMMKernels(args.library);records=[];timings={};guard_rows=[];total=diffs=0
 for m,n,k in CERTIFIED_CONFIGS:
  linear=torch.nn.Linear(k,n,bias=True,device="cuda",dtype=torch.bfloat16).eval().requires_grad_(False)
  for pi,pattern in enumerate(("random","constant","alternating")):
   for exponent in (-4,0,4):
    x=fill(pattern,exponent,m,n,k,1000+m+n+pi*10+exponent,linear);plan=kernels.register_linear(linear,m);out=torch.empty(m,n,device="cuda",dtype=torch.bfloat16)
    def candidate():return kernels.linear_into(plan,x,linear.weight,linear.bias,out)
    reference=F.linear(x,linear.weight,linear.bias);candidate();torch.cuda.synchronize();diff=int((reference.view(torch.int16)!=out.view(torch.int16)).sum());first=out.clone();det=True
    for _ in range(4):candidate();torch.cuda.synchronize();det &= torch.equal(first.view(torch.int16),out.view(torch.int16))
    row={"M":m,"N":n,"K":k,"pattern":pattern,"exponent":exponent,"words":reference.numel(),"differing_words":diff,"deterministic":det};records.append(row);total+=reference.numel();diffs+=diff;print(row,flush=True)
    if pattern=="random" and exponent==0:timings[f"{m}x{n}x{k}"]=bench(lambda:F.linear(x,linear.weight,linear.bias),candidate);print("TIMING",timings[f"{m}x{n}x{k}"],flush=True)
  guard_rows.extend(guards(kernels,linear,plan))
 source=Path(__file__).resolve().parents[2]/"instinctflash/native/wan_gemm_sm120.cu"
 result={"schema_version":1,"device":torch.cuda.get_device_name(),"capability":list(torch.cuda.get_device_capability()),"torch":torch.__version__,"cuda":torch.version.cuda,"tier":"BITEXACT","configs":{str(k):list(v) for k,v in CERTIFIED_CONFIGS.items()},"compared_words":total,"differing_words":diffs,"all_bitexact":diffs==0,"all_deterministic":all(r["deterministic"] for r in records),"records":records,"guards":guard_rows,"all_invalid_inputs_rejected":all(r["rejected"] for r in guard_rows),"timings":timings,"kernel_calls":dict((str(k),v) for k,v in kernels.calls.items()),"artifacts":{"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"library_sha256":hashlib.sha256(args.library.read_bytes()).hexdigest()}}
 result["status"]="pass" if result["all_bitexact"] and result["all_deterministic"] and result["all_invalid_inputs_rejected"] else "fail";args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return 0 if result["status"]=="pass" else 1
if __name__=="__main__":raise SystemExit(main())
