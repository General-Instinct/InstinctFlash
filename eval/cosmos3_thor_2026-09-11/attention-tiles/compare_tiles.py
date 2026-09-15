"""Screen persistent scheduling against the existing experimental BF16 FMHA."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from probe import FMHA,time_call

root=Path(sys.argv[1]); output=root/'tile-screen.json'
assert not output.exists()
paths=[Path('/home/guanming/ifl_eval/cosmos_attention_reuse_20260911/bf16_fmha.so'),root/'bf16_fmha.so']
ops=[FMHA(p) for p in paths]
rows=[]
torch.manual_seed(9173)
try:
    for nq,nk,hq,hk in ((17,31,8,2),(129,257,8,2),(3094,3245,16,8),(3094,3112,16,8),(3094,3182,32,8),(3094,3103,32,8)):
        q=torch.randn(1,nq,hq,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,nk,hk,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        a,b=[op(q,k,v) for op in ops]
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        row=dict(shape=[nq,nk,hq,hk],byte_equal=a.view(torch.uint8).equal(b.view(torch.uint8)),
            max_abs=float((a.float()-b.float()).abs().max()),
            individual_ms=time_call(lambda:ops[0](q,k,v)),persistent_ms=time_call(lambda:ops[1](q,k,v)),
            individual_repeat_ms=time_call(lambda:ops[0](q,k,v)))
        row['speedup']=row['individual_repeat_ms']/row['persistent_ms']
        rows.append(row);print(json.dumps(row),flush=True)
    output.write_text(json.dumps(dict(rows=rows,libraries={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        qualification='Synthetic GQA shapes; comparison to experimental BF16 kernel, not upstream BITEXACT or task quality'),indent=2)+'\n')
finally:
    for op in ops:op.close()
