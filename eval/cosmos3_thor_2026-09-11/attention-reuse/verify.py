"""Tail masking, strided input, output lifetime and graph replay checks on Thor."""
import json
from pathlib import Path
import sys
import torch
from probe import FMHA

op=FMHA(Path(__file__).with_name('bf16_fmha.so'))
checks=[]
try:
    for nq,nk in [(1,1),(17,31),(129,257)]:
        q=torch.zeros(1,nq,4,128,device='cuda',dtype=torch.bfloat16)
        k=torch.zeros(1,nk,4,128,device='cuda',dtype=torch.bfloat16)
        v=torch.ones_like(k)
        out=op(q,k,v)
        ok=torch.equal(out,torch.ones_like(out))
        assert ok,('tail-mask',nq,nk)
        checks.append({'test':'tail-mask-constant-values','q':nq,'k':nk,'pass':ok})
    torch.manual_seed(709)
    packed=torch.randn(1,129,3,4,128,device='cuda',dtype=torch.bfloat16)
    q,k,v=packed[:,:,0],packed[:,:,1],packed[:,:,2]
    a=op(q,k,v);b=op(q.contiguous(),k.contiguous(),v.contiguous())
    assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    kept=a.clone();op(q*2,k,v)
    assert torch.equal(a.view(torch.uint8),kept.view(torch.uint8))
    checks.append({'test':'strided-input-and-retained-output','pass':True})
    for _ in range(3):op(q,k,v)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=op(q,k,v)
    q.mul_(.5)
    graph.replay()
    expected=op(q,k,v)
    assert torch.equal(captured.view(torch.uint8),expected.view(torch.uint8))
    checks.append({'test':'graph-replay-changed-input','pass':True})
    result={'status':'PASS','scope':'mechanics only, not task quality or NATTEN byte equivalence','checks':checks}
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
finally:op.close()
