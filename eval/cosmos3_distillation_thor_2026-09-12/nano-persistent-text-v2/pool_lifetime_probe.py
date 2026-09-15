"""Tiny isolated CUDA pool-lifetime diagnostic; no model or quality claim."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['retired','fresh'],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
r=dict(mode=a.mode,torch=torch.__version__,cuda=torch.version.cuda,
       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cycles=[])
x=torch.ones(1024,device='cuda')
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):y=x.sin()
torch.cuda.current_stream().wait_stream(stream)
pool=torch.cuda.graph_pool_handle()
held=[]
try:
    for i in range(3):
        if a.mode=='fresh':pool=torch.cuda.graph_pool_handle()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,pool=pool):y=x.sin()
        graph.replay();torch.cuda.synchronize()
        assert torch.equal(y,x.sin())
        held.append(y)  # A live tensor can outlast its owning graph.
        r['cycles'].append(dict(cycle=i,pool=list(pool),exact=True))
        del graph
        gc.collect()
    r['status']='success'
except Exception as error:
    r.update(status='error',error=repr(error))
finally:
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r),flush=True)
