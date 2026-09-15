"""Exercise the shared ConditioningCache wrapper with actual CUDA LayerGraphs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import gc
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();assert not a.output.exists()
spec=importlib.util.spec_from_file_location('cosmos3_iwm.pool_probe_cache',a.source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
r=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
       probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,cycles=[])
class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.ones(64,device='cuda'))
    def forward(self,x):return x.sin()*self.scale
layers=[Layer().eval(),Layer().eval()]
cache=module.ConditioningCache.__new__(module.ConditioningCache)
cache.disabled=cache.filling=cache.validating=False
cache.pool=torch.cuda.graph_pool_handle() # Required only by the preserved old source.
wrappers=[cache.wrap_layer(layer.forward,layer.forward,i) for i,layer in enumerate(layers)]
held=[]
try:
    with torch.no_grad():
        for cycle in range(3):
            cache.active=dict(graphs={},graph_stats={})
            for request in range(4):
                x=torch.full((32,64),float(cycle+request),device='cuda')
                for layer,wrapper in zip(layers,wrappers):
                    actual=wrapper(x);expected=layer(x)
                    assert torch.equal(actual,expected)
            torch.cuda.synchronize()
            r['cycles'].append(dict(cycle=cycle,stats=list(cache.active['graph_stats'].values())))
            # Retain graph output storage beyond graph destruction, as residual
            # or request references can. Reusing its retired pool must be avoided.
            for graph in cache.active['graphs'].values():
                held.extend(entry[1] for entry in graph.entries.values())
            del graph
            cache.active=None;gc.collect()
    stats=[s for c in r['cycles'] for s in c['stats']]
    r.update(status='success',captures=sum(s['captures'] for s in stats),
             checks=sum(s['checks'] for s in stats),replays=sum(s['replays'] for s in stats),
             rejected=[e for s in stats for e in s['rejected']])
except Exception as error:r.update(status='error',error=repr(error))
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
