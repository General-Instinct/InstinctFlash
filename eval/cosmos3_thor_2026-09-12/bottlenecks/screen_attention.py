"""Screen same-backend tile tuning at Cosmos generation attention geometries."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch
import natten
from natten.backends.configs.cutlass import check_cutlass_fmha_forward_config
p=argparse.ArgumentParser(); p.add_argument('output', type=Path); a=p.parse_args()
assert not a.output.exists()
assert torch.cuda.get_device_capability() == (11, 0)
rows=[]
def measure(fn):
    for _ in range(3): fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5): fn()
    times=[]
    for _ in range(5):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record(); graph.replay(); end.record(); end.synchronize()
        times.append(start.elapsed_time(end)/5)
    return statistics.median(times)
with torch.inference_mode():
    for heads,lengths in [(16,(3245,3112)),(32,(3182,3103))]:
        for n in lengths:
            for seed in (11,29):
                torch.manual_seed(seed)
                q=torch.randn(1,3094,heads,128,device='cuda',dtype=torch.bfloat16)
                k=torch.randn(1,n,8,128,device='cuda',dtype=torch.bfloat16)
                v=torch.randn_like(k)
                default=check_cutlass_fmha_forward_config(q)
                def run(config):
                    return natten.attention(q,k,v,backend='cutlass-fmha',q_tile_size=config[0],kv_tile_size=config[1])
                reference=run(default)
                for config in natten.get_configs_for_cutlass_fmha(q,k,v):
                    result=run(config)
                    rows.append(dict(heads=heads,kv_length=n,seed=seed,config=config,default=default,
                        byte_equal=bool(torch.equal(result.view(torch.uint8),reference.view(torch.uint8))),
                        finite=bool(torch.isfinite(result).all()),max_abs_delta=float((result.float()-reference.float()).abs().max()),
                        graph_ms=measure(lambda:run(config))))
                    print(json.dumps(rows[-1]),flush=True)
a.output.write_text(json.dumps(dict(diagnostic_only=True,synthetic_inputs=True,torch=torch.__version__,natten=natten.__version__,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
