"""Aggregate actual CUDA events without double-counting CPU operator totals."""
import argparse
import collections
import gzip
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
summary={}
for family in ('edge','nano'):
    path=a.root/f'{family}-profile-20260912.json'
    r=json.loads(path.read_text() if path.exists() else gzip.decompress(path.with_suffix('.json.gz').read_bytes()))
    groups=collections.Counter(); names=collections.Counter(); counts=collections.Counter()
    for e in r['cuda_events']:
        name=e['name']; ms=e['us']/1000
        names[name]+=ms; counts[name]+=1
        if 'fmha' in name.lower(): group='attention'
        elif any(t in name.lower() for t in ('nvjet','gemm','gemv')): group='matrix_and_convolution'
        else: group='other'
        groups[group]+=ms
    total=sum(groups.values())
    summary[family]={'unprofiled_ms':r['unprofiled_ms'],'summed_cuda_event_ms':total,
        'groups':{k:{'ms':v,'fraction':v/total} for k,v in groups.items()},
        'top_kernels':[dict(name=k,ms=v,count=counts[k]) for k,v in names.most_common(20)],
        'cache_admitted':r['stats']['conditioning_cache_status']['admitted'],
        'cache_disabled':r['stats']['conditioning_cache']['disabled'],
        'attention_2x_estimated_overall_speedup':1/(1-groups['attention']/total/2),
        'notes':'Diagnostic event-time decomposition, not a paired speed certificate. Consecutive requests advance the service RNG; profile_action_byte_equal is not a controlled equivalence check.'}
(a.root/'profile-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:{n:v for n,v in r.items() if n!='top_kernels'} for k,r in summary.items()},indent=2))
