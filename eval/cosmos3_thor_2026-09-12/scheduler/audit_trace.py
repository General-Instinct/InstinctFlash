"""Summarize host synchronization waits separately from actual CUDA execution."""
import argparse,collections,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not a.output.exists()
r=json.loads(a.trace.read_text());durations=collections.Counter();counts=collections.Counter()
for event in r['traceEvents']:
    name=event.get('name','')
    if 'Synchronize' in name or 'Memcpy DtoH' in name or name in ('aten::item','aten::nonzero','aten::_local_scalar_dense'):
        durations[name]+=event.get('dur',0)/1000;counts[name]+=1
result=dict(trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),events=[dict(name=k,count=counts[k],total_ms=v) for k,v in durations.items()],note='Nested CPU ATen events and synchronization waits overlap GPU execution. Do not sum these rows or interpret wait time as removable overhead.')
a.output.write_text(json.dumps(result,indent=2)+'\n')
