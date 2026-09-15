"""Summarize measured Nano operating-point budgets without claiming distillation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args()
reports={}; rows=[]
for name,steps,cfg in [('one-cfg3',1,3.),('one-cfg1',1,1.),('two-cfg1',2,1.)]:
    path=a.directory/f'{name}.json'; r=json.loads(path.read_text())
    assert r['ok'] and r['trained_student'] is False and r['category']=='OPERATING-POINT'
    assert (r['requested_steps'],r['requested_guidance'])==(steps,cfg)
    assert r['actual_velocity_branches']==16*steps*(1 if cfg==1 else 2)
    archive=path.with_suffix('.npz')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['actions_sha256']
    with np.load(archive) as data: x=data['actions']
    assert x.shape==(16,32,8) and np.isfinite(x).all()
    assert r['backend_stats']['numeric_attention']['eligible_python_calls']>0
    times=[c['ms'] for c in r['calls'] if c['phase']=='measured']
    assert len(times)==10 and all(np.isfinite(t) and t>0 for t in times)
    rows.append(dict(name=name,steps=steps,guidance=cfg,p50_ms=float(np.median(times)),
        p95_ms=float(np.percentile(times,95)),max_ms=max(times),
        conditional_15hz_budgets=[dict(executed_actions=n,budget_ms=1000*n/15,
            measured_misses=sum(t>1000*n/15 for t in times)) for n in (8,16,32)]))
    reports[name]=r
first=reports['one-cfg3']
for r in reports.values():
    for key in ('revision','checkpoint_index_sha256','benchmark_sha256','fixture_sha256','sources','torch','cuda','cudnn'):
        assert r[key]==first[key],key
result=dict(validation_complete=True,trained_student=False,task_quality_certified=False,
    rows=rows,scope='Original Nano weights with explicitly changed UniPC step/guidance settings. Conditional chunk budgets do not establish closed-loop realtime operation.')
(a.directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
