"""Validate paired student execution and summarize speed/deltas, not task success."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args()
output=a.directory/'comparison.json'
output.write_text(json.dumps(dict(validation_complete=False))+'\n')
reports={}; arrays={}
for arm in ('native','cudnn'):
    path=a.directory/f'{arm}.json';r=json.loads(path.read_text())
    assert r['ok'] and r['attention']==arm
    archive=path.with_suffix('.npz')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['actions_sha256']
    with np.load(archive) as data: arrays[arm]=data['actions'].copy()
    assert arrays[arm].shape==(16,32,8) and np.isfinite(arrays[arm]).all()
    assert len(r['calls'])==16 and [c['i'] for c in r['calls']]==list(range(16))
    assert all(np.isfinite(c['ms']) and c['ms']>0 for c in r['calls'])
    stats=r['backend_stats'];steps=len(stats['sampling']['sigmas'])-1
    assert stats['sampler_calls']==16 and stats['velocity_evaluations']==16*steps
    assert r['execution_policy']['precision']=='native' and not r['execution_policy']['changed_schedule']
    reports[arm]=r
ref,cand=reports['native'],reports['cudnn']
for key in ('checkpoint_manifest_sha256','benchmark_sha256','fixture_sha256','torch','cuda','cudnn'):
    assert ref[key]==cand[key],key
for key in ('sampling','guidance','action_steps','action_chunk_size','action_padding'):
    assert ref['backend_stats'].get(key)==cand['backend_stats'].get(key),key
assert all(cand['sources'].get(k)==v for k,v in ref['sources'].items())
assert cand['backend_stats']['numeric_attention']['eligible_python_calls']>0
assert arrays['native'].dtype==arrays['cudnn'].dtype
assert ref['execution_policy']['category']=='BITEXACT' and cand['execution_policy']['category']=='NUMERIC'
delta=np.abs(arrays['native'].astype(np.float64)-arrays['cudnn'].astype(np.float64))
result=dict(validation_complete=True,category='SCREEN',task_quality_certified=False,
    schedule=ref['backend_stats']['sampling'],guidance=ref['backend_stats']['guidance'],
    p50_ms={k:v['p50_ms'] for k,v in reports.items()},p95_ms={k:v['p95_ms'] for k,v in reports.items()},
    speedup=ref['p50_ms']/cand['p50_ms'],max_abs_action_delta=float(delta.max()),
    mean_abs_action_delta=float(delta.mean()),per_dimension_max_abs=delta.max(axis=(0,1)).tolist(),
    scope='Single matched attention pair on a trained checkpoint, not student-vs-teacher quality or closed-loop realtime evidence')
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
