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
    assert r['qualification_only'] is True
    from runtime_assets import WAN_BYTES, WAN_SHA256
    assets = r['external_runtime_assets']
    assert len(assets) == 1 and assets[0]['bytes'] == WAN_BYTES and assets[0]['sha256'] == WAN_SHA256
    declared = r['declared_execution']
    from realtime_adapter import GRIDS
    assert declared['sigmas'] == GRIDS[declared['steps']]
    assert declared['guidance'] in (1., 4.)
    branches = 1 if declared['guidance'] == 1. else 2
    assert declared['branches_per_callback'] == branches
    expected_clocks = [1000.*s for s in declared['sigmas'][:-1] for _ in range(branches)]
    clocks = r['first_request_native_branch_clocks']
    assert len(clocks) == len(expected_clocks)
    assert all(abs(x-y) <= float(np.spacing(np.float32(y))) for x,y in zip(clocks,expected_clocks))
    archive=path.with_suffix('.npz')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['actions_sha256']
    with np.load(archive) as data: arrays[arm]=data['actions'].copy()
    assert arrays[arm].shape==(16,32,8) and np.isfinite(arrays[arm]).all()
    assert len(r['calls'])==16 and [c['i'] for c in r['calls']]==list(range(16))
    assert [c['phase'] for c in r['calls']] == ['warmup']*6 + ['measured']*10
    measured = [c['ms'] for c in r['calls'][6:]]
    assert np.isclose(r['p50_ms'], np.median(measured))
    assert np.isclose(r['p95_ms'], np.percentile(measured,95))
    assert all(np.isfinite(c['ms']) and c['ms']>0 for c in r['calls'])
    stats=r['backend_stats'];steps=len(stats['sampling']['sigmas'])-1
    assert stats['sampling']['sigmas'] == declared['sigmas']
    assert stats['guidance'] == declared['guidance']
    assert stats['action_steps'] == steps and stats['action_chunk_size'] == 32
    assert stats['action_padding'] == 'zero' and stats['padding_hooks_restored']
    assert stats['padding_projection_calls'] == 16
    assert stats['padding_projected_velocity_branches'] == 16*steps*branches
    assert stats['sampler_calls']==16 and stats['velocity_evaluations']==16*steps
    assert r['execution_policy']['precision']=='native' and not r['execution_policy']['changed_schedule']
    reports[arm]=r
ref,cand=reports['native'],reports['cudnn']
for key in ('external_runtime_assets','declared_execution','checkpoint_manifest_sha256','benchmark_sha256','fixture_sha256','torch','cuda','cudnn'):
    assert ref[key]==cand[key],key
for key in ('sampling','guidance','action_steps','action_chunk_size','action_padding'):
    assert ref['backend_stats'].get(key)==cand['backend_stats'].get(key),key
assert ref['sources'] and any(k.endswith('/realtime_adapter.py') for k in ref['sources'])
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
    scope='Single matched attention pair on a declared V17 checkpoint (training status not inferred), not student-vs-teacher quality or closed-loop realtime evidence')
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
