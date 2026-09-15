"""Require full finite action identity and a meaningful A/candidate/B speed gain."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('family',choices=('edge','nano'));p.add_argument('--minimum-speedup',type=float,default=1.02);p.add_argument('--archived-reference',type=Path);a=p.parse_args()
assert np.isfinite(a.minimum_speedup) and a.minimum_speedup>0
output=a.root/f'{a.family}-comparison.json';output.write_text(json.dumps(dict(passed=False,validation_complete=False))+'\n')
reports={};arrays={}
for arm in ('baseline_a','host','baseline_b'):
    path=a.root/f'{a.family}-scheduler-{arm}.json';r=json.loads(path.read_text())
    assert r['ok'] and not r['competing_gpu_processes'],arm
    archive=path.with_suffix('.npz');assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['actions_sha256']
    with np.load(archive) as d:arrays[arm]=d['actions'].copy()
    assert arrays[arm].shape==(16,32,8) and np.isfinite(arrays[arm]).all()
    s=r['host_timesteps'];assert s['enabled']==(arm=='host') and s['set_timesteps_calls']==16
    assert set(s['devices'])==({'cpu'} if arm=='host' else {'cuda'})
    c=r['backend_stats']['conditioning_cache'];assert r['backend_stats']['conditioning_cache_status']['admitted']
    assert not c['disabled'] and not c['rejected'] and c['verified_tensors']>0
    assert sum(g['replays'] for g in c['graph_stats'])>0 and not any(g['rejected'] for g in c['graph_stats'])
    reports[arm]=r
ref=reports['baseline_a'];rows=[]
for arm,r in reports.items():
    for key in ('family','arm','model_id','revision','precision','torch','device','input_archive_sha256','reference_constructor_sha256','benchmark_sha256','cases','effective_schedule','numeric_environment','sources','scheduler_experiment_sources'):
        assert ref[key]==r[key],(arm,key)
    assert arrays[arm].dtype==arrays['baseline_a'].dtype
    delta=np.abs(arrays[arm].astype(float)-arrays['baseline_a'].astype(float))
    times=[c['ms'] for c in r['calls'] if c['phase']=='measured'];assert len(times)==10
    rows.append(dict(arm=arm,p50_ms=statistics.median(times),p95_ms=float(np.percentile(times,95)),action_bytes_equal=arrays[arm].tobytes()==arrays['baseline_a'].tobytes(),max_abs_delta=float(delta.max())))
speedups=[rows[i]['p50_ms']/rows[1]['p50_ms'] for i in (0,2)]
drift=max(rows[i]['p50_ms'] for i in (0,2))/min(rows[i]['p50_ms'] for i in (0,2))-1
exact=all(r['action_bytes_equal'] for r in rows)
perf=min(speedups)>=a.minimum_speedup and drift<=.05 and rows[1]['p95_ms']<=max(rows[i]['p95_ms'] for i in (0,2))*1.10
result=dict(validation_complete=True,family=a.family,rows=rows,speedup_vs_a=speedups[0],speedup_vs_b=speedups[1],reference_drift=drift,action_bytes_passed=exact,performance_passed=perf,passed=exact and perf,thresholds=dict(minimum_speedup=a.minimum_speedup,maximum_reference_drift=.05,maximum_p95_regression=.10),scope='Recorded-input paired native experiment; no closed-loop task-quality certificate')
if a.archived_reference:
    old=json.loads(a.archived_reference.read_text());archive=a.archived_reference.with_suffix('.npz')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==old['actions_sha256']
    for key in ('sources','cases','revision','effective_schedule','input_archive_sha256','numeric_environment'):
        assert old[key]==ref[key],key
    with np.load(archive) as data:prior=data['actions'].copy()
    same=prior.shape==arrays['baseline_a'].shape and prior.dtype==arrays['baseline_a'].dtype and prior.tobytes()==arrays['baseline_a'].tobytes()
    result['reference_matches_archived_bytes']=same
    result['passed']=result['passed'] and same
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit('Scheduler candidate failed action/performance gate; inspect saved receipt')
