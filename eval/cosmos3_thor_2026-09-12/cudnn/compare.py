"""Compare isolated cuDNN candidate and fresh native reference; no quality gate."""
import argparse, hashlib, json, statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('family',choices=('edge','nano'));p.add_argument('--archived-reference',type=Path);a=p.parse_args()
output=a.root/f'{a.family}-comparison.json'
output.write_text(json.dumps(dict(validation_complete=False))+'\n')
reports={};arrays={}
for arm in ('model','reference'):
    path=a.root/f'{a.family}-cudnn-{arm}.json'
    r=json.loads(path.read_text());assert r['ok'] and not r['competing_gpu_processes']
    archive=path.with_suffix('.npz')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['actions_sha256']
    with np.load(archive) as data:arrays[arm]=data['actions'].copy()
    assert arrays[arm].shape==(16,32,8) and np.isfinite(arrays[arm]).all()
    cache=r['backend_stats']['conditioning_cache']
    assert r['backend_stats']['conditioning_cache_status']['admitted'] and not cache['disabled'] and not cache['rejected']
    assert cache['verified_tensors']>0 and sum(g['replays'] for g in cache['graph_stats'])>0
    assert not any(g['rejected'] for g in cache['graph_stats'])
    reports[arm]=r
ref,cand=reports['reference'],reports['model']
for key in ('family','arm','model_id','revision','precision','torch','device','input_archive_sha256','reference_constructor_sha256','benchmark_sha256','cases','effective_schedule','numeric_environment','sources'):
    assert ref[key]==cand[key],key
assert cand['cudnn_experiment']['eligible_python_calls']>0
assert arrays['reference'].dtype==arrays['model'].dtype
rows={}
for arm,r in reports.items():
    times=[c['ms'] for c in r['calls'] if c['phase']=='measured']
    assert len(times)==10
    rows[arm]=dict(p50_ms=statistics.median(times),p95_ms=float(np.percentile(times,95)),peak_allocated_bytes=r['peak_allocated_bytes'])
delta=np.abs(arrays['model'].astype(np.float64)-arrays['reference'].astype(np.float64))
result=dict(validation_complete=True,family=a.family,scope='Fresh isolated-process numeric screen; one candidate/reference pair, no A/B/A drift gate or closed-loop quality certificate',matched_protocol=True,finite_complete_actions=True,rows=rows,speedup=rows['reference']['p50_ms']/rows['model']['p50_ms'],action_bytes_equal=arrays['model'].tobytes()==arrays['reference'].tobytes(),max_abs_action_delta=float(delta.max()),mean_abs_action_delta=float(delta.mean()),per_dimension_max_abs=delta.max(axis=(0,1)).tolist(),task_quality_certified=False,production_promoted=False)
if a.archived_reference:
    old=json.loads(a.archived_reference.read_text())
    archive=a.archived_reference.with_suffix('.npz')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==old['actions_sha256']
    for key in ('sources','cases','revision','effective_schedule','input_archive_sha256','numeric_environment'):
        assert old[key]==ref[key],key
    with np.load(archive) as data: prior=data['actions'].copy()
    result['fresh_reference_matches_archived_bytes']=(prior.dtype==arrays['reference'].dtype and prior.shape==arrays['reference'].shape and prior.tobytes()==arrays['reference'].tobytes())
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
