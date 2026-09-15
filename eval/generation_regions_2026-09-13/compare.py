"""Verify full-action GEN-region receipts and report numerical deltas."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for family,layers in [('edge',28),('nano',36)]:
 paths=[a.results/f'{family}-{arm}.json' for arm in ('reference','generation-regions')]
 reports=[json.loads(p.read_text()) for p in paths]
 assert all(r['ok'] for r in reports),family
 assert reports[0]['cases']==reports[1]['cases']
 assert reports[0]['revision']==reports[1]['revision']
 assert reports[0]['effective_schedule']==reports[1]['effective_schedule']
 assert not any(r['competing_gpu_processes'] for r in reports)
 stats=reports[1]['backend_stats']['generation_regions']
 assert not stats['rejected'] and stats['eager_checks']>=layers*2
 assert stats['compiled_calls']>0 and len(stats['regions'])==layers
 assert all(r['state']=='ready' and r['calls']>0 for r in stats['regions'])
 arrays=[np.load(p.with_suffix('.npz'))['actions'] for p in paths]
 assert arrays[0].shape==arrays[1].shape==(16,32,8)
 assert all(np.isfinite(x).all() for x in arrays)
 for r,p in zip(reports,paths):assert hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()==r['actions_sha256']
 latency=[float(np.median([c['ms'] for c in r['calls'] if c['phase']=='measured'])) for r in reports]
 delta=np.abs(arrays[0].astype(float)-arrays[1].astype(float))
 rows.append(dict(family=family,reference_p50_ms=latency[0],candidate_p50_ms=latency[1],
                  latency_reduction_percent=100*(1-latency[1]/latency[0]),speedup=latency[0]/latency[1],
                  extraction_checks=stats['eager_checks'],compiled_calls=stats['compiled_calls'],
                  final_actions_byte_equal=arrays[0].dtype==arrays[1].dtype and arrays[0].tobytes()==arrays[1].tobytes(),
                  action_mae=float(delta.mean()),action_max_abs=float(delta.max()),
                  joint_mae=float(delta[...,:7].mean()),gripper_mae=float(delta[...,7:].mean()),
                  schedule=reports[0]['effective_schedule'],
                  receipts=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
report=dict(status='complete_numerical_latency_screen',quality_certified=False,default_enabled=False,rows=rows)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows,indent=2))
