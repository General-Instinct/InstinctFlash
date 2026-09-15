"""Validate all measured pairs before producing a publishable latency table."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
manifest=json.loads((r/'source-v1.json').read_text())['files']
for name,digest in manifest.items():assert hashlib.sha256((r/'source-v1'/name).read_bytes()).hexdigest()==digest,name
families=['va','va_2v4a','vla4','vla2','edge','nano','pi05','groot','dreamzero'];rows=[]
for family in families:
 reports={}
 for precision in ('native','fp8'):
  path=r/f'{family}-{precision}.json';d=json.loads(path.read_text())
  assert d['ok'] is True and d['family']==family and d['precision']==precision
  assert 'H100' in d['device']
  assert d['benchmark_sha256']==manifest['benchmark.py']
  data=path.with_suffix('.npz');assert hashlib.sha256(data.read_bytes()).hexdigest()==d['actions_sha256']
  actions=np.load(data)['actions'];assert np.isfinite(actions).all() and len(actions)==len(d['calls'])
  times=[c['ms'] for c in d['calls'] if c['phase']=='measured'];assert len(times)==d['measured_count']
  assert all(0<t<1000000 for t in times) and np.median(times)==d['p50_ms']
  assert len(times)==(16 if family.startswith('va') or family=='dreamzero' else 20)
  if precision=='fp8':assert d['backend_stats']['fp8_recipe']['projections']
  reports[precision]=(d,actions,hashlib.sha256(path.read_bytes()).hexdigest())
 n,na,nh=reports['native'];f,fa,fh=reports['fp8']
 for key in ('model_id','revision','device','torch','physical_gpu','input_archive_sha256','benchmark_sha256','default_schedule','schedule_override','guidance','numeric_environment','history_feedback'):
  assert n[key]==f[key],(family,key,n[key],f[key])
 assert na.shape==fa.shape
 for nc,fc in zip(n['calls'],f['calls']):
  for key in ('i','cycle','phase','shape'):assert nc[key]==fc[key],(family,key)
 for source in n['sources'].keys() & f['sources'].keys():assert n['sources'][source]==f['sources'][source],source
 rows.append({'family':family,'model_id':n['model_id'],'revision':n['revision'],'native_p50_ms':n['p50_ms'],'fp8_p50_ms':f['p50_ms'],'speedup':n['p50_ms']/f['p50_ms'],'measured_calls_per_arm':n['measured_count'],'fp8_projections':len(f['backend_stats']['fp8_recipe']['projections']),'native_receipt_sha256':nh,'fp8_receipt_sha256':fh,'returned_action_mae':float(np.abs(na.astype(np.float64)-fa.astype(np.float64)).mean()),'returned_action_max_abs_delta':float(np.abs(na.astype(np.float64)-fa.astype(np.float64)).max()),'quality_scope':'Numerical output deltas only; not simulator task accuracy'})
report={'ok':True,'source_manifest_sha256':hashlib.sha256((r/'source-v1.json').read_bytes()).hexdigest(),'rows':rows,'scope':'Matched public Runtime H100 latency; separate Q/K/V FP8 executor, not Thor fused kernels; no H100 closed-loop certificate'}
a.output.write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(row['family'],round(row['native_p50_ms'],2),'->',round(row['fp8_p50_ms'],2),'ms',round(row['speedup'],3),'x')
