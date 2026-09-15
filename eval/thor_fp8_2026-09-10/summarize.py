"""Validate all measured pairs before producing a publishable latency table."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
manifest=json.loads((r/'source-v1.json').read_text())['files']
for name,digest in manifest.items():assert hashlib.sha256((r/'source-v1'/name).read_bytes()).hexdigest()==digest,name
final=json.loads((r/'source-final.json').read_text());assert final['ok'] and final['source_files']==len(manifest) and final['compiled_libraries']==4
assert final['source_manifest_sha256']==hashlib.sha256((r/'source-v1.json').read_bytes()).hexdigest()
assert final['compiled_manifest_sha256']==hashlib.sha256((r/'compiled-libraries.json').read_bytes()).hexdigest()
assert final['progress_sha256']==hashlib.sha256((r/'progress.json').read_bytes()).hexdigest()
manifest2=json.loads((r/'source-v2.json').read_text())['files']
assert final['source_v2_manifest_sha256']==hashlib.sha256((r/'source-v2.json').read_bytes()).hexdigest()
for name,digest in manifest2.items():assert hashlib.sha256((r/'source-v2'/name).read_bytes()).hexdigest()==digest,name
assert (r/'source-v1/benchmark.py').read_text().replace('depth>16','depth>128')==(r/'source-v2/benchmark.py').read_text()
assert {k for k in manifest if manifest[k]!=manifest2[k]}=={'benchmark.py'}
manifest3=json.loads((r/'source-v3.json').read_text())['files']
assert final['source_v3_manifest_sha256']==hashlib.sha256((r/'source-v3.json').read_bytes()).hexdigest()
for name,digest in manifest3.items():assert hashlib.sha256((r/'source-v3'/name).read_bytes()).hexdigest()==digest,name
assert (r/'source-v2/benchmark.py').read_text().replace("'dreamzero','pi05'","'dreamzero','pi05','eval_utils'")==(r/'source-v3/benchmark.py').read_text()
assert {k for k in manifest2 if manifest2[k]!=manifest3[k]}=={'benchmark.py'}
families=['va','va_2v4a','vla4','vla2','edge','nano','pi05','groot','dreamzero'];rows=[]
for family in families:
 reports={}
 for precision in ('native','fp8'):
  path=r/f'{family}-{precision}.json';d=json.loads(path.read_text())
  assert d['ok'] is True and d['family']==family and d['precision']==precision
  assert 'Thor' in d['device']
  assert d['benchmark_sha256'] in (manifest['benchmark.py'],manifest2['benchmark.py'],manifest3['benchmark.py'])
  data=path.with_suffix('.npz');assert hashlib.sha256(data.read_bytes()).hexdigest()==d['actions_sha256']
  actions=np.load(data)['actions'];assert np.isfinite(actions).all() and len(actions)==len(d['calls'])
  times=[c['ms'] for c in d['calls'] if c['phase']=='measured'];assert len(times)==d['measured_count']
  assert all(0<t<1000000 for t in times) and np.median(times)==d['p50_ms']
  assert len(times)==(9 if family.startswith('va') or family=='dreamzero' else 12)
  if precision=='fp8':assert d['e4m3_tensors']
  reports[precision]=(d,actions,hashlib.sha256(path.read_bytes()).hexdigest())
 n,na,nh=reports['native'];f,fa,fh=reports['fp8']
 for key in ('model_id','revision','device','torch','physical_gpu','input_archive_sha256','default_schedule','schedule_override','guidance','history_feedback'):
  assert n[key]==f[key],(family,key,n[key],f[key])
 assert na.shape==fa.shape
 for nc,fc in zip(n['calls'],f['calls']):
  for key in ('i','cycle','phase','shape'):assert nc[key]==fc[key],(family,key)
 for source in n['sources'].keys() & f['sources'].keys():assert n['sources'][source]==f['sources'][source],source
 rows.append({'family':family,'model_id':n['model_id'],'revision':n['revision'],'native_p50_ms':n['p50_ms'],'fp8_p50_ms':f['p50_ms'],'speedup':n['p50_ms']/f['p50_ms'],'measured_calls_per_arm':n['measured_count'],'e4m3_tensors':len(f['e4m3_tensors']),'native_numeric_environment':n['numeric_environment'],'fp8_numeric_environment':f['numeric_environment'],'native_benchmark_sha256':n['benchmark_sha256'],'fp8_benchmark_sha256':f['benchmark_sha256'],'native_receipt_sha256':nh,'fp8_receipt_sha256':fh,'returned_action_mae':float(np.abs(na.astype(np.float64)-fa.astype(np.float64)).mean()),'returned_action_max_abs_delta':float(np.abs(na.astype(np.float64)-fa.astype(np.float64)).max()),'quality_scope':'Numerical output deltas only; not simulator task accuracy'})
report={'ok':True,'source_manifest_sha256':hashlib.sha256((r/'source-v1.json').read_bytes()).hexdigest(),'rows':rows,'scope':'Matched Thor public Runtime native vs model-specific FP8 latency; no task-quality certificate'}
a.output.write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(row['family'],round(row['native_p50_ms'],2),'->',round(row['fp8_p50_ms'],2),'ms',round(row['speedup'],3),'x')
