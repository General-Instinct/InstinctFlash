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
versions={}
for path in sorted(r.glob('source-v*.json')):
 version=path.stem; files=json.loads(path.read_text())['files']
 assert final['source_versions'][version]['manifest_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
 for name,digest in files.items():assert hashlib.sha256((r/version/name).read_bytes()).hexdigest()==digest,(version,name)
 versions[version]=files
jobs=json.loads((r/'progress.json').read_text())['jobs']
families=['va','va_2v4a','vla4','vla2','edge','nano','pi05','groot','dreamzero'];rows=[]
for family in families:
 reports={}
 for precision in ('native','fp8'):
  path=r/f'{family}-{precision}.json';d=json.loads(path.read_text())
  assert d['ok'] is True and d['family']==family and d['precision']==precision
  assert 'Thor' in d['device']
  job=next(j for j in jobs if j['family']==family and j['precision']==precision)
  version=job.get('source','source-v1');active=versions[version]
  assert d['benchmark_sha256']==active['benchmark.py']
  for source,digest in d['sources'].items():
   marker='/'+version+'/'
   if marker in source:assert active[source.split(marker,1)[1]]==digest,source
  data=path.with_suffix('.npz');assert hashlib.sha256(data.read_bytes()).hexdigest()==d['actions_sha256']
  actions=np.load(data)['actions'];assert np.isfinite(actions).all() and len(actions)==len(d['calls'])
  times=[c['ms'] for c in d['calls'] if c['phase']=='measured'];assert len(times)==d['measured_count']
  assert all(0<t<1000000 for t in times) and np.median(times)==d['p50_ms']
  assert len(times)==(9 if family.startswith('va') or family=='dreamzero' else 12)
  if precision=='fp8':
   assert d['e4m3_tensors']
   if family in ('vla4','vla2'):
    graph=d['graph_stats']
    assert graph['vision_graph'] and graph['vision_self_check']['passed'], (family,graph)
   if family=='groot':
    stats=d['backend_stats']
    assert stats['text_graphs']==stats['fp8_recipe']['graph_boundaries']>0
    assert not stats['text_graph_rejections']
   if family in ('edge','nano'):
    assert 'qkv_dense_mlp_' in d['backend_stats']['fp8_recipe']['recipe']
   if family=='dreamzero':
    assert 'qkv_ffn_' in d['backend_stats']['fp8_recipe']['recipe']
  if family=='pi05':
   assert d['input_camera_format']=='float32_CHW_0_1'
   assert d['observation_camera_keys']==['observation.images.image','observation.images.image2']
   if precision=='fp8':assert d['active_camera_counts']==[2]
  d['source_version']=version
  reports[precision]=(d,actions,hashlib.sha256(path.read_bytes()).hexdigest())
 n,na,nh=reports['native'];f,fa,fh=reports['fp8']
 for key in ('source_version','model_id','revision','device','torch','physical_gpu','input_archive_sha256','default_schedule','schedule_override','guidance','history_feedback'):
  assert n[key]==f[key],(family,key,n[key],f[key])
 assert na.shape==fa.shape
 for nc,fc in zip(n['calls'],f['calls']):
  for key in ('i','cycle','phase','shape'):assert nc[key]==fc[key],(family,key)
 for source in n['sources'].keys() & f['sources'].keys():assert n['sources'][source]==f['sources'][source],source
 measured_indices=[i for i,c in enumerate(n['calls']) if c['phase']=='measured']
 delta=np.abs(na[measured_indices].astype(np.float64)-fa[measured_indices].astype(np.float64))
 rows.append({'family':family,'source_version':n['source_version'],'model_id':n['model_id'],'revision':n['revision'],'native_p50_ms':n['p50_ms'],'fp8_p50_ms':f['p50_ms'],'speedup':n['p50_ms']/f['p50_ms'],'measured_calls_per_arm':n['measured_count'],'e4m3_tensors':len(f['e4m3_tensors']),'native_numeric_environment':n['numeric_environment'],'fp8_numeric_environment':f['numeric_environment'],'native_benchmark_sha256':n['benchmark_sha256'],'fp8_benchmark_sha256':f['benchmark_sha256'],'native_receipt_sha256':nh,'fp8_receipt_sha256':fh,'returned_action_mae':float(delta.mean()),'returned_action_max_abs_delta':float(delta.max()),'quality_scope':'Measured-call numerical output deltas only; not simulator task accuracy'})
report={'ok':True,'source_manifest_sha256':hashlib.sha256((r/'source-v1.json').read_bytes()).hexdigest(),'rows':rows,'scope':'Matched Thor public Runtime native vs model-specific FP8 latency; no task-quality certificate'}
a.output.write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(row['family'],round(row['native_p50_ms'],2),'->',round(row['fp8_p50_ms'],2),'ms',round(row['speedup'],3),'x')
