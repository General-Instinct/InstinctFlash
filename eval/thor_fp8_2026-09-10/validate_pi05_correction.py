"""Validate the separately retained two-camera correction to the pi05 speed row."""
import hashlib,json
from pathlib import Path
import numpy as np

def validate(root):
 root=Path(root);reports={};benchmark=root/'pi05-two-camera-benchmark.py'
 for precision in ('native','fp8'):
  path=root/f'pi05-two-camera-{precision}.json';d=json.loads(path.read_text())
  assert d['ok'] and d['family']=='pi05' and d['precision']==precision
  assert 'Thor' in d['device'] and d['model_id']=='lerobot/pi05_libero_finetuned_v044'
  assert d['benchmark_sha256']==hashlib.sha256(benchmark.read_bytes()).hexdigest()
  assert d['input_camera_format']=='float32_CHW_0_1'
  assert d['observation_camera_keys']==['observation.images.image','observation.images.image2']
  actions=path.with_suffix('.npz');assert hashlib.sha256(actions.read_bytes()).hexdigest()==d['actions_sha256']
  a=np.load(actions)['actions'];assert len(a)==15 and np.isfinite(a).all()
  times=[c['ms'] for c in d['calls'] if c['phase']=='measured'];assert len(times)==12 and all(t>0 for t in times)
  assert float(np.median(times))==d['p50_ms']
  reports[precision]=(d,a,hashlib.sha256(path.read_bytes()).hexdigest())
 n,na,nh=reports['native'];f,fa,fh=reports['fp8'];assert f['active_camera_counts']==[2] and f['e4m3_tensors']
 for key in ('revision','device','torch','physical_gpu','input_archive_sha256','benchmark_sha256','default_schedule','schedule_override','guidance','history_feedback'):
  assert n[key]==f[key],key
 assert na.shape==fa.shape
 for key in n['sources'].keys() & f['sources'].keys():assert n['sources'][key]==f['sources'][key]
 return dict(family='pi05',model_id=n['model_id'],revision=n['revision'],native_p50_ms=n['p50_ms'],fp8_p50_ms=f['p50_ms'],speedup=n['p50_ms']/f['p50_ms'],measured_calls_per_arm=12,e4m3_tensors=len(f['e4m3_tensors']),native_numeric_environment=n['numeric_environment'],fp8_numeric_environment=f['numeric_environment'],native_receipt_sha256=nh,fp8_receipt_sha256=fh,benchmark_sha256=n['benchmark_sha256'],observation_camera_keys=n['observation_camera_keys'],active_fp8_cameras=f['active_camera_counts'],returned_action_mae=float(np.abs(na.astype(float)-fa.astype(float)).mean()),quality_scope='Two-camera latency correction, not task quality',input_camera_format=n['input_camera_format'],correction='Restore image2 second camera and supply the float32 [0,1] image range required by native preprocessing')

if __name__=='__main__':
 import sys
 r=Path(sys.argv[1]);result=validate(r);(r/'pi05-two-camera-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
