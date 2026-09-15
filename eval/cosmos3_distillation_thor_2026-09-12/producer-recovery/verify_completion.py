from pathlib import Path
import json,hashlib
p=Path('/home/ubuntu/InstinctCompress/results/cosmos3_droid/realtime_v16/completion.json')
blob=p.read_bytes();d=json.loads(blob)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def check(r):
 p=Path(r['path']);assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],p
assert d['status']=='success'
assert len(d['verified_artifacts'])==313
for r in d['verified_artifacts']:check(r)
for r in d['artifacts'].values():check(r)
assert {x['arm_id'] for x in d['candidates']}=={f'edge_sde1_cfg{cfg}_seed{seed}' for cfg in (1,4) for seed in (12031,12032)}
for c in d['candidates']:
 assert c['steps']==1 and c['times']==[1.,0.]
 check(c['export_gate'])
 gate=json.loads(Path(c['export_gate']['path']).read_text())
 assert gate['status']=='success' and gate['native_serving_bitexact'] and gate['cold_reload_bitexact']
 assert gate['arm_id']==c['arm_id'] and gate['student_updates']==64 and gate['state_key']=='student'
audit=json.loads(Path(d['artifacts']['quality_audit']['path']).read_text())
assert audit['status']=='success' and audit['valid'] is True
assert d['artifacts']['quality_analysis']['sha256']=='449ac8319443c5fedbbdc68d1c4e97cdcb1d512d991f53fa900075d98669ad95'
assert p.read_bytes()==blob
out=dict(status='success',quality_certified=False,producer_completion=dict(path=str(p),sha256=hashlib.sha256(blob).hexdigest()),
 verified_artifacts=313,verified_bytes=sum(x['bytes'] for x in d['verified_artifacts']),
 producer_artifacts=d['artifacts'],candidates=d['candidates'],limits=d['limits'],recovery=d['recovery'],
 scope='Independent Flash rehash of producer completion inventory and four final64 native/cold gates; historical native quality only')
Path('/home/ubuntu/ifl_cosmos_quality_preflight/producer_completion_verification_v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],verified_artifacts=313,verified_bytes=out['verified_bytes'])))
