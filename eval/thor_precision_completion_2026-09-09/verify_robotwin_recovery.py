"""Verify continuation provenance and all completed episodes, including retained failures."""
from recover_robotwin_fp8 import ROOT,OUT,OLD,audit_parent,retryable
from run_robotwin_arm import preflight
from verify_robotwin_episode_v2 import verify
from benchmarks.vla.util import load_json,sha256_file,sha256_json,write_json_atomic

def collect():
 index,scenes=preflight();old=audit_parent();p=load_json(OUT/'progress.json')
 assert p['runner_sha256']==sha256_file(ROOT/'recover_robotwin_fp8.py')
 assert p['original_runner_sha256']==old['runner_sha256']
 assert p['recovery']['parent_progress_sha256']==sha256_file(OLD/'progress.json')
 assert p['recovery']['imported_episodes']==13
 for k in ('identity','expected_jobs','verifier_sha256','scene_sha256','source_manifest_sha256'):assert p[k]==old[k],k
 assert p['jobs'][:13]==old['jobs'][:13]
 assert 13<=len(p['jobs'])<=40
 rows=[]
 for i,row in enumerate(p['jobs']):
  planned=index['jobs'][i];assert row['result']==planned['job']
  if not row['verified']:
   assert i==len(p['jobs'])-1;continue
  path=OUT/row['result']
  if i<13:
   assert path.resolve()==(OLD/row['result']).resolve()
  else:
   attempts=[a for a in p['recovery']['attempts'] if a['scene_key']==planned['scene_key']]
   assert 1<=len(attempts)<=3
   for n,a in enumerate(attempts,1):
    expected=OUT/'attempts'/Path(planned['job']).stem/str(n)/planned['job']
    assert a['attempt']==n and a['result']==str(expected)
    assert a['log_sha256']==sha256_file(expected.with_suffix('.log'))
    assert a['trace_sha256']==sha256_file(expected.with_suffix('.episode')/'trace/trace.json')
    if n<len(attempts):
     assert a['status']=='expert_rejected_before_policy' and a['exit_code']!=0
     assert retryable(expected,p['identity'],scenes['scenes'][planned['scene_key']]['resolved_seed'])
    else:
     assert a['status']=='verified' and a['exit_code']==0
     assert path.resolve()==expected.resolve() and a['result_sha256']==sha256_file(path)
  for suffix in ('.episode','.log'):
   assert path.with_suffix(suffix).resolve()==path.resolve().with_suffix(suffix)
  raw=load_json(path)
  assert raw['scene_manifest_sha256']==p['scene_sha256']
  assert raw['source_request_sha256']==sha256_json(load_json(ROOT/'robotwin-jobs'/planned['job'])['request'])
  checked=verify(path,p['identity'],scenes)
  assert checked['scene_key']==planned['scene_key'] and row['exit_code']==0
  assert all(row[k]==v for k,v in checked.items() if k!='scene_key')
  rows.append(checked)
 if p['status']=='complete':assert len(rows)==40 and load_json(OUT/'complete.json')==p
 report=dict(verified_episodes=len(rows),successes=sum(r['success'] for r in rows),controller_status=p['status'],rows=rows,
             recovery=p['recovery'],scope='Paired screen with explicitly audited pre-policy expert retries; no model trial retries; incomplete until 40 verified')
 write_json_atomic(ROOT/'robotwin-fp8-recovery-verified-progress.json',report)
 return report
from pathlib import Path
if __name__=='__main__':
 r=collect();print({k:v for k,v in r.items() if k not in ('rows','recovery')})
