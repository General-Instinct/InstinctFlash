"""Audited continuation: retain 13 completed trials; retry only zero-call expert rejection.

No simulator/model source changes. At most three new attempts per remaining seed.
Any policy call, timeout, unexpected exception, or evidence mismatch stops recovery.
"""
import copy,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source-robotwin-v2'))
from benchmarks.vla.util import load_json,sha256_file,sha256_json,write_json_atomic
from run_robotwin_arm import preflight
from verify_robotwin_progress import collect_progress
from verify_robotwin_episode import verify as original_verify
from verify_robotwin_episode_v2 import verify
OLD=ROOT/'robotwin-fp8-run'
OUT=ROOT/'robotwin-fp8-recovery-v1'

def retryable(path,identity,seed):
    """Fail closed; this predicate never admits a trial that reached the model."""
    trace=path.with_suffix('.episode')/'trace/trace.json'
    if path.exists() or not trace.is_file():return False
    data=load_json(trace)
    log=path.with_suffix('.log').read_text()
    return (data.get('complete') is True and data.get('calls')==[] and
            data.get('identity')==identity and
            f'expert gate rejected pinned seed {seed} ' in log and
            'exhausted after 0/1 accepted episodes' in log and
            'RuntimeError: [InstinctWM] pinned seed list' in log)

def audit_parent():
    report=collect_progress('fp8')
    assert report['verified_episodes']==13 and report['successes']==11
    old=load_json(OLD/'progress.json')
    assert old['status']=='failed' and len(old['jobs'])==14
    assert sha256_file(OLD/'progress.json')=='9a912c9821cc7ceba8eb5de2a8c8b798a0e8d3072381e21126f6ecf3c1d192e1'
    assert retryable(OLD/old['jobs'][-1]['result'],old['identity'],110101)
    return old

def main():
    index,scenes=preflight();old=audit_parent()
    OUT.mkdir(exist_ok=False)
    p=copy.deepcopy(old);p.pop('error',None)
    p.update(status='running',jobs=p['jobs'][:13],
             original_runner_sha256=old['runner_sha256'],runner_sha256=sha256_file(Path(__file__)),
             recovery={'parent_progress_sha256':sha256_file(OLD/'progress.json'),
                       'parent_path':str(OLD),'imported_episodes':13,
                       'policy':'same pinned seed, at most 3 fresh attempts; retry only explicit expert rejection with complete empty policy trace',
                       'attempts':[]})
    for row in p['jobs']:
        for suffix in ('.json','.episode','.log'):
            name=Path(row['result']).with_suffix(suffix).name
            (OUT/name).symlink_to(OLD/name)
    def save():write_json_atomic(OUT/'progress.json',p)
    save()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='2',PYTHONHASHSEED='0',PYTHONUNBUFFERED='1',
             PYTHONPATH=str(ROOT/'source-robotwin-v2'),ROBOTWIN_ROOT='/home/ubuntu/RoboTwin',LINGBOT_ROOT='/home/ubuntu/lingbot-va')
    try:
        for planned in index['jobs'][13:]:
            job=ROOT/'robotwin-jobs'/planned['job'];assert sha256_file(job)==planned['sha256']
            scene=scenes['scenes'][planned['scene_key']]
            row={'scene_key':planned['scene_key'],'result':planned['job'],'verified':False}
            p['jobs'].append(row);save()
            for number in range(1,4):
                attempt=OUT/'attempts'/Path(planned['job']).stem/str(number);attempt.mkdir(parents=True,exist_ok=False)
                result=attempt/planned['job']
                command=['/home/ubuntu/RoboTwin/.venv/bin/python','-m','benchmarks.vla.wan_va_runtime_robotwin_scene',
                         '--job',str(job),'--scenes',str(ROOT/'robotwin-scenes.json'),
                         '--identity',str(ROOT/'robotwin-fp8-serve-v1.json'),'--output',str(result),
                         '--robotwin','/home/ubuntu/RoboTwin','--lingbot','/home/ubuntu/lingbot-va','--endpoint','ws://127.0.0.1:19062']
                record={'scene_key':planned['scene_key'],'attempt':number,'result':str(result),'status':'running'}
                p['recovery']['attempts'].append(record);save()
                with result.with_suffix('.log').open('x') as log:
                    run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                record.update(exit_code=run.returncode,log_sha256=sha256_file(result.with_suffix('.log')))
                trace=result.with_suffix('.episode')/'trace/trace.json'
                if trace.exists():record['trace_sha256']=sha256_file(trace)
                if run.returncode:
                    allowed=retryable(result,p['identity'],scene['resolved_seed'])
                    record['status']='expert_rejected_before_policy' if allowed else 'operational_failure';save()
                    assert allowed and number<3, f'Recovery stopped: {planned["scene_key"]}, attempt {number}'
                    continue
                raw=load_json(result)
                assert raw['scene_manifest_sha256']==p['scene_sha256']
                assert raw['source_request_sha256']==sha256_json(load_json(job)['request'])
                checked=verify(result,p['identity'],scenes)
                assert checked['scene_key']==planned['scene_key']
                row.update(original_verify(result,p['identity'],scenes),verified=True,exit_code=0)
                record.update(status='verified',result_sha256=sha256_file(result))
                for suffix in ('.json','.episode','.log'):
                    name=Path(planned['job']).with_suffix(suffix).name
                    (OUT/name).symlink_to(attempt/name)
                save();print('VERIFIED',planned['scene_key'],checked,flush=True)
                break
        p['status']='complete';save();write_json_atomic(OUT/'complete.json',p)
    except BaseException as e:
        p.update(status='failed',error=repr(e));save();raise

if __name__=='__main__':main()
