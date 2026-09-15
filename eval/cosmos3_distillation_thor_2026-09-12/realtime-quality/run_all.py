"""Resume only validated complete banks; run remaining frozen variants serially."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from validate_capture import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study_root',type=Path)
    p.add_argument('--capture-source',type=Path,required=True)
    p.add_argument('--status',type=Path,required=True)
    a=p.parse_args()
    root=a.study_root.resolve();source=a.capture_source.resolve()
    plan=json.loads((source/'plan.json').read_text())
    if a.status.exists():raise FileExistsError('Use a fresh scheduler status; do not overwrite execution history')
    state=dict(status='running',scheduler_pid=os.getpid(),active_job=None,completed=[],quality_certified=False)
    def write():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        temporary=a.status.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(a.status)
    write()
    try:
        for variant in plan['variants']:
            config=variant['configuration_id'];attention=variant['attention']
            arm=config.removeprefix('edge_sde1_').removesuffix('_update0064').replace('_','-')
            output=root/f'quality-v16-{arm}-{attention}-v1'
            if output.exists():
                result=validate(output,source)
                assert result['configuration_id']==config and result['attention']==attention
                state['completed'].append(dict(**variant,output=str(output),validation=result,reused=True))
                write();continue
            args=[sys.executable,str(source/'run_capture.py'),str(root/f'student-v16-{arm}'),str(output),
                '--data',str(source/'development.npz'),'--configuration',config,'--attention',attention]
            with output.with_suffix('.log').open('x') as log:
                child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                state['active_job']=dict(**variant,pid=child.pid,output=str(output),argv=args)
                write();print(json.dumps(state['active_job']),flush=True)
                while child.poll() is None:
                    time.sleep(5);write()
                if child.returncode!=0:
                    state['active_job']['exit_code']=child.returncode
                    raise RuntimeError(f'Quality capture failed: {config}/{attention}, exit {child.returncode}')
            result=validate(output,source)
            (output/'capture_validation.json').write_text(json.dumps(result,indent=2)+'\n')
            state['completed'].append(dict(**variant,output=str(output),validation=result,reused=False))
            state['active_job']=None;write()
        state['status']='success';write()
    except BaseException as error:
        state.update(status='interrupted_or_failed',error=f'{type(error).__name__}: {error}')
        write();raise


if __name__=='__main__':main()
