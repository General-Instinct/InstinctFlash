"""Run after pi05 finishes to keep the Thor GPU exclusive to one measured arm."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(sys.argv[1]).resolve()
while not (root/'complete.json').exists():
    time.sleep(15)
env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
env['PYTHONPATH']=str(root/'source')+':'+str(Path.home()/'ifl_eval/native_qualification_20260906/final-execution')
status=[]
arms=['stock','capture','engine8']
for repeat in range(3):
    for arm in arms[repeat:]+arms[:repeat]:
        label=f'vla4.{arm}.{repeat}'
        log=root/'results'/f'{label}.log'
        command=[str(Path.home()/'venv_vla4b/bin/python'),str(root/'source/probe_vla4.py'),'--arm',arm,'--repeat',str(repeat),'--root',str(root)]
        print('START',label,flush=True)
        start=time.time()
        with log.open('x') as stream:
            try:
                code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired:
                code=124
        status.append(dict(label=label,command=command,exit_code=code,elapsed_s=time.time()-start))
        (root/'vla4-status.json').write_text(json.dumps(status,indent=2))
        print('DONE',label,code,flush=True)
(root/'vla4-complete.json').write_text(json.dumps(dict(attempts=len(status),failures=sum(x['exit_code']!=0 for x in status))))
