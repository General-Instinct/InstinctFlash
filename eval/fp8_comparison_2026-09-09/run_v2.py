"""Sequential V2 campaign, after other Thor measurements release the GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(sys.argv[1]).resolve()
while not (root/'vla4-complete.json').exists():time.sleep(15)
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
base=Path.home()/'ifl_eval/next_steps_20260906'
env['QWEN3VL_PATH']=str(base/'qwen-config')
env['TRITON_CACHE_DIR']=str(base/'cache/triton')
env['TRITON_PTXAS_PATH']='/usr/local/cuda/bin/ptxas'
env['TRITON_PTXAS_BLACKWELL_PATH']='/usr/local/cuda/bin/ptxas'
env['PYTHONPATH']=str(root/'source')+':'+str(Path.home()/'ifl_eval/native_qualification_20260906/final-execution')
status=[]
arms=['stock','capture','engine8']
for repeat in range(3):
    for arm in arms[repeat:]+arms[:repeat]:
        label=f'v2.{arm}.{repeat}'
        command=[str(base/'run-env/bin/python'),str(root/'source/probe_v2.py'),'--arm',arm,'--repeat',str(repeat),'--root',str(root)]
        print('START',label,flush=True);start=time.time()
        with (root/'results'/f'{label}.log').open('x') as stream:
            try:code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired:code=124
        status.append(dict(label=label,command=command,exit_code=code,elapsed_s=time.time()-start))
        (root/'v2-status.json').write_text(json.dumps(status,indent=2))
        print('DONE',label,code,flush=True)
(root/'v2-complete.json').write_text(json.dumps(dict(attempts=len(status),failures=sum(x['exit_code']!=0 for x in status))))
