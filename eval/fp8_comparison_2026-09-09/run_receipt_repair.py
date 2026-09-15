"""One declared replacement for the V2 capture metadata-write failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(sys.argv[1]).resolve()
repair=root/'receipt-repair'
repair.mkdir()
(repair/'results').mkdir()
while not (root/'cwd-repair/vendor-complete.json').exists():time.sleep(15)
base=Path.home()/'ifl_eval/next_steps_20260906'
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
env['PYTHONPATH']=str(root/'source')+':'+str(Path.home()/'ifl_eval/native_qualification_20260906/final-execution')
env['QWEN3VL_PATH']=str(base/'qwen-config')
env['TRITON_CACHE_DIR']=str(base/'cache/triton')
env['TRITON_PTXAS_PATH']='/usr/local/cuda/bin/ptxas'
env['TRITON_PTXAS_BLACKWELL_PATH']='/usr/local/cuda/bin/ptxas'
command=[str(base/'run-env/bin/python'),str(root/'source/probe_v2.py'),'--arm','capture','--repeat','0','--root',str(repair)]
cwd=root/'native-vla2'
print('START v2.capture.0 receipt repair',flush=True);start=time.time()
with (repair/'results/v2.capture.0.log').open('x') as stream:
    try:code=subprocess.run(command,env=env,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,timeout=1200).returncode
    except subprocess.TimeoutExpired:code=124
status=[dict(label='v2.capture.0',command=command,cwd=str(cwd),exit_code=code,elapsed_s=time.time()-start)]
(repair/'status.json').write_text(json.dumps(status,indent=2))
(repair/'complete.json').write_text(json.dumps(dict(attempts=1,failures=int(code!=0))))
print('DONE',code,flush=True)
