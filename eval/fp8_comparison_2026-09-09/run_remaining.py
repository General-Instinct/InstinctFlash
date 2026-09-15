"""Complete the declared V2 set and unstarted vendor controls after cwd repair."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(sys.argv[1]).resolve()
repair=root/'cwd-repair'
repair.mkdir()
(repair/'results').mkdir()
(repair/'native-vla4').symlink_to(root/'native-vla4',target_is_directory=True)
(repair/'vla4-inputs.pt').symlink_to(root/'vla4-inputs.pt')
base=Path.home()/'ifl_eval/next_steps_20260906'
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
env['PYTHONPATH']=str(root/'source')+':'+str(Path.home()/'ifl_eval/native_qualification_20260906/final-execution')
env['QWEN3VL_PATH']=str(base/'qwen-config')
env['TRITON_CACHE_DIR']=str(base/'cache/triton')
env['TRITON_PTXAS_PATH']='/usr/local/cuda/bin/ptxas'
env['TRITON_PTXAS_BLACKWELL_PATH']='/usr/local/cuda/bin/ptxas'
sets=[]
arms=['stock','capture','engine8']
sets.append(('v2', [('v2',arm,repeat) for repeat in range(3) for arm in arms[repeat:]+arms[:repeat]]))
sets.append(('vendor',[(family,arm,0) for family in ('vla4','v2') for arm in ('stock_vendor','capture_vendor')]))
for group,entries in sets:
    status=[]
    for family,arm,repeat in entries:
        label=f'{family}.{arm}.{repeat}'
        exe=base/'run-env/bin/python' if family=='v2' else Path.home()/'venv_vla4b/bin/python'
        command=[str(exe),str(root/'source'/f'probe_{family}.py'),'--arm',arm,'--repeat',str(repeat),'--root',str(repair)]
        cwd=root/'native-vla2' if family=='v2' else Path.home()
        print('START',label,flush=True);start=time.time()
        with (repair/'results'/f'{label}.log').open('x') as stream:
            try:code=subprocess.run(command,env=env,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired:code=124
        status.append(dict(label=label,command=command,cwd=str(cwd),exit_code=code,elapsed_s=time.time()-start))
        (repair/f'{group}-status.json').write_text(json.dumps(status,indent=2))
        print('DONE',label,code,flush=True)
    (repair/f'{group}-complete.json').write_text(json.dumps(dict(attempts=len(status),failures=sum(x['exit_code']!=0 for x in status))))
