import os,subprocess,json
from pathlib import Path
r=Path(__file__).parent;old=r.parent/'next_steps_20260906';e=r/'execution'
env=dict(os.environ,PYTHONPATH=f'{e}:{e}/examples/lingbot_vla_v2',LINGBOT_VLA_V2_ROOT=str(old/'native-vla2'),HF_HUB_OFFLINE='1',HF_HUB_CACHE=str(old/'hub'),QWEN3VL_PATH=str(old/'qwen-config'),XDG_CACHE_HOME=str(r/'cache'),TRITON_CACHE_DIR=str(old/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(r/'cache/inductor'),TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',USE_TF='0')
for arm in ['capture','stock','current','current','stock','capture']:
    index=sum(1 for x in r.glob(arm+'.*.json'))
    out=r/f'{arm}.{index}.json'
    cmd=[str(old/'run-env/bin/python'),'-u',str(r/'v2_probe.py'),'--arm',arm,'--output',str(out)]
    with out.with_suffix('.log').open('w') as f:
        p=subprocess.Popen(cmd,env=env,cwd=e,stdout=f,stderr=subprocess.STDOUT)
        (r/'active.json').write_text(json.dumps({'pid':p.pid,'arm':arm,'output':str(out)}))
        rc=p.wait()
    if rc:raise RuntimeError((arm,rc))
(r/'v2.complete.json').write_text(json.dumps({'complete':True}))
