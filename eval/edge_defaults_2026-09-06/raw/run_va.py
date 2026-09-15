import os,subprocess,json,hashlib
from pathlib import Path
r=Path(__file__).parent;e=r/'execution'
env=dict(os.environ,PYTHONPATH=f'{e}:/home/guanming/iwm_shims',LINGBOT_ROOT='/home/guanming/lingbot-va',LINGBOT_CKPT='/home/guanming/ckpt_lingbot/lingbot-va-posttrain-robotwin',IFL_ROOT=str(e),XDG_CACHE_HOME=str(r/'va-cache'),TRITON_CACHE_DIR=str(r/'va-cache/triton'))
# Validate the final production planner/adapter before starting the VA model.
old=r.parent/'next_steps_20260906';final=r/'final-execution'
v2env=dict(os.environ,PYTHONPATH=f'{final}:{final}/examples/lingbot_vla_v2',LINGBOT_VLA_V2_ROOT=str(old/'native-vla2'),HF_HUB_OFFLINE='1',HF_HUB_CACHE=str(old/'hub'),QWEN3VL_PATH=str(old/'qwen-config'),XDG_CACHE_HOME=str(r/'cache'),TRITON_CACHE_DIR=str(old/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(r/'cache/inductor'),TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',USE_TF='0')
for tag,ceiling in [('final_numeric','numeric'),('final_strict','bitexact')]:
    out=r/(tag+'.0.json')
    cmd=[str(old/'run-env/bin/python'),'-u',str(r/'probe_v2.py'),'--trace',str(old/'input'),'--output',str(out),'--tier-ceiling',ceiling]
    with (r/(tag+'.0.log')).open('w') as log:
        subprocess.run(cmd,env=v2env,cwd=final,stdout=log,stderr=subprocess.STDOUT,check=True)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
weights=Path(env['LINGBOT_CKPT']);native=Path(env['LINGBOT_ROOT'])
identity={'checkpoint_root':str(weights),'weight_files':{str(p.relative_to(weights)):digest(p) for p in sorted(weights.rglob('*')) if p.is_file()},
          'native_root':str(native),'native_python':{str(p.relative_to(native)):digest(p) for p in sorted((native/'wan_va').rglob('*.py'))},
          'execution_archive_sha256':digest(r/'source.tar'),'probe_sha256':digest(r/'probe_va.py')}
(r/'va.identity.json').write_text(json.dumps(identity,indent=2))
for arm in ['va_base','va_terminal']:
    out=r/arm
    cmd=['/home/guanming/venv_va/bin/python','-u','-m','torch.distributed.run','--nproc_per_node','1','--master_port','29963',str(r/'probe_va.py'),'--input',str(r/'va-input.npz'),'--output',str(out)]
    if arm=='va_terminal':cmd+=['--terminal-elision']
    with (r/(arm+'.log')).open('w') as f:
        p=subprocess.Popen(cmd,env=env,cwd='/home/guanming/lingbot-va',stdout=f,stderr=subprocess.STDOUT)
        (r/'va.active.json').write_text(json.dumps({'pid':p.pid,'arm':arm,'command':cmd}))
        rc=p.wait()
    if rc:raise RuntimeError((arm,rc))
(r/'va.complete.json').write_text(json.dumps({'complete':True}))
