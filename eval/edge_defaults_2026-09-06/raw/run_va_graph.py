import os,subprocess,json,hashlib
from pathlib import Path
r=Path(__file__).parent;e=r/'execution'
assert (r/'va.complete.json').exists(), 'reference arms must finish first'
env=dict(os.environ,PYTHONPATH=f'{e}:/home/guanming/iwm_shims',LINGBOT_ROOT='/home/guanming/lingbot-va',LINGBOT_CKPT='/home/guanming/ckpt_lingbot/lingbot-va-posttrain-robotwin',IFL_ROOT=str(e),XDG_CACHE_HOME=str(r/'va-cache'),TRITON_CACHE_DIR=str(r/'va-cache/triton'))
cmd=['/home/guanming/venv_va/bin/python','-u','-m','torch.distributed.run','--nproc_per_node','1','--master_port','29965',str(r/'probe_va.py'),'--input',str(r/'va-input.npz'),'--output',str(r/'va_action_graph'),'--terminal-elision','--action-graphs']
with (r/'va_action_graph.log').open('w') as log:
    p=subprocess.Popen(cmd,env=env,cwd='/home/guanming/lingbot-va',stdout=log,stderr=subprocess.STDOUT)
    (r/'va_graph.launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd,'probe_sha256':hashlib.sha256((r/'probe_va.py').read_bytes()).hexdigest()},indent=2))
    rc=p.wait()
if rc:raise RuntimeError(rc)
(r/'va_graph.complete.json').write_text(json.dumps({'complete':True}))
