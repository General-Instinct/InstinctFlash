"""Native baseline/host-timestep paired benchmark; no cuDNN override."""
import hashlib,json,os,sys
from pathlib import Path
from cosmos3_iwm import conditioning_cache
from benchmarks.regression import cosmos
from host_timesteps import install
stats,restore=install(os.environ.get('IFL_HOST_TIMESTEPS')=='1')
output=Path(sys.argv[3])
try:
    code=cosmos.main()
finally:
    restore()
r=json.loads(output.read_text())
r['host_timesteps']=stats
if stats['enabled']:
    r['runtime_declared_execution_policy']=r.pop('execution_policy',None)
    r['execution_policy']={'category':'SCREEN','precision':'native','experimental_override':'Host timestep control; paired byte validation required','task_quality_certified':False}
r['scheduler_experiment_sources']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('benchmark.py','host_timesteps.py')}
if r.get('ok'):
    expected='cpu' if stats['enabled'] else 'cuda'
    if stats['set_timesteps_calls']!=len(r['cases']) or set(stats['devices'])!={expected}:
        r.update(ok=False,error='Timestep experiment coverage failed');code=1
output.write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(code)
