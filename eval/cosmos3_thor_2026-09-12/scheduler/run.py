"""Run the narrow host-timestep A/candidate/B experiment on Thor."""
import argparse,fcntl,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--family',choices=('edge','nano'),default='edge');p.add_argument('--wait',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[3];out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith('IFL_COSMOS3_')}
env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0',IFL_COSMOS3_CONDITIONING_CACHE='1',PYTHONPATH=f'{root}:{root}/examples/cosmos3_policy:'+env.get('PYTHONPATH',''))
env.setdefault('TRITON_PTXAS_PATH','/usr/local/cuda/bin/ptxas');env.setdefault('TRITON_PTXAS_BLACKWELL_PATH','/usr/local/cuda/bin/ptxas')
with open('/tmp/thor_gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|(0 if a.wait else fcntl.LOCK_NB))
    for arm in ('baseline_a','host','baseline_b'):
        output=out/f'{a.family}-scheduler-{arm}.json';assert not output.exists() and not output.with_suffix('.npz').exists()
        env['IFL_HOST_TIMESTEPS']='1' if arm=='host' else '0'
        with output.with_suffix('.log').open('x') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('benchmark.py')),a.family,'current',str(output),'--iterations','10'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
