"""Reproduce the offline cuDNN screen on the existing Thor Cosmos environment."""
import argparse,fcntl,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--mode',choices=('screen','pair'),default='pair');p.add_argument('--family',choices=('edge','nano'),action='append');a=p.parse_args()
root=Path(__file__).resolve().parents[3]
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith('IFL_COSMOS3_')}
env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0',IFL_COSMOS3_CONDITIONING_CACHE='1',PYTHONPATH=f'{root}:{root}/examples/cosmos3_policy:'+env.get('PYTHONPATH',''))
env.setdefault('TRITON_PTXAS_PATH','/usr/local/cuda/bin/ptxas')
env.setdefault('TRITON_PTXAS_BLACKWELL_PATH','/usr/local/cuda/bin/ptxas')
with open('/tmp/thor_gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for family in a.family or ['edge','nano']:
        arms=['screen'] if a.mode=='screen' else ['model','reference']
        for arm in arms:
            output=out/f'{family}-cudnn-{arm}.json';assert not output.exists() and not output.with_suffix('.npz').exists()
            if arm=='screen':
                cmd=[sys.executable,str(Path(__file__).with_name('screen.py')),family,str(output),'--fixture',str(root/'eval/native_total_2026-09-10/fixtures/va_eval_obs.npz')]
            else:
                script=Path(__file__).with_name('benchmark.py') if arm=='model' else root/'eval/cosmos3_thor_2026-09-11/runtime-conditioning/benchmark.py'
                cmd=[sys.executable,str(script),family,'current',str(output),'--iterations','10']
            with output.with_suffix('.log').open('x') as log:
                subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
