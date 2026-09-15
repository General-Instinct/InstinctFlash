import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/home/guanming/ifl_eval/numeric_total_20260910_fp8_completion');s=r/'source';out=r/'thor-0';out.mkdir(exist_ok=True)
lock=open('/tmp/thor_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
manifest=json.loads((r/'source.json').read_text())
def verify():
 for name,digest in manifest.items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest,name
verify();status={'status':'running','pid':os.getpid(),'started':time.time(),'family':'va','arm':'current_fewstep_fp8'};(out/'progress.json').write_text(json.dumps(status,indent=2))
py='/home/guanming/venv_va/bin/python';dest=out/'va-current_fewstep_fp8.json'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0',MASTER_PORT='29810',NO_ALBUMENTATIONS_UPDATE='1',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',PATH=str(Path(py).parent)+':/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/usr/sbin:/bin',PYTHONPATH=':'.join([str(s),str(s/'serving')]+[str(x) for x in (s/'examples').iterdir() if x.is_dir()]),DYNAMIC_CACHE_SCHEDULE='false',NUM_DIT_STEPS='8',ENABLE_TENSORRT='false',IFL_BENCH_TIER='numeric',LINGBOT_ROOT='/home/guanming/lingbot-va')
with dest.with_suffix('.log').open('x') as log:
 result=subprocess.run([py,str(s/'eval/numeric_total_2026-09-10/benchmark.py'),'va','fp8',str(dest),'--arm','current_fewstep_fp8','--iterations','20','--input-archive',str(s/'eval/native_total_2026-09-10/fixtures/va_eval_obs.npz')],cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
verify();report=json.loads(dest.read_text()) if dest.exists() else {};status.update(status='complete' if result.returncode==0 and report.get('ok') else 'failed',exit_code=result.returncode,p50_ms=report.get('p50_ms'),error=report.get('error'),ended=time.time());(out/'progress.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status))
