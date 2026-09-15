"""Fresh-process V2 admission and fixed-real-input repeatability; no gate relaxation."""
import argparse,inspect,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from benchmarks.vla.util import write_json_atomic,sha256_file,sha256_json


def diagnose(rows,arrays):
    import lingbot_vla_v2_iwm.static_capture as module
    from instinctflash.runtime.capture_self_check import compare_tensors
    original=module.run_capture_self_check
    def instrumented(**kw):
        def cases():
            for i,(label,eager,replay) in enumerate(kw['cases']):
                captured={}
                def a():
                    captured['a1']=eager().detach().clone()
                    return captured['a1']
                def b():
                    captured['b1']=replay().detach().clone()
                    captured['a2']=eager().detach().clone()
                    captured['b2']=replay().detach().clone()
                    driver=inspect.getclosurevars(eager).nonlocals['self']
                    captured['static_eager']=driver._forward_static().detach().clone()
                    rows.append({'case':i,'stage':label,'domain':'denoise_velocity',
                        'dtype':str(captured['a1'].dtype),
                        'aa':compare_tensors(captured['a1'],captured['a2']),
                        'ab':compare_tensors(captured['a1'],captured['b1']),
                        'bb':compare_tensors(captured['b1'],captured['b2']),
                        'a_static':compare_tensors(captured['a1'],captured['static_eager'])})
                    for name,value in captured.items():arrays[f'velocity_{i}_{name}']=value.float().cpu().numpy()
                    return captured['b1']
                yield label,a,b
        return original(**dict(kw,cases=cases()))
    module.run_capture_self_check=instrumented


def main(a):
    from benchmarks.vla.instinctflash_driver import RuntimeArm,STOCK_ARMS,seed_everything,make_observation
    from benchmarks.vla.policy_trace import read_trace,V2Noise
    from benchmarks.vla.plan import pipeline_digest
    import instinctflash,importlib.metadata
    rows=[];arrays={}
    if a.diagnose:diagnose(rows,arrays)
    if a.fault:os.environ['IFL_VLA2_SELFCHECK_FAULT']='1'
    req={'model':{'backbone':'lingbot_vla_v2','checkpoint':{
        'id':'robbyant/lingbot-vla-v2-6b-robotwin','revision':'0451855729ec904f970600e0aec8b84661423afe'}},
        'arm':{'operating_point':{'placement':'in_process','precision':'native','tier_ceiling':'numeric'}}}
    seed_everything(0);t0=time.perf_counter()
    arm=(RuntimeArm if a.mode=='runtime_default' else STOCK_ARMS['lingbot_vla_v2'])(req)
    seed_everything(0);arm.new_episode('benchmark startup shape probe')
    obs=make_observation(req,0,'benchmark startup shape probe')
    for _ in range(8):arm.predict(obs)
    torch.cuda.synchronize()
    stats={};verdict=None
    if a.mode=='runtime_default':
        impl=arm._runtime._backend._impl;stats=impl.graph_stats
        verdict=impl._driver.self_check if impl._driver else None
    result={'schema_version':1,'synthetic':False,'mode':a.mode,'diagnostic':a.diagnose,'fault':a.fault,
        'startup_ms':(time.perf_counter()-t0)*1000,'startup_calls':8,'graph_stats':stats,'self_check':verdict,
        'velocity_diagnostics':rows,'numeric_environment':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
        'hardware':{'gpu_name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
                    'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
        'packages':{k:importlib.metadata.version(k) for k in ('torch','numpy','transformers')},
        'pipeline_sha256':pipeline_digest(),'probe_sha256':sha256_file(Path(__file__)),
        'scope':'Instrumented trials are attribution diagnostics, not production startup-rate samples. Normal trials retain the unmodified gate. Synthetic startup probes; real recorded observations/noise in action repeats. No closed-loop quality certificate.'}
    core=Path(instinctflash.__file__).parent
    result['runtime_source_sha256']=sha256_json({str(p.relative_to(core)):sha256_file(p) for p in sorted(core.rglob('*.py'))})
    write_json_atomic(Path(str(a.output)+'.startup.json'),result)
    noise=V2Noise(arm);action_rows=[]
    for trace_i,trace in enumerate(a.trace):
        manifest,calls=read_trace(trace)
        for repeat in range(a.repeats):
            for call_i,(request,response) in enumerate(calls):
                if request.get('reset'):
                    seed_everything(request['benchmark_seed']);arm.new_episode(request['prompt']);continue
                if 'benchmark_noise' not in response:raise ValueError('actual recorded noise required')
                noise.pending=response['benchmark_noise'];noise.last=None
                action=np.asarray(arm.predict(request)).copy()
                if not np.array_equal(noise.last,response['benchmark_noise']):raise ValueError('noise differs')
                name=f'action_{trace_i}_{repeat}_{call_i}';arrays[name]=action
                action_rows.append({'trace':trace_i,'repeat':repeat,'call':call_i,'array':name,
                                    'noise_identical':True,'trace_sha256':sha256_file(Path(trace)/'trace.json')})
    arm.close();result['actions']=action_rows
    np.savez_compressed(str(a.output)+'.npz',**arrays)
    result['arrays_sha256']=sha256_file(Path(str(a.output)+'.npz'))
    write_json_atomic(a.output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['stock','runtime_default'],required=True)
    p.add_argument('--diagnose',action='store_true');p.add_argument('--fault',action='store_true')
    p.add_argument('--trace',type=Path,action='append',required=True);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or Path(str(a.output)+'.startup.json').exists():raise ValueError('refusing overwrite')
    try:main(a)
    except Exception as e:
        write_json_atomic(Path(str(a.output)+'.failure.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
