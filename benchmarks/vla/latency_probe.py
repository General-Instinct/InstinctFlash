"""Measure observation-to-ready-action calls on the actual policy device with real inputs."""
from pathlib import Path
import argparse,importlib.metadata,os,platform,subprocess,time
from .util import ConfigurationError,sha256_json,sha256_file,write_json_atomic

def measure(trace,model,revision,mode,output,*,warmup=8,iterations=128,tier_ceiling=None):
    if type(warmup) is not int or warmup<1 or type(iterations) is not int or iterations<1:
        raise ConfigurationError('warmup and iterations must be positive')
    if tier_ceiling not in (None, "bitexact", "numeric"):
        raise ConfigurationError("native probe tier ceiling must be bitexact or numeric")
    if mode == "stock" and tier_ceiling is not None:
        raise ConfigurationError("a Runtime tier ceiling does not apply to stock")
    output=Path(output)
    if output.exists():raise ConfigurationError('refusing to overwrite timings')
    from .policy_trace import read_trace,V2Noise
    from .instinctflash_driver import STOCK_ARMS,RuntimeArm,seed_everything,resolve_snapshot
    import torch
    manifest,calls=read_trace(trace);identity=manifest['identity']
    if (identity['model_id'],identity['model_revision'])!=(model,revision):raise ConfigurationError('recorded checkpoint differs')
    # The current native probe contract covers this family; other families need their own arm bridge.
    if model!='robbyant/lingbot-vla-v2-6b-robotwin':raise ConfigurationError('latency probe currently implements native V2 only')
    snapshot=resolve_snapshot(model,revision)
    files={str(f.relative_to(snapshot)):sha256_file(f) for f in sorted(snapshot.rglob('*')) if f.is_file()}
    weight_digest=sha256_json(files)
    if weight_digest!=identity['checkpoint_sha256']:raise ConfigurationError('local checkpoint inventory differs from recorded source')
    source=Path(os.environ['LINGBOT_VLA_V2_ROOT']).resolve()
    source_files={str(f.relative_to(source)):sha256_file(f) for folder in ('deploy','lingbotvla','configs','assets/norm_stats')
                  for f in sorted((source/folder).rglob('*')) if f.is_file() and f.suffix in {'.py','.json','.yaml','.yml'}}
    source_digest=sha256_json(source_files)
    if source_digest!=identity['upstream_sha256']:raise ConfigurationError('native source differs from recorded source')
    req={'model':{'backbone':'lingbot_vla_v2','checkpoint':{'id':model,'revision':revision}},'arm':{'operating_point':{'placement':'in_process','precision':'native','tier_ceiling':tier_ceiling}}}
    started=time.perf_counter();arm=(STOCK_ARMS['lingbot_vla_v2'] if mode=='stock' else RuntimeArm)(req)
    seed_everything(calls[0][0]['benchmark_seed']);arm.new_episode(calls[0][0]['prompt'])
    # Runtime builds its concrete policy lazily on first prediction.
    first=next(request for request,response in calls if not request.get('reset'))
    arm.predict(first);torch.cuda.synchronize();startup=(time.perf_counter()-started)*1000
    noise=V2Noise(arm);inputs=[(request,response['benchmark_noise']) for request,response in calls if not request.get('reset')]
    samples=[];digests=[]
    try:
        for i in range(warmup+iterations):
            request,initial=inputs[i%len(inputs)];noise.pending=initial
            torch.cuda.synchronize();start=time.perf_counter()
            action=arm.predict(request)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            if i>=warmup:
                samples.append(elapsed);digests.append(sha256_json(action.tolist()))
        numeric={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark}
        explanation=arm._runtime.explain() if mode!='stock' else None
    finally:arm.close()
    # The noise hook is diagnostic overhead included in every sample, not subtracted.
    result={'schema_version':1,'synthetic':False,'mode':mode,'tier':'BASELINE' if mode=='stock' else 'NUMERIC',
        'model':model,'revision':revision,'checkpoint_sha256':weight_digest,'upstream_sha256':source_digest,
        'hardware':{'gpu_name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
                    'machine':platform.machine(),'hostname':platform.node()},
        'packages':{k:importlib.metadata.version(k) for k in ['torch','numpy','transformers']},
        'numeric_environment':numeric,'runtime_explanation':explanation,'precision':'native','tier_ceiling':tier_ceiling,
        'startup_ms':startup,'warmup':warmup,'iterations':iterations,'executed_actions':50,
        'samples_ms':samples,'action_sha256':digests,'trace_sha256':sha256_file(Path(trace)/'trace.json'),
        'input_cases':len(inputs),'scope':'CPU observation preprocessing through synchronized ready CPU actions; native NFE=10 and 50-action chunk. Explicit recorded noise. Includes diagnostic noise-copy overhead; excludes sensor capture, network and actuator transport.'}
    from .plan import pipeline_digest
    result['pipeline_sha256']=pipeline_digest();result['sha256']=sha256_json(result)
    write_json_atomic(output,result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--revision',required=True);p.add_argument('--mode',choices=['stock','runtime_default'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--warmup',type=int,default=8);p.add_argument('--iterations',type=int,default=128);p.add_argument('--tier-ceiling',choices=['bitexact','numeric']);a=p.parse_args()
    measure(a.trace,a.model,a.revision,a.mode,a.output,warmup=a.warmup,iterations=a.iterations,tier_ceiling=a.tier_ceiling)
if __name__=='__main__':main()
