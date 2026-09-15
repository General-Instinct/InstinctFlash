"""Observed native startup receipts for pi05 and GR00T LIBERO; no simulator scores."""
import argparse,importlib.metadata,inspect,os,sys,uuid
from pathlib import Path
from .util import ConfigurationError,load_json,sha256_file,sha256_json,write_json_atomic


def probe(model,revision,mode,output,*,checkpoint=None,subdir='libero_10',require_capture=False):
    output=Path(output)
    if output.exists():raise ConfigurationError('refusing to overwrite a startup receipt')
    if mode not in ('stock','runtime_default'):raise ConfigurationError('unknown startup arm')
    if mode=='stock' and require_capture:raise ConfigurationError('stock cannot require capture')
    if model not in ('lerobot/pi05_base','lerobot/pi05_libero_finetuned_v044','nvidia/GR00T-N1.7-LIBERO'):
        raise ConfigurationError('startup protocol not implemented for this checkpoint; do not substitute another family')
    from .instinctflash_driver import RuntimeArm,Pi05Stock,known_execution,make_observation,resolve_snapshot,seed_everything
    import numpy as np,torch,instinctflash
    snap=resolve_snapshot(model,revision)
    if model in ('lerobot/pi05_base','lerobot/pi05_libero_finetuned_v044'):
        backbone='pi05';declared=known_execution(model)
        req={'model':{'backbone':backbone,'checkpoint':{'id':model,'revision':revision}},
             'arm':{'operating_point':{'placement':'in_process','precision':'native','tier_ceiling':'bitexact'}}}
        arm=(RuntimeArm if mode=='runtime_default' else Pi05Stock)(req)
        obs=make_observation(req,0,'benchmark startup shape probe')
        import lerobot
        source=Path(lerobot.__file__).parent
        files={str(f.relative_to(snap)):sha256_file(f) for f in sorted(snap.rglob('*')) if f.is_file()}
    elif model=='nvidia/GR00T-N1.7-LIBERO':
        if checkpoint is None:raise ConfigurationError('GR00T requires a reviewed checkpoint view')
        from .groot_policy_server import Arm,KEYS,REVISION
        from .checkpoint_view import GROOT_VARIANTS
        if revision!=REVISION or subdir not in GROOT_VARIANTS:raise ConfigurationError('unsupported GR00T revision/variant')
        source=Path(os.environ['GR00T_ROOT']).resolve()/'gr00t';sys.path.insert(0,str(source.parent))
        checkpoint=Path(checkpoint);base=snap/subdir
        def inventory(path):return {f.name:sha256_file(f) for f in sorted(path.iterdir()) if f.is_file() and f.suffix in ('.json','.safetensors') and f.name!='instinctflash.json'}
        files=inventory(base)
        if inventory(checkpoint)!=files:raise ConfigurationError('GR00T view differs from pinned variant')
        declared=load_json(checkpoint/'instinctflash.json')['execution'];backbone='groot_n17'
        arm=Arm(checkpoint,mode)
        obs={f'video.{k}':np.zeros((256,256,3),np.uint8) for k in ('image','wrist_image')}
        obs.update({f'state.{k}':np.zeros(2 if k=='gripper' else 1,np.float32) for k in KEYS})
    else:raise ConfigurationError('startup protocol not implemented for this checkpoint; do not substitute another family')
    seed_everything(0);arm.new_episode('benchmark startup shape probe')
    try:
        for _ in range(8):
            if backbone=='pi05':arm.reset_chunk()
            action=np.asarray(arm.predict(obs))
        if not np.isfinite(action).all() or not action.size:raise ConfigurationError('invalid startup actions')
        shape=[1,int(action.size)] if backbone=='pi05' else list(action.shape)
        rt=None if mode=='stock' else (arm._runtime if backbone=='pi05' else arm.policy)
        stats={};transforms=[];adapter=None
        if rt is not None:
            impl=rt._backend._impl
            if backbone=='pi05':
                d=getattr(impl._p.model,'_ifl_static_denoiser',None)
                stats={'captured':bool(d and d._graph is not None),'replays':int(d.replays if d else 0)}
            else:stats=impl.backend_stats
            transforms=[{'name':p.name,'tier':p.tier.name,'params':p.params} for p in rt._plan.applied]
            plugin=Path(inspect.getfile(type(rt._adapter))).parent
            adapter=sha256_json({str(f.relative_to(plugin)):sha256_file(f) for f in sorted(plugin.rglob('*.py'))})
        rejected=bool(require_capture and not stats.get('captured'))
        core=Path(instinctflash.__file__).parent
        ex={'mode':mode,'precision':'native','backend':'stock' if rt is None else 'in_process',
            'declared':declared,'transforms':transforms,'graph_stats':stats,'action_shape':shape,
            'capture_required':require_capture,'tier_ceiling':None if rt is None else 'bitexact',
            'runtime_explanation':rt.explain() if rt is not None else None}
        if backbone=='groot_n17':ex['checkpoint_subdir']=subdir
        result={'schema_version':1,'synthetic':False,'model_id':model,'model_revision':revision,
            'checkpoint_sha256':sha256_json(files),'upstream_sha256':sha256_json({str(f.relative_to(source)):sha256_file(f) for f in sorted(source.rglob('*.py'))}),
            'runtime_source_sha256':sha256_json({str(f.relative_to(core)):sha256_file(f) for f in sorted(core.rglob('*.py'))}),
            'adapter_sha256':adapter,'execution':ex,
            'hardware':{'gpu_name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
                        'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
            'packages':{k:importlib.metadata.version(k) for k in ('torch','numpy','transformers')},
            'numeric_environment':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                'cudnn_benchmark':torch.backends.cudnn.benchmark,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
            'startup':{'protocol':'seeded-eight-call-admission-v1','attempt_id':uuid.uuid4().hex,'fault_injection':{k:os.environ[k] for k in ('IFL_VLA2_SELFCHECK_FAULT','IFL_VLA4B_SELFCHECK_FAULT','IFL_GROOT_SELFCHECK_FAULT','IFL_PI05_SELFCHECK_FAULT') if os.environ.get(k)=='1'},'calls':8,'seed':0,
                'status':'rejected' if rejected else 'ready','scope':'Synthetic startup observations. Pi05 resets action buffering each call; not latency or closed-loop evidence.'}}
        from .plan import pipeline_digest
        from .execution_evidence import execution_profile
        result['pipeline_sha256']=pipeline_digest();execution_profile(result)
        write_json_atomic(output,result)
        if rejected:raise ConfigurationError('required capture rejected; receipt retained')
        return result
    finally:arm.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--mode',choices=['stock','runtime_default'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--subdir',default='libero_10');p.add_argument('--require-capture',action='store_true')
    a=p.parse_args();probe(a.model,a.revision,a.mode,a.output,checkpoint=a.checkpoint,subdir=a.subdir,require_capture=a.require_capture)

if __name__=='__main__':main()
