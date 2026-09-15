"""Pinned original/Runtime LingBot-VLA policy endpoint for joint-action simulation."""
from pathlib import Path
import argparse
import asyncio
import importlib.metadata
import os
import uuid

from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic

MODELS = {
    'robbyant/lingbot-vla-4b-posttrain-robotwin': ('lingbot_vla',25,'LINGBOT_VLA_ROOT'),
    'robbyant/lingbot-vla-v2-6b-robotwin': ('lingbot_vla_v2',50,'LINGBOT_VLA_V2_ROOT'),
}
PROTOCOL = 'lingbot-joint-robotwin-paused-v1'


def validate_precision_options(args):
    if args.precision == 'fp8':
        if args.mode != 'runtime_default':
            raise ConfigurationError('FP8 requires the Runtime arm')
        if args.require_capture or args.capture_noise or args.tier_ceiling == 'bitexact':
            raise ConfigurationError('FP8 cannot use native capture/noise requirements or a bitexact ceiling')
        if args.startup_observation is None or not (args.startup_prompt or '').strip():
            raise ConfigurationError('FP8 requires an explicit startup observation and prompt')
        if args.tier_ceiling is None:
            args.tier_ceiling = 'numeric'


def load_startup_observation(path, prompt):
    import numpy as np
    from .instinctflash_driver import LINGBOT_CAMERAS
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {*LINGBOT_CAMERAS, 'observation.state'}:
            raise ConfigurationError('startup observation must contain three native cameras and state')
        obs = {key:data[key].copy() for key in data.files}
    for key in LINGBOT_CAMERAS:
        value = obs[key]
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape) <= 0:
            raise ConfigurationError(f'invalid startup camera {key}')
    state = obs['observation.state']
    if state.shape != (14,) or state.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(state).all():
        raise ConfigurationError('invalid startup state')
    obs.update(task=prompt, prompt=prompt)
    return obs


def fp8_execution_receipt(runtime, backbone):
    import torch
    if runtime.precision != 'fp8':
        raise ConfigurationError('Runtime did not select FP8')
    loop = runtime._backend._loop
    stats = loop.graph_stats
    generator = loop._server.vla.model
    if backbone == 'lingbot_vla':
        frontend = generator.frontend
        groups = [list(frontend._l_qkv_w), list(frontend._e_qkv_w)]
        vision = generator.vision
    elif backbone == 'lingbot_vla_v2':
        frontend = generator.engine.frontend
        groups = [list(frontend._moe['gateup_fp8']), list(frontend._moe['down_fp8'])]
        vision = generator.engine.vision
    else:
        raise ConfigurationError('unsupported FP8 joint-policy backbone')
    weights = [weight for group in groups for weight in group]
    if any(not group for group in groups) or any(w.dtype != torch.float8_e4m3fn for w in weights):
        raise ConfigurationError('expected packed E4M3 weights are absent')
    if {p.dtype for p in vision.model.parameters()} != {torch.bfloat16}:
        raise ConfigurationError('joint-policy FP8 requires the corrected native BF16 vision path')
    if not stats.get('captured') or stats.get('replays',0) < 1:
        raise ConfigurationError('FP8 engine graph did not execute')
    return {'precision':'fp8','dtype':'mixed','vision_dtype':'bfloat16',
            'verified_e4m3_weight_tensors':len(weights),'graph_stats':stats}


class Policy:
    def __init__(self, arm, identity, seed_fn, noise=None, require_capture=False):
        self.arm,self.identity,self.seed_fn=arm,identity,seed_fn
        self.ready=False
        self.noise=noise
        self.require_capture=require_capture
    def infer(self, obs):
        import numpy as np
        if obs.get('reset'):
            seed=obs.get('benchmark_seed');digest=sha256_json(self.identity)
            if type(seed) is not int or seed < 0 or obs.get('benchmark_identity_sha256') != digest:
                raise ConfigurationError('invalid seeded reset')
            self.seed_fn(seed);self.arm.new_episode(obs['prompt']);self.ready=True
            return {'benchmark_seed':seed,'benchmark_identity_sha256':digest}
        if not self.ready:raise ConfigurationError('inference before seeded reset')
        obs=dict(obs)
        supplied=obs.pop('benchmark_noise',None)
        if supplied is not None and self.noise is None:raise ConfigurationError('explicit noise unsupported')
        if self.noise is not None:
            self.noise.pending=supplied;self.noise.last=None
        values=np.asarray(self.arm.predict(obs),dtype=np.float64)
        if self.require_capture and not self.arm._runtime._backend._impl.graph_stats['captured']:
            raise ConfigurationError('required capture is not executing')
        shape=tuple(self.identity['execution']['action_shape'])
        if values.size != shape[0]*shape[1] or not np.isfinite(values).all():
            raise ConfigurationError('invalid controller action chunk')
        result={'action':values.reshape(shape)}
        if self.noise is not None:
            if self.noise.last is None:raise ConfigurationError('native sampler did not consume noise')
            result['benchmark_noise']=self.noise.last
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--revision',required=True)
    p.add_argument('--mode',choices=['stock','runtime_default'],required=True)
    precision = p.add_mutually_exclusive_group()
    precision.add_argument('--precision',choices=['native','fp8'],default='native')
    precision.add_argument('--fp8',dest='precision',action='store_const',const='fp8')
    p.add_argument('--startup-observation',type=Path)
    p.add_argument('--startup-prompt')
    p.add_argument('--capture-noise',action='store_true')
    p.add_argument('--tier-ceiling',choices=['bitexact','numeric'])
    p.add_argument('--require-capture',action='store_true')
    p.add_argument('--startup-only',action='store_true',help='write observed admission receipt and exit; no policy listener')
    p.add_argument('--port',type=int,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args(argv)
    validate_precision_options(a)
    prompt=a.startup_prompt or 'benchmark startup shape probe'
    startup_observation=load_startup_observation(a.startup_observation,prompt) if a.startup_observation else None
    if a.receipt.exists():raise ConfigurationError('refusing to overwrite an endpoint receipt')
    if a.mode == "stock" and (a.tier_ceiling is not None or a.require_capture):
        raise ConfigurationError("stock does not accept Runtime capture policy")
    from .instinctflash_driver import STOCK_ARMS,RuntimeArm,resolve_snapshot,seed_everything,known_execution,make_observation
    from .plan import pipeline_digest
    backbone,chunk,root_env=MODELS[a.model]
    snapshot=resolve_snapshot(a.model,a.revision)
    files={str(f.relative_to(snapshot)):sha256_file(f) for f in sorted(snapshot.rglob('*')) if f.is_file()}
    if not any(k.endswith('.safetensors') for k in files):raise ConfigurationError('missing checkpoint weights')
    root=Path(os.environ[root_env]).resolve()
    source={str(f.relative_to(root)):sha256_file(f) for folder in ('deploy','lingbotvla','configs','assets/norm_stats')
            for f in sorted((root/folder).rglob('*')) if f.is_file() and f.suffix in {'.py','.json','.yaml','.yml'}}
    request={'model':{'backbone':backbone,'checkpoint':{'id':a.model,'revision':a.revision}},
             'arm':{'operating_point':{'name':a.mode,'placement':'in_process','precision':a.precision,'tier_ceiling':a.tier_ceiling}}}
    arm=(RuntimeArm if a.mode=='runtime_default' else STOCK_ARMS[backbone])(request)
    seed_everything(0)
    arm.new_episode(prompt)
    # Static capture deliberately warms multiple denoise calls before it captures.
    if startup_observation is None:
        startup_observation=make_observation(request,0,prompt)
    for _ in range(8):
        probe=arm.predict(startup_observation)
    import numpy as np
    if probe.size != chunk*14 or not np.isfinite(probe).all():
        raise ConfigurationError('startup probe action shape mismatch')
    import torch, inspect, instinctflash
    core_root=Path(instinctflash.__file__).parent
    runtime_sources={str(f.relative_to(core_root)):sha256_file(f) for f in sorted(core_root.rglob('*.py'))}
    graph_stats={}
    transforms=[]
    adapter_sha=None
    precision_receipt={'precision':a.precision}
    if a.mode=='runtime_default':
        rt=arm._runtime
        if a.precision == 'fp8':
            precision_receipt=fp8_execution_receipt(rt,backbone)
            graph_stats=precision_receipt['graph_stats']
        else:
            graph_stats=getattr(rt._backend._impl,'graph_stats',{})
        transforms=[{'name':r.name,'tier':r.tier.name,'params':r.params} for r in rt._plan.applied]
        plugin_root=Path(inspect.getfile(type(rt._adapter))).parent
        adapter_sha=sha256_json({str(f.relative_to(plugin_root)):sha256_file(f) for f in sorted(plugin_root.rglob('*.py'))})
    identity={'schema_version':1,'protocol':f'{backbone}-joint-robotwin-paused-v1','model_id':a.model,'model_revision':a.revision,
        'checkpoint_sha256':sha256_json(files),'upstream_sha256':sha256_json(source),
        'runtime_source_sha256':sha256_json(runtime_sources),'adapter_sha256':adapter_sha,
        'hardware':{'gpu_name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
                    'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
        'pipeline_sha256':pipeline_digest(),'seed_mode':'episode','synthetic':False,
        'execution':{'mode':a.mode,**precision_receipt,'tier_ceiling':a.tier_ceiling,
                     'startup_observation_sha256':sha256_file(a.startup_observation) if a.startup_observation else None,
                     'startup_prompt':prompt,
                     'backend':'stock' if a.mode=='stock' else ('engine' if a.precision=='fp8' else 'in_process'),'transforms':transforms,
                     'capture_required':a.require_capture,'graph_stats':graph_stats,'action_shape':[chunk,14],'declared':known_execution(a.model),
                     'runtime_explanation':arm._runtime.explain() if a.mode=='runtime_default' else None},
        'packages':{k:importlib.metadata.version(k) for k in ('torch','numpy','transformers')},
        'numeric_environment':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                               'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                               'cudnn_benchmark':torch.backends.cudnn.benchmark,
                               'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
        'startup_check':'eight seeded startup shape-probe calls; not quality evidence','gpu':torch.cuda.get_device_name(0)}
    rejected = a.require_capture and not graph_stats.get('captured')
    identity['startup']={'protocol':'seeded-eight-call-admission-v1','seed':0,'calls':8,
                         'attempt_id':uuid.uuid4().hex,'fault_injection':{k:os.environ[k] for k in ('IFL_VLA2_SELFCHECK_FAULT','IFL_VLA4B_SELFCHECK_FAULT','IFL_GROOT_SELFCHECK_FAULT','IFL_PI05_SELFCHECK_FAULT') if os.environ.get(k)=='1'},
                         'status':'rejected' if rejected else 'ready',
                         'scope':'Observed startup only; no task-quality or reliability-rate certificate.'}
    if a.startup_only or rejected:
        write_json_atomic(a.receipt,identity)
        arm.close()
        if rejected:raise ConfigurationError('startup did not execute required capture; rejected receipt retained')
        return
    noise=None
    if a.capture_noise:
        if backbone!='lingbot_vla_v2':raise ConfigurationError('noise capture currently supports V2 only')
        from .policy_trace import V2Noise
        noise=V2Noise(arm)
        identity['noise_protocol']='native-initial-tensor-v1'
    policy=Policy(arm,identity,seed_everything,noise,require_capture=a.require_capture)
    from instinctflash.serving.msgpack_numpy import Packer,unpackb
    from websockets.asyncio.server import serve
    async def run():
        busy=False
        async def handle(ws):
            nonlocal busy
            if busy:await ws.close(code=1013);return
            busy=True;packer=Packer();policy.ready=False
            try:
                await ws.send(packer.pack({'benchmark_identity':identity}))
                async for frame in ws:await ws.send(packer.pack(policy.infer(unpackb(frame))))
            except Exception as error:
                try:await ws.send(f'{type(error).__name__}: {error}')
                except Exception:pass
            finally:policy.ready=False;busy=False
        async with serve(handle,'127.0.0.1',a.port,compression=None,max_size=None,ping_interval=None):
            write_json_atomic(a.receipt,identity)
            print('ready',a.receipt,flush=True)
            await asyncio.Future()
    try:asyncio.run(run())
    finally:arm.close()

if __name__=='__main__':main()
