"""Measure public pi05 action replies, separating generation from queue reads.

Run each precision in its own process on an otherwise idle Thor. This includes
checkpoint processors and CPU action replies; transport and simulation are absent.
The varying-state phase reports graph/calibration overhead instead of hiding it
in warmup. These short checks are not a sustained tail-latency qualification.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from instinctflash import Runtime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--precision', choices=['native', 'fp8'], required=True)
parser.add_argument('--checkpoint', choices=['base', 'libero'], default='libero')
parser.add_argument('--chunks', type=int, default=8)
args = parser.parse_args()
if args.output.exists() or args.output.with_suffix('.npz').exists():
    raise RuntimeError('refusing to overwrite evidence')
if args.chunks < 2:
    raise ValueError('at least two chunks required')
base = args.checkpoint == 'base'
model = 'lerobot/pi05_base' if base else 'lerobot/pi05_libero_finetuned_v044'
revision = ('b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba' if base else
            '8e174154ef5f6c60a8da12ae99c303d8963138c1')
checkpoint = Path.home()/'.cache/huggingface/hub'/('models--'+model.replace('/', '--'))/'snapshots'/revision
cfg = json.loads((checkpoint/'config.json').read_text())
cameras = [k for k in cfg['input_features'] if k.startswith('observation.images.')]
state_dim = cfg['input_features']['observation.state']['shape'][0]
action_dim = cfg['output_features']['action']['shape'][0]
frames = np.load(Path.home()/'ifl/t3_assets/calib_obs.npz')

def observation(frame, state):
    return {cameras[0]: ((frames['image'][frame].transpose(2,0,1)+1)/2).astype(np.float32),
            cameras[1]: ((frames['wrist_image'][frame].transpose(2,0,1)+1)/2).astype(np.float32),
            'observation.state': np.full(state_dim, state, dtype=np.float32)}

# Record the actual switches after construction; do not silently rewrite a
# vendor's precision settings for the purpose of making the numbers match.
torch.manual_seed(2718)
np.random.seed(2718)
started = time.perf_counter()
rt = Runtime.from_pretrained(model, revision=revision, device='cuda:0', precision=args.precision)
rt.reset(prompt='pick up the cup')
setup_s = time.perf_counter()-started
latencies, actions, records = [], [], []
try:
    for phase, chunks in [('initial', 1), ('fixed_state', args.chunks), ('varying_state', args.chunks)]:
        for chunk in range(chunks):
            state = 0.0 if phase != 'varying_state' else (0.5 if chunk % 2 == 0 else -0.25)
            obs = observation(3 + chunk % 2, state)
            for index in range(cfg['n_action_steps']):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = rt.predict(obs)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter()-t0)*1000
                assert set(result) == {'action'} and result['action'].shape == (action_dim,)
                assert np.isfinite(result['action']).all()
                latencies.append(elapsed)
                actions.append(result['action'].copy())
                records.append(dict(phase=phase, chunk=chunk, action_index=index,
                                    generates_chunk=index == 0))
                if args.precision == 'fp8' and index == 0:
                    engine = rt._backend._loop._frontends[2]
                    records[-1].update(
                        prompt_tokens=engine.prefix_tokens-engine.sig_S,
                        profile_cache_hits=engine.prompt_cache_hits,
                        profile_cache_misses=engine.prompt_cache_misses)
    def stats(values):
        return dict(count=len(values), p50_ms=float(np.percentile(values,50)),
                    p99_ms=float(np.percentile(values,99)), max_ms=float(max(values)))
    groups = {}
    for phase in ('initial','fixed_state','varying_state'):
        for generates in (True, False):
            values = [v for v,r in zip(latencies,records)
                      if r['phase']==phase and r['generates_chunk']==generates]
            if values:
                groups[phase+('/generation' if generates else '/buffered')] = stats(values)
    np.savez(args.output.with_suffix('.npz'), actions=np.stack(actions), latency_ms=latencies)
    result = dict(model_id=model,revision=revision,precision=args.precision,
                  action_dim=action_dim,chunk_size=cfg['chunk_size'],n_action_steps=cfg['n_action_steps'],
                  setup_seconds=setup_s,groups=groups,calls=records,
                  matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  runtime_backend=type(rt._backend).__name__,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  artifacts_sha256=hashlib.sha256(args.output.with_suffix('.npz').read_bytes()).hexdigest(),
                  scope='Public API latency on recorded images and synthetic states; no transport, simulator or task-quality certificate')
    if args.precision == 'fp8':
        from instinctflash.runtime.pi05_engine import Pi05EngineLoop
        from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
        result['source_sha256'] = {cls.__name__:hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                                   for cls in (Pi05EngineLoop,Pi05TorchFrontendThor)}
    else:
        policy = rt._backend._impl._p
        driver = getattr(policy.model, '_ifl_static_denoiser', None)
        result['graph_stats'] = dict(
            installed=driver is not None,
            captured=driver is not None and driver._graph is not None,
            rejected=driver.rejected if driver is not None else False,
            replays=driver.replays if driver is not None else 0)
        result['parameter_dtypes'] = sorted({str(p.dtype) for p in policy.parameters()})
        result['capture_self_check'] = [r.params.get('self_check')
            for r in rt._backend._plan.results if r.name == 'graph_capture']
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='calls'},indent=2))
finally:
    rt.close()
