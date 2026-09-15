"""Identity-bound Thor VA Runtime endpoint, with native or explicit FP8 execution.

Startup replays recorded windows; only subsequent simulator rollouts are closed
loop. A receipt is published only after startup passes and the socket is bound.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import os
from pathlib import Path

from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic
from .wan_va_runtime_policy import RuntimePolicy

PROTOCOL = 'wan-va-runtime-observed-history-v1'


def source_digest(root):
    root = Path(root)
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in {'.py', '.so', '.json', '.yaml', '.yml'}
             and not any(part in {'.venv', '__pycache__', '.git'} for part in p.relative_to(root).parts)}
    if not files:
        raise ConfigurationError(f'no source files found in {root}')
    return sha256_json(files)


def native_source_root(native):
    # Native optimizations may replace __init__ with an InstinctFlash wrapper:
    # prefer its real class module. FP8 deliberately uses an unregistered isolated
    # module, whose untouched constructor still identifies the upstream file.
    try:
        filename = inspect.getfile(type(native))
    except TypeError:
        filename = inspect.getfile(type(native).__init__)
    return Path(filename).resolve().parent


def verify_checkpoint(root, manifest, revision):
    if manifest.get('revision') != revision or not manifest.get('files'):
        raise ConfigurationError('checkpoint manifest revision/files mismatch')
    files = manifest['files']
    for folder in ('transformer', 'vae', 'text_encoder', 'tokenizer'):
        if not any(name.startswith(folder + '/') for name in files):
            raise ConfigurationError(f'checkpoint manifest lacks {folder}')
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ConfigurationError('invalid checkpoint manifest path')
        path = Path(root) / relative
        if (not path.is_file() or path.stat().st_size != expected['bytes']
                or sha256_file(path) != expected['sha256']):
            raise ConfigurationError(f'checkpoint bytes differ: {name}')
    # Refuse unaccounted model/config files that a loader might discover.
    actual = {str(p.relative_to(root)) for folder in ('transformer', 'vae', 'text_encoder', 'tokenizer')
              for p in (Path(root) / folder).rglob('*') if p.is_file()}
    if actual != set(files):
        raise ConfigurationError('checkpoint file inventory differs from manifest')
    return sha256_json(files)


def load_windows(path, config):
    import numpy as np
    first_count = (config.frame_chunk_size - 1) * 4
    count = 1 + first_count + config.frame_chunk_size * 4
    with np.load(path, allow_pickle=False) as data:
        frames = [{} for _ in range(count)]
        for key in config.obs_cam_keys:
            if key not in data:
                raise ConfigurationError(f'missing startup camera {key}')
            value = data[key]
            if value.dtype != np.uint8 or value.ndim != 4 or value.shape[0] != count or value.shape[-1] != 3 or min(value.shape) <= 0:
                raise ConfigurationError(f'invalid recorded startup history {key}')
            for index, frame in enumerate(frames):
                frame[key] = value[index].copy()
    return [frames[:1], frames[1:1+first_count], frames[1+first_count:]]


def validate_config(config, family):
    expected = (20, 50, 4, 4, 0.05) if family == 'libero' else (25, 50, 2, 16, 1.0)
    actual = (config.num_inference_steps, config.action_num_inference_steps,
              config.frame_chunk_size, config.action_per_frame, float(config.action_snr_shift))
    if actual != expected or (float(config.guidance_scale), float(config.action_guidance_scale), float(config.snr_shift)) != (5., 1., 5.):
        raise ConfigurationError('Runtime changed the native VA schedule/guidance')
    if config.video_exec_step != -1 or str(config.param_dtype) != 'torch.bfloat16':
        raise ConfigurationError('requires full native schedule and BF16 conditioning')


def startup(policy, windows, prompt, shape):
    import numpy as np
    outputs = []
    for episode in range(2):
        policy.infer({'reset': True, 'prompt': prompt, 'benchmark_seed': 9173,
                      'benchmark_identity_sha256': sha256_json(policy.identity)})
        for window in windows:
            result = policy.infer({'obs': window, 'prompt': prompt, 'save_visualization': False})
            action = np.asarray(result.get('action'))
            if action.shape != shape or not np.isfinite(action).all():
                raise ConfigurationError('invalid startup action shape/values')
            outputs.append(action.copy())
    if any(a.dtype != b.dtype or a.tobytes() != b.tobytes() for a, b in zip(outputs[:3], outputs[3:])):
        raise ConfigurationError('startup episode reset did not reproduce action bytes')
    policy.seed = None
    return np.stack(outputs)


async def serve_policy(policy, port, receipt):
    from instinctflash.serving.msgpack_numpy import Packer, unpackb
    from websockets.asyncio.server import serve
    busy = False

    async def handle(ws):
        nonlocal busy
        if busy:
            await ws.close(code=1013)
            return
        busy = True
        policy.seed = None
        packer = Packer()
        try:
            await ws.send(packer.pack({'benchmark_identity': policy.identity}))
            async for frame in ws:
                await ws.send(packer.pack(policy.infer(unpackb(frame))))
        except Exception as error:
            try:
                await ws.send(f'{type(error).__name__}: {error}')
            except Exception:
                pass
        finally:
            policy.seed = None
            busy = False

    async with serve(handle, '127.0.0.1', port, compression=None, max_size=None, ping_interval=None):
        write_json_atomic(receipt, policy.identity)
        print('ready', receipt, flush=True)
        await asyncio.Future()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='declared local Runtime package')
    p.add_argument('--checkpoint-manifest', type=Path, required=True, help='independently pinned component hashes')
    precision = p.add_mutually_exclusive_group()
    precision.add_argument('--precision', choices=('native', 'fp8'), default='native')
    precision.add_argument('--fp8', dest='precision', action='store_const', const='fp8')
    p.add_argument('--startup-observation', type=Path, required=True)
    p.add_argument('--startup-prompt', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--startup-only', action='store_true')
    p.add_argument('--port', type=int, default=19061)
    a = p.parse_args(argv)
    from .adapters import load_adapters
    supported = {(c['id'], c['revision']) for adapter in load_adapters() if adapter['backbone'] == 'wan_va'
                 for c in adapter['checkpoints']}
    if (a.model, a.revision) not in supported or not a.startup_prompt.strip():
        raise ConfigurationError('requires a pinned supported VA checkpoint and startup prompt')
    actions_path = a.receipt.with_suffix('.actions.npz')
    if a.receipt.exists() or actions_path.exists():
        raise ConfigurationError('refusing to overwrite startup evidence')
    import numpy as np
    import torch
    import instinctflash
    from .instinctflash_driver import seed_everything
    from .wan_va_benchmark_server import execution_identity
    from .plan import pipeline_digest
    family = 'libero' if a.model.endswith('libero-long') else 'robotwin'
    os.environ['IFL_CFG'] = family
    # LINGBOT_CKPT overrides declarations; do not allow an unrelated frozen stack.
    if os.environ.get('LINGBOT_CKPT'):
        raise ConfigurationError('unset LINGBOT_CKPT; this benchmark binds the declared package')
    if torch.cuda.get_device_capability(0) != (11, 0):
        raise ConfigurationError('this campaign requires Thor SM110')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = instinctflash.Runtime.from_pretrained(a.checkpoint, precision=a.precision,
        placement='in_process', seed=None, tier_ceiling='bitexact' if a.precision == 'native' else 'numeric')
    policy = None
    try:
        runtime.reset(prompt=a.startup_prompt)
        server = (runtime._backend._impl._server if a.precision == 'native'
                  else runtime._backend._loop._loop._server)
        native = getattr(server, 'native', server)
        cfg = native.job_config
        validate_config(cfg, family)
        weights = verify_checkpoint(Path(cfg.wan22_pretrained_model_name_or_path),
                                    load_json(a.checkpoint_manifest), a.revision)
        import flash_rt
        runtime_root = Path(instinctflash.__file__).parent
        kernel_root = Path(flash_rt.__file__).parent
        upstream_root = native_source_root(native)
        windows = load_windows(a.startup_observation, cfg)
        identity = {'schema_version': 1, 'protocol': PROTOCOL, 'model_id': a.model,
                    'model_revision': a.revision, 'checkpoint_sha256': weights,
                    'precision': a.precision, 'seed_mode': 'episode_plus_frame', 'synthetic': False,
                    'runtime_source_sha256': source_digest(runtime_root),
                    'engine_source_sha256': source_digest(kernel_root),
                    'upstream_source_sha256': source_digest(upstream_root),
                    'execution': execution_identity(cfg, [x.name for x in runtime._plan.applied]),
                    'pipeline_sha256': pipeline_digest(), 'runtime_explanation': runtime.explain(),
                    'checkpoint_manifest_sha256': sha256_file(a.checkpoint_manifest),
                    'package_declaration_sha256': sha256_file(a.checkpoint / 'instinctflash.json'),
                    'startup_observation_sha256': sha256_file(a.startup_observation),
                    'startup_prompt': a.startup_prompt,
                    'hardware': {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                                 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()},
                    'packages_sha256': sha256_json(sorted(f"{d.metadata.get('Name')}=={d.version}" for d in importlib.metadata.distributions())),
                    'numeric_environment': {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark}}
        policy = RuntimePolicy(runtime, server, identity, seed_everything)
        shape = (7, 4, 4) if family == 'libero' else (16, 2, 16)
        outputs = startup(policy, windows, a.startup_prompt, shape)
        if a.precision == 'fp8':
            packed = [w for group in server.frontend._blk.values() for w in group
                      if w.dtype == torch.float8_e4m3fn]
            if not packed or not server.calibrated or not server.frontend._graphs:
                raise ConfigurationError('FP8 packed weights/calibration/graphs not observed')
            identity['fp8'] = {'packed_e4m3_tensors': len(packed), 'declaration': server.frontend.declaration(),
                               'graphs': len(server.frontend._graphs)}
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(actions_path, actions=outputs)
        identity['startup'] = {'calls': 6, 'seed': 9173, 'repeat_byte_equal': True,
            'actions_sha256': sha256_file(actions_path), 'action_shape': list(shape),
            'scope': 'Recorded-history startup only; not closed-loop quality or real-time qualification'}
        if a.startup_only:
            write_json_atomic(a.receipt, identity)
        else:
            asyncio.run(serve_policy(policy, a.port, a.receipt))
    finally:
        if policy is not None:
            policy.close()
        runtime.close()


if __name__ == '__main__':
    main()
