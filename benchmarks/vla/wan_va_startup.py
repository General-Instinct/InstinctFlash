"""Native VA startup through eight predictions and seven deferred KV commits."""
import importlib.metadata
import os
from pathlib import Path
import uuid

from .util import ConfigurationError, sha256_file, sha256_json, write_json_atomic


def probe(model, revision, mode, output):
    from .adapters import load_adapters
    supported = {(c['id'], c['revision']) for a in load_adapters() if a['backbone'] == 'wan_va'
                 for c in a['checkpoints']}
    if (model, revision) not in supported:
        raise ConfigurationError('VA startup requires a pinned native simulator checkpoint')
    output = Path(output)
    if output.exists():
        raise ConfigurationError('refusing to overwrite startup evidence')
    config_name = 'libero' if model.endswith('libero-long') else 'robotwin'
    os.environ['IFL_CFG'] = config_name
    from .instinctflash_driver import resolve_snapshot, seed_everything
    snapshot = resolve_snapshot(model, revision)
    os.environ['LINGBOT_CKPT'] = str(snapshot)
    import numpy as np
    import torch
    import instinctflash
    from instinctflash.runtime.lingbot_install import import_lingbot_server, install_deterministic_seed
    from instinctflash.adapters.lingbot_va import _ControlLoop
    from .wan_va_benchmark_server import execution_identity, snapshot_identity
    from .execution_evidence import execution_profile
    from .plan import pipeline_digest
    S = import_lingbot_server()
    runtime = None
    if mode == 'runtime_default':
        runtime = instinctflash.Runtime.from_pretrained(model, revision=revision,
            placement='in_process', precision='native', tier_ceiling='bitexact', seed=0)
        runtime.reset(prompt='benchmark startup shape probe')
        loop = runtime._backend._impl
        server = loop._server
        predict = runtime.predict
    elif mode == 'stock':
        from easydict import EasyDict
        cfg = EasyDict(dict(S.VA_CONFIGS[config_name]))
        cfg.wan22_pretrained_model_name_or_path = str(snapshot)
        cfg.save_root = str(output.parent / (output.stem + '-debug'))
        Path(cfg.save_root).mkdir()
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29881')
        os.environ.setdefault('RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1')
        S.init_distributed(1, 0, 0)
        cfg.rank = cfg.local_rank = 0; cfg.world_size = 1
        install_deterministic_seed(S, 0)
        server = S.VA_Server(cfg)
        loop = _ControlLoop(server, tuple(cfg.obs_cam_keys), frame_chunk_size=cfg.frame_chunk_size)
        loop.reset(prompt='benchmark startup shape probe')
        def predict(obs):
            result = loop.predict(obs)
            loop.commit(obs, result['action'])
            return result
    else:
        raise ConfigurationError('unsupported VA startup mode')
    cfg = server.job_config
    expected = (4, 4, 20, 50) if config_name == 'libero' else (2, 16, 25, 50)
    actual = (cfg.frame_chunk_size, cfg.action_per_frame, cfg.num_inference_steps, cfg.action_num_inference_steps)
    if actual != expected:
        raise ConfigurationError(f'native VA geometry/schedule changed: {actual} != {expected}')
    messages = []
    original = server.infer
    def observed(obs):
        result = original(obs)
        messages.append({'commit': bool(obs.get('compute_kv_cache')),
                         'frames': len(obs['obs']) if isinstance(obs.get('obs'), list) else 1})
        return result
    server.infer = observed
    rng = np.random.default_rng(0)
    actions = []
    seed_everything(0)
    try:
        for i in range(8):
            n = 1 if i == 0 else (cfg.frame_chunk_size - (1 if i == 1 else 0)) * 4
            frames = [{k: rng.integers(0, 256, (cfg.height, cfg.width, 3), dtype=np.uint8)
                       for k in cfg.obs_cam_keys} for _ in range(n)]
            value = np.asarray(predict({'obs': frames, 'prompt': 'benchmark startup shape probe',
                                        'save_visualization': False})['action'])
            expected_shape = (7 if config_name == 'libero' else 16, cfg.frame_chunk_size, cfg.action_per_frame)
            if value.shape != expected_shape or not np.isfinite(value).all():
                raise ConfigurationError(f'invalid native VA action geometry/values: {value.shape}, expected {expected_shape}')
            actions.append(sha256_json(value.tolist()))
        commits = [m for m in messages if m['commit']]
        expected_frames = [(cfg.frame_chunk_size - 1) * 4] + [cfg.frame_chunk_size * 4] * 6
        if [m['frames'] for m in commits] != expected_frames or len(messages) != 15:
            raise ConfigurationError('VA startup did not execute its native deferred-commit protocol')
        core = Path(instinctflash.__file__).parent
        source = Path(S.__file__).resolve().parent
        def sources(folder):
            return sha256_json({str(f.relative_to(folder)): sha256_file(f)
                                for f in sorted(folder.rglob('*.py')) if '.venv' not in f.parts})
        transforms = [] if runtime is None else [
            {'name': p.name, 'tier': p.tier.name, 'params': p.params} for p in runtime._plan.applied]
        result = {'schema_version': 1, 'synthetic': False, **snapshot_identity(snapshot),
            'upstream_sha256': sources(source), 'runtime_source_sha256': sources(core),
            'adapter_sha256': None if runtime is None else sha256_file(core / 'adapters/lingbot_va.py'),
            'pipeline_sha256': pipeline_digest(),
            'hardware': {'gpu_name': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
                         'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
            'packages': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'transformers')},
            'numeric_environment': {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
            'execution': {'mode': mode, 'precision': 'native', 'backend': 'stock' if runtime is None else 'in_process',
                'declared': execution_identity(cfg, []), 'transforms': transforms, 'graph_stats': {},
                'action_shape': [cfg.frame_chunk_size * cfg.action_per_frame, expected_shape[0]],
                'tier_ceiling': None if runtime is None else 'bitexact', 'capture_required': False},
            'startup': {'protocol': 'seeded-eight-predict-seven-commit-v1', 'seed': 0, 'calls': 8,
                'commit_calls': 7, 'commit_frames': expected_frames, 'attempt_id': uuid.uuid4().hex,
                'status': 'ready', 'fault_injection': {}, 'action_sha256': actions,
                'scope': 'Synthetic frames through real native prediction/state updates. Final prediction remains pending commit. No saturation, latency or closed-loop certificate.'}}
        execution_profile(result)
        write_json_atomic(output, result)
        return result
    finally:
        if runtime is not None:
            runtime.close()
        else:
            loop.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
