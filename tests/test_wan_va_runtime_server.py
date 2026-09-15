from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.vla.util import ConfigurationError, sha256_file
from benchmarks.vla.wan_va_runtime_server import load_windows, verify_checkpoint, startup, native_source_root
from benchmarks.vla.wan_va_runtime_libero_scene import validate_identity


def test_source_binding_handles_fp8_isolated_module(tmp_path):
    import importlib.util
    path = tmp_path / 'server.py'
    path.write_text('class Server:\n    def __init__(self):\n        pass\n')
    spec = importlib.util.spec_from_file_location('_unregistered_va_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert native_source_root(module.Server()) == tmp_path


def test_native_constructor_patch_does_not_change_upstream_identity(tmp_path, monkeypatch):
    import importlib.util
    import sys
    path = tmp_path / 'native_server.py'
    path.write_text('class Server:\n    def __init__(self):\n        pass\n')
    spec = importlib.util.spec_from_file_location('_registered_va_test', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.Server.__init__ = lambda self: None
    assert native_source_root(module.Server()) == tmp_path


def test_checkpoint_verifier_rejects_changed_or_unaccounted_component(tmp_path):
    files = {}
    for folder in ('transformer', 'vae', 'text_encoder', 'tokenizer'):
        path = tmp_path / folder / 'config.json'
        path.parent.mkdir()
        path.write_text('{}')
        files[f'{folder}/config.json'] = {'bytes': 2, 'sha256': sha256_file(path)}
    manifest = {'revision': 'pinned', 'files': files}
    verify_checkpoint(tmp_path, manifest, 'pinned')
    path.write_text('[]')
    with pytest.raises(ConfigurationError, match='checkpoint bytes differ'):
        verify_checkpoint(tmp_path, manifest, 'pinned')
    path.write_text('{}')
    extra = tmp_path / 'transformer' / 'unexpected.safetensors'
    extra.write_bytes(b'extra')
    with pytest.raises(ConfigurationError, match='inventory differs'):
        verify_checkpoint(tmp_path, manifest, 'pinned')
    with pytest.raises(ConfigurationError, match='revision'):
        verify_checkpoint(tmp_path, manifest, 'different')


@pytest.mark.parametrize('chunk,counts', [(4, [1, 12, 16]), (2, [1, 4, 8])])
def test_recorded_startup_preserves_temporal_pixels(tmp_path, chunk, counts):
    data = np.arange(sum(counts), dtype=np.uint8)[:, None, None, None]
    data = np.broadcast_to(data, (sum(counts), 3, 4, 3)).copy()
    path = tmp_path / 'obs.npz'
    np.savez(path, camera=data)
    cfg = SimpleNamespace(frame_chunk_size=chunk, obs_cam_keys=['camera'])
    windows = load_windows(path, cfg)
    assert [len(w) for w in windows] == counts
    np.testing.assert_array_equal(np.stack([f['camera'] for w in windows for f in w]), data)
    np.savez(path, camera=data[:-1])
    with pytest.raises(ConfigurationError, match='invalid recorded startup history'):
        load_windows(path, cfg)


def test_startup_refuses_reset_drift_instead_of_publishing_success():
    class Policy:
        identity = {}
        episode = 0

        def infer(self, obs):
            if obs.get('reset'):
                self.episode += 1
                return {}
            return {'action': np.full((7, 4, 4), self.episode, dtype=np.float32)}

    with pytest.raises(ConfigurationError, match='reset did not reproduce'):
        startup(Policy(), [[{}], [{}], [{}]], 'test', (7, 4, 4))


def admitted_identity():
    from benchmarks.vla.wan_va_runtime_server import PROTOCOL
    from benchmarks.vla.wan_va_libero_driver import MODEL, REVISION
    return dict(protocol=PROTOCOL, model_id=MODEL, model_revision=REVISION,
        synthetic=False, seed_mode='episode_plus_frame', precision='native',
        execution=dict(nfe={'video': 20, 'action': 50}, guidance={'video': 5., 'action': 1.},
            grid_shifts={'video': 5., 'action': .05}, dtype='torch.bfloat16',
            geometry=dict(height=128, width=128, frame_chunk_size=4, action_per_frame=4,
                env_type='none', obs_cam_keys=['observation.images.agentview_rgb', 'observation.images.eye_in_hand_rgb'])),
        startup=dict(calls=6, repeat_byte_equal=True, action_shape=[7, 4, 4]),
        **{key: 'a' * 64 for key in ('checkpoint_sha256', 'runtime_source_sha256',
            'engine_source_sha256', 'upstream_source_sha256', 'pipeline_sha256', 'packages_sha256')})


def test_runtime_admission_refuses_legacy_or_unverified_fp8():
    identity = admitted_identity()
    validate_identity(identity)
    with pytest.raises(ConfigurationError, match='protocol/checkpoint mismatch'):
        validate_identity({**identity, 'protocol': 'wan-va-libero-paused-v1'})
    with pytest.raises(ConfigurationError, match='actual FP8 weight'):
        validate_identity({**identity, 'precision': 'fp8'})
    with pytest.raises(ConfigurationError, match='source/weight'):
        validate_identity({**identity, 'runtime_source_sha256': None})


def test_runtime_admission_refuses_distillation_and_geometry_drift():
    identity = admitted_identity()
    identity['execution']['nfe'] = {'video': 2, 'action': 4}
    with pytest.raises(ConfigurationError, match='original LIBERO schedule'):
        validate_identity(identity)
    identity = admitted_identity()
    identity['execution']['geometry']['frame_chunk_size'] = 2
    with pytest.raises(ConfigurationError, match='geometry mismatch'):
        validate_identity(identity)
