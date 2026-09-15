import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.vla import wan_va_runtime_robotwin_scene as scene
from benchmarks.vla.util import ConfigurationError


@pytest.fixture
def identity():
    # Use the actual endpoint receipt schema, changing the checkpoint-specific
    # fields for this unit fixture. This fixture is not a Thor startup receipt.
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / 'eval/thor_precision_completion_2026-09-09/'
                       'va-runtime-libero-native-serve-v2.json').read_text())
    value['model_id'], value['model_revision'] = scene.MODEL, scene.REVISION
    value['execution'].update(nfe={'video': 25, 'action': 50},
        grid_shifts={'video': 5., 'action': 1.},
        geometry={'height': 256, 'width': 320, 'frame_chunk_size': 2, 'action_per_frame': 16,
                  'env_type': 'robotwin_tshape', 'obs_cam_keys': list(scene.driver.CAMERAS)})
    value['startup']['action_shape'] = [16, 2, 16]
    return value


def test_native_and_explicit_fp8_admission(identity):
    scene.validate_identity(identity)
    identity.update(precision='fp8', fp8={'packed_e4m3_tensors': 180})
    scene.validate_identity(identity)


@pytest.mark.parametrize('path,value', [
    (('protocol',), scene.driver.PROTOCOL),
    (('model_revision',), 'wrong'),
    (('execution', 'nfe'), {'video': 2, 'action': 4}),
    (('execution', 'geometry', 'frame_chunk_size'), 4),
    (('startup', 'repeat_byte_equal'), False),
    (('runtime_source_sha256',), 'z' * 64),
    (('precision',), 'automatic'),
])
def test_incompatible_endpoint_is_refused(identity, path, value):
    modified = copy.deepcopy(identity)
    target = modified
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigurationError):
        scene.validate_identity(modified)


@pytest.mark.parametrize('count', [None, 0, -1, True, '180'])
def test_fp8_requires_real_packed_weight_count(identity, count):
    identity.update(precision='fp8', fp8={'packed_e4m3_tensors': count})
    with pytest.raises(ConfigurationError, match='FP8 weight'):
        scene.validate_identity(identity)


def test_episode_records_normal_failure_and_closes_transport(identity, tmp_path, monkeypatch):
    from test_robotwin_driver import job, seal
    from benchmarks.vla.util import write_json_atomic
    request = job()['request']
    request['model']['checkpoint']['derived_from_revision'] = scene.REVISION
    write_json_atomic(tmp_path / 'job.json', seal(request))
    write_json_atomic(tmp_path / 'identity.json', identity)
    write_json_atomic(tmp_path / 'scenes.json', {})
    sources = {'robotwin_revision': request['dataset']['revision']}
    frozen = {'resolved_seed': request['requested_seed'], 'prompt': 'test'}
    monkeypatch.setattr(scene.driver, 'load_scene', lambda *a: (
        {'sources': sources, 'assets_sha256': 'assets'}, frozen))
    monkeypatch.setattr(scene.driver, 'source_identity', lambda *a: sources)
    monkeypatch.setattr(scene.driver, 'assets_identity', lambda *a: 'assets')
    monkeypatch.setattr(scene.driver, 'import_client', lambda *a: SimpleNamespace(
        add_init_pose=lambda action, origin: action))
    closed = []

    class Remote:
        timings = []

        def __init__(self, endpoint, actual_identity, *, timeout, record_dir):
            assert actual_identity == identity
            write_json_atomic(record_dir / 'trace.json', {'fixture': True})

        def close(self):
            closed.append(True)

    def rollout(client, actual_request, actual_scene, remote, work, *, bridge_factory):
        assert actual_request == request and actual_scene == frozen
        assert bridge_factory is scene.WanVaRuntimeWireBridge
        client.add_init_pose([0.] * 16, [0.] * 16)
        write_json_atomic(work / 'executed_actions.json', [[0.] * 16])
        return {'success': False, 'executed_steps': 1}

    monkeypatch.setattr(scene, 'RemotePolicy', Remote)
    monkeypatch.setattr(scene.driver, 'rollout', rollout)
    argv = ['--endpoint', 'ws://localhost:19061']
    for name in ('job', 'identity', 'scenes', 'output', 'robotwin', 'lingbot'):
        argv += ['--' + name, str(tmp_path / (name + '.json'))]
    scene.main(argv)
    result = json.loads((tmp_path / 'output.json').read_text())
    assert result['ok'] is True and result['metrics']['success'] is False
    assert result['protocol'] == scene.PROTOCOL and closed == [True]
    with pytest.raises(ConfigurationError, match='overwrite'):
        scene.main(argv)


def test_actual_origin_is_preserved_and_changes_are_refused(tmp_path):
    import numpy as np
    received = []

    def native(action, origin):
        received.append((action, origin))
        return action

    client = SimpleNamespace(add_init_pose=native)
    action, origin = np.arange(16.), np.arange(16.) / 10
    with scene.record_controller_origin(client, tmp_path / 'origin.json'):
        assert client.add_init_pose(action, origin) is action
        assert received[0][0] is action and received[0][1] is origin
        with pytest.raises(ConfigurationError, match='origin changed'):
            client.add_init_pose(action, origin + 1)
    assert client.add_init_pose is native and len(received) == 1
    saved = json.loads((tmp_path / 'origin.json').read_text())
    assert np.asarray(saved['pose'], dtype=saved['dtype']).tobytes() == origin.tobytes()
