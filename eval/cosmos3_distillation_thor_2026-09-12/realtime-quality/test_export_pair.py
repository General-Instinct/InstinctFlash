"""Pairing integrity failures must prevent compact-bank admission."""
import json

import numpy as np
import pytest

import export_pair as module


@pytest.fixture
def pair(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('plan.json', 'protocol.json'):
        (source / name).write_text('{}')
    directories = [tmp_path / name for name in ('native', 'cudnn')]
    for directory in directories:
        directory.mkdir()
        (directory / 'report.json').write_text('{}')
        for index in range(128):
            arrays = {key: np.zeros((32, 8), np.float32) for key in
                      ('action', 'model_action', 'measured_action', 'model_measured_action')}
            arrays.update({key: np.zeros((1, 16), np.float32)
                           for key in ('reference', 'preserve_mask', 'initial_noise')})
            np.savez(directory / f'{index:04d}.npz', **arrays)
            (directory / f'{index:04d}.json').write_text(json.dumps(dict(
                request=dict(index=index, seed=index), arrays_sha256='fixture',
                trace=dict(normalizer={}, effective_sampler_arguments={},
                           action_offset=0, action_shape=[32, 8]))))
    # Full capture validation is covered separately; exercise pairing/export admission here.
    monkeypatch.setattr(module, 'validate', lambda directory, source: dict(
        attention=directory.name, configuration_id='same',
        report_sha256=module.digest(directory / 'report.json')))
    return *directories, tmp_path / 'output', source


def test_compact_bank_preserves_order_and_refuses_overwrite(pair):
    result = module.export_pair(*pair)
    assert len(result['requests']) == 128 and not result['quality_certified']
    with np.load(pair[2] / 'native.npz') as bank:
        assert bank['action'].shape == (128, 32, 8)
    with pytest.raises(FileExistsError):
        module.export_pair(*pair)


@pytest.mark.parametrize('key', ['seed', 'normalizer', 'initial_noise'])
def test_unpaired_input_never_exports(pair, key):
    path = pair[1] / ('0042.npz' if key == 'initial_noise' else '0042.json')
    if key == 'initial_noise':
        with np.load(path) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays[key][0, 0] = 1
        np.savez(path, **arrays)
    else:
        meta = json.loads(path.read_text())
        if key == 'seed':
            meta['request']['seed'] += 1
        else:
            meta['trace']['normalizer'] = {'changed': True}
        path.write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        module.export_pair(*pair)
    assert not pair[2].exists()
