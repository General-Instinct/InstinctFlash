"""Admission and clock regression for the real V17 export contract."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('prepare_v17', Path(__file__).with_name('prepare.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

def gate(seed=12031):
    return dict(schema_version=17, status='success', student_updates=64, steps=2,
        times=[1., .5, 0.], action_shape=[33,64], future_action_shape=[32,8],
        action_padding='zero', state_key='student', live_state_matches_snapshot=True,
        cold_reload_bitexact=True, native_serving_bitexact=True, actual_callbacks=6,
        actual_branches=6, guidance=1, training_seed=seed, arm_id=f'edge_sde2_cfg1_seed{seed}')

@pytest.mark.parametrize('seed', [12031,12032])
def test_deployed_clock_matches_native_export(seed):
    source = gate(seed)
    execution = prepare.execution_declaration(source, 'test-v17')
    sampling = execution['sampling']
    assert sampling['sigmas'] == source['times']
    assert sampling['num_train_timesteps'] == 1000.
    assert [s*sampling['num_train_timesteps'] for s in sampling['sigmas'][:-1]] == [1000.,500.]
    assert execution['nfe'] == {'prefix':1, 'action':2}
    assert execution['guidance']['action']['scale'] == 1
    assert execution['servable'] is False

@pytest.mark.parametrize('key,value', [
    ('steps',1), ('times',[1.,0.]), ('guidance',4), ('guidance',True),
    ('student_updates',2), ('schema_version',16), ('actual_callbacks',3),
    ('actual_branches',3), ('cold_reload_bitexact',False), ('state_key','fake'),
    ('arm_id','edge_sde1_cfg1_seed12031'), ('action_padding','native')])
def test_incompatible_export_rejected(key,value):
    source = copy.deepcopy(gate()); source[key] = value
    with pytest.raises(ValueError):
        prepare.execution_declaration(source, 'test-v17')

@pytest.mark.parametrize('seed', [12031,12032])
def test_real_runtime_adapter_contract(seed, tmp_path):
    import json
    from instinctflash.descriptors.package import from_pretrained
    from realtime_adapter import realtime_contract, Cosmos3RealtimeAdapter
    declaration = prepare.execution_declaration(gate(seed), 'test-v17')
    (tmp_path/'instinctflash.json').write_text(json.dumps(dict(instinctflash_schema=1,execution=declaration)))
    (tmp_path/'config.json').write_text('{}')
    import torch
    from safetensors.torch import save_file
    save_file({'fixture':torch.zeros(1)}, tmp_path/'model.safetensors')
    cp = from_pretrained(tmp_path, require_servable=False)
    contract, steps, guidance = realtime_contract(cp)
    assert contract['num_train_timesteps'] == 1000 and steps == 2 and guidance == 1
    spec = Cosmos3RealtimeAdapter().spec_for_checkpoint(cp)
    assert next(p for p in spec.phases if p.name == 'action').nfe == 2
    declaration['nfe']['prefix'] = 2
    (tmp_path/'instinctflash.json').write_text(json.dumps(dict(instinctflash_schema=1,execution=declaration)))
    with pytest.raises(ValueError, match='one prefix phase'):
        realtime_contract(from_pretrained(tmp_path, require_servable=False))
