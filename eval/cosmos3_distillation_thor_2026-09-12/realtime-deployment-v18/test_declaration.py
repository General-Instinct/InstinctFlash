"""Nano formal export admission and actual Runtime descriptor parsing."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('prepare_v18',Path(__file__).with_name('prepare.py'))
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)

def gate(seed=12031):
    return dict(schema_version=18,status='success',student_updates=64,steps=1,
        times=[1.,0.],action_shape=[33,64],future_action_shape=[32,8],action_padding='zero',
        state_key='student',live_state_matches_snapshot=True,cold_reload_bitexact=True,
        native_serving_bitexact=True,actual_callbacks=3,actual_branches=3,guidance=1,
        training_seed=seed,arm_id=f'nano_sde1_cfg1_seed{seed}')

@pytest.mark.parametrize('seed',[12031,12032])
def test_actual_descriptor_and_adapter_keep_nano_contract(seed,tmp_path):
    import torch
    from safetensors.torch import save_file
    from instinctflash.descriptors.package import from_pretrained
    from realtime_adapter import Cosmos3RealtimeAdapter,realtime_contract
    declaration=prepare.execution_declaration(gate(seed),prepare.NANO_FAMILY)
    (tmp_path/'instinctflash.json').write_text(json.dumps(dict(instinctflash_schema=1,execution=declaration)))
    (tmp_path/'config.json').write_text('{}');save_file({'fixture':torch.zeros(1)},tmp_path/'model.safetensors')
    cp=from_pretrained(tmp_path,require_servable=False)
    contract,steps,guidance=realtime_contract(cp)
    assert contract['sigmas']==[1.,0.] and contract['num_train_timesteps']==1000.
    assert steps==1 and guidance==1 and cp.execution.nfe=={'prefix':1,'action':1}
    assert cp.execution.model_id==prepare.NANO_FAMILY
    assert cp.execution.extra['format_prompt_as_json'] is False
    assert cp.execution.extra['checkpoint_role']=='dmd_student'
    assert cp.execution.extra['training_seed']==seed
    assert declaration['servable'] is False
    adapter=Cosmos3RealtimeAdapter().spec_for_checkpoint(cp)
    assert next(phase for phase in adapter.phases if phase.name=='action').nfe==1
    declaration['nfe']['prefix']=2
    (tmp_path/'instinctflash.json').write_text(json.dumps(dict(instinctflash_schema=1,execution=declaration)))
    with pytest.raises(ValueError):realtime_contract(from_pretrained(tmp_path,require_servable=False))

@pytest.mark.parametrize('field,value',[
    ('schema_version',17),('student_updates',2),('steps',2),('steps',True),
    ('times',[1.,.5,0.]),('times',[True,False]),('actual_callbacks',6),
    ('actual_branches',True),('guidance',4),('guidance',True),
    ('cold_reload_bitexact',False),('native_serving_bitexact',False),
    ('state_key','fake'),('arm_id','edge_sde1_cfg1_seed12031'),('action_padding','native')])
def test_pilots_incomplete_exports_and_other_contracts_rejected(field,value):
    g=gate();g[field]=value
    with pytest.raises(ValueError):prepare.execution_declaration(g,prepare.NANO_FAMILY)

def test_wrong_family_selector_rejected():
    with pytest.raises(ValueError):prepare.execution_declaration(gate(),'nvidia/Cosmos3-Edge-Policy-DROID')
