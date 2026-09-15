"""Receipt rejection tests with synthetic metadata, not model-quality evidence."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('nano_contract_audit',Path(__file__).with_name('audit_contract.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def report():
    return dict(ok=True,trained_student=True,task_quality_certified=False,category='SCREEN',
        student_provenance=dict(training_seed=12031,student_updates=64,state_key='student',arm_id='nano_sde1_cfg1_seed12031'),
        backend_stats=dict(precision='native',action_chunk_size=32,action_steps=1,guidance=1,action_padding='zero',
            native_optimizations=dict(elided_lm_head_bytes=1244659712),sampling=dict(sigmas=[1.,0.]),
            sampler_calls=36,velocity_evaluations=36,padding_projection_calls=36,
            padding_projected_velocity_branches=36,padding_hooks_restored=True),
        prompt_contract=dict(format_prompt_as_json=False,history_length=1,action_chunk_size=32,max_action_dim=64,
            metadata_augmentors=['viewpoint_augmentor','duration_fps_augmentor','resolution_info_augmentor']),
        first_request_native_branch_clocks=[1000.])

def test_declared_nano_contract():audit.require_contract(report(),12031)

@pytest.mark.parametrize('section,key,value',[
    ('backend_stats','native_optimizations',dict(elided_lm_head_bytes=0)),
    ('backend_stats','action_steps',2),('backend_stats','precision','fp8'),
    ('backend_stats','velocity_evaluations',72),('backend_stats','padding_hooks_restored',False),
    ('prompt_contract','format_prompt_as_json',True),('prompt_contract','metadata_augmentors',[]),
    ('prompt_contract','history_length',0),('student_provenance','student_updates',2),
    ('student_provenance','state_key','fake')])
def test_changed_contract_rejected(section,key,value):
    r=report();r[section][key]=value
    with pytest.raises(ValueError):audit.require_contract(r,12031)

def test_incomplete_clock_rejected():
    r=report();r['first_request_native_branch_clocks']=[500.]
    with pytest.raises(ValueError):audit.require_contract(r,12031)
