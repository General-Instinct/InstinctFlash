"""Do not package pilot, EMA, truncated or unverified exports as final students."""
import pytest
from prepare import validate_gate


def gate(guidance=1, seed=12031):
    return dict(schema_version=16, status='success', student_updates=64, steps=1,
        times=[1., 0.], action_shape=[33,64], future_action_shape=[32,8],
        action_padding='zero', state_key='student', live_state_matches_snapshot=True,
        cold_reload_bitexact=True, native_serving_bitexact=True, actual_callbacks=3,
        actual_branches=3*(1 if guidance==1 else 2), guidance=guidance,
        training_seed=seed, arm_id=f'edge_sde1_cfg{guidance}_seed{seed}')


@pytest.mark.parametrize('guidance,seed', [(1,12031),(1,12032),(4,12031),(4,12032)])
def test_preserves_all_declared_arms(guidance, seed):
    validate_gate(gate(guidance, seed))


@pytest.mark.parametrize('key,value', [('student_updates',2), ('state_key','ema'),
    ('times',[1,.75,0]), ('cold_reload_bitexact',False), ('native_serving_bitexact',False),
    ('actual_branches',6), ('guidance',True), ('arm_id','nano_sde1_cfg1_seed12031'),
    ('status','running'), ('action_padding','native')])
def test_refuses_nonfinal_or_incompatible_gate(key, value):
    report=gate(); report[key]=value
    with pytest.raises(ValueError): validate_gate(report)
