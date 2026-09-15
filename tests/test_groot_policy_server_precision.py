"""The paired simulator endpoint must select and attest the requested arithmetic."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from benchmarks.vla.groot_policy_server import Arm, KEYS, startup_observation
from benchmarks.vla.util import ConfigurationError


@pytest.mark.parametrize('mode,precision,tier',[
    ('runtime_default','native','bitexact'),('runtime_fp8','fp8','numeric')])
def test_runtime_arms_share_decoding_and_explicit_precision(mode,precision,tier):
    split={key:np.full((16,1),i,dtype=np.float32) for i,key in enumerate(KEYS)}
    policy=Mock()
    policy.predict.return_value={'actions':split}
    with patch('instinctflash.Runtime.from_pretrained',return_value=policy) as load:
        arm=Arm('/checkpoint',mode)
    load.assert_called_once_with('/checkpoint',precision=precision,
                                 placement='in_process',tier_ceiling=tier)
    arm.new_episode('pick up cup')
    obs={'video.image':np.zeros((2,2,3),dtype=np.uint8)}
    out=arm.predict(obs)
    assert out.shape==(16,7)
    np.testing.assert_array_equal(out[0],np.arange(7))
    assert policy.predict.call_args.args[0]['prompt']=='pick up cup'
    assert 'prompt' not in obs
    arm.close()
    policy.close.assert_called_once()


@pytest.mark.parametrize('dtype,replays,reported',[
    (torch.bfloat16,1,'fp8'),(torch.float8_e4m3fn,0,'fp8'),
    (torch.float8_e4m3fn,1,'native')])
def test_fp8_requires_packed_weights_execution_and_matching_stats(dtype,replays,reported):
    arm=Arm.__new__(Arm);arm.precision='fp8'
    loop=SimpleNamespace(_frontend=SimpleNamespace(_vlsa_q_w=[SimpleNamespace(dtype=dtype)]*4),
                         _runner=SimpleNamespace(replays=replays),backend_stats={'precision':reported})
    arm.policy=SimpleNamespace(_backend=SimpleNamespace(_loop=loop))
    with pytest.raises(ConfigurationError,match='did not execute'):
        arm.precision_receipt()


def test_unknown_mode_does_not_silently_load_native():
    with patch('instinctflash.Runtime.from_pretrained') as load:
        with pytest.raises(ConfigurationError,match='unknown GR00T arm'):
            Arm('/checkpoint','typo_fp8')
    load.assert_not_called()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_declared_startup_observation_roundtrip(tmp_path,dtype):
    obs=startup_observation(None)
    obs['video.image'][:]=73
    obs['state.gripper'][:]=[.02,.03]
    for key in KEYS:obs['state.'+key]=obs['state.'+key].astype(dtype)
    path=tmp_path/'obs.npz';np.savez(path,**obs)
    loaded=startup_observation(path)
    for key in obs:
        np.testing.assert_array_equal(loaded[key],obs[key])
        assert loaded[key].dtype==obs[key].dtype


@pytest.mark.parametrize('corruption',['missing_camera','wrong_gripper_shape','nonfinite_state'])
def test_bad_calibration_input_refused(tmp_path,corruption):
    obs=startup_observation(None)
    if corruption=='missing_camera':del obs['video.wrist_image']
    elif corruption=='wrong_gripper_shape':obs['state.gripper']=np.zeros(1,np.float32)
    else:obs['state.x'][0]=np.nan
    path=tmp_path/'obs.npz';np.savez(path,**obs)
    with pytest.raises(ConfigurationError):startup_observation(path)
