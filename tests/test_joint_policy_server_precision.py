"""Admission checks for the public joint-policy FP8 simulator endpoint."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import torch

from benchmarks.vla.joint_policy_server import (
    fp8_execution_receipt, load_startup_observation, main, validate_precision_options,
)
from benchmarks.vla.instinctflash_driver import LINGBOT_CAMERAS
from benchmarks.vla.util import ConfigurationError


def test_fp8_admission_sets_numeric_tier_and_refuses_stock_or_blank_prompt():
    args = NS(precision='fp8', mode='runtime_default', require_capture=False,
              capture_noise=False, tier_ceiling=None, startup_observation='obs.npz',
              startup_prompt='pick up cup')
    validate_precision_options(args)
    assert args.tier_ceiling == 'numeric'
    args.mode = 'stock'
    with pytest.raises(ConfigurationError): validate_precision_options(args)
    args.mode = 'runtime_default'
    args.startup_prompt = '  '
    with pytest.raises(ConfigurationError): validate_precision_options(args)


@pytest.mark.parametrize('flags', [
    ['--fp8'], ['--precision', 'fp8'],
    ['--fp8', '--tier-ceiling', 'bitexact'],
    ['--fp8', '--require-capture'], ['--fp8', '--capture-noise'],
])
def test_invalid_fp8_configuration_never_loads_checkpoint(tmp_path, flags):
    with patch('benchmarks.vla.instinctflash_driver.resolve_snapshot') as load:
        with pytest.raises(ConfigurationError):
            main(['--model', 'robbyant/lingbot-vla-4b-posttrain-robotwin',
                  '--revision', 'pinned', '--mode', 'runtime_default',
                  '--port', '29500', '--receipt', str(tmp_path/'receipt.json'), *flags])
        load.assert_not_called()


@pytest.mark.parametrize('flags,expected', [([], 'native'), (['--fp8'], 'fp8'),
                                          (['--precision', 'fp8'], 'fp8')])
def test_cli_precision_selection(tmp_path, flags, expected):
    # Stop immediately after parsing, before model or simulator imports execute.
    class Parsed(Exception):
        pass
    def inspect(args):
        assert args.precision == expected
        raise Parsed
    with patch('benchmarks.vla.joint_policy_server.validate_precision_options', inspect):
        with pytest.raises(Parsed):
            main(['--model', 'robbyant/lingbot-vla-4b-posttrain-robotwin',
                  '--revision', 'pinned', '--mode', 'runtime_default',
                  '--port', '29500', '--receipt', str(tmp_path/'receipt.json'), *flags])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_startup_preserves_native_observation(tmp_path, dtype):
    obs = {key: np.full((12, 16, 3), 73+i, np.uint8)
           for i, key in enumerate(LINGBOT_CAMERAS)}
    obs['observation.state'] = np.arange(14, dtype=dtype)/17
    path = tmp_path/'obs.npz'
    np.savez(path, **obs)
    loaded = load_startup_observation(path, 'pick up cup')
    for key, value in obs.items():
        assert loaded[key].dtype == value.dtype
        assert loaded[key].tobytes() == value.tobytes()
    assert loaded['prompt'] == loaded['task'] == 'pick up cup'


@pytest.mark.parametrize('corruption', ['missing_camera', 'float_camera', 'empty_camera',
                                      'state_shape', 'state_nan', 'extra_key'])
def test_bad_startup_observations_refused(tmp_path, corruption):
    obs = {key: np.ones((12, 16, 3), np.uint8) for key in LINGBOT_CAMERAS}
    obs['observation.state'] = np.ones(14, np.float32)
    if corruption == 'missing_camera': del obs[LINGBOT_CAMERAS[0]]
    elif corruption == 'float_camera': obs[LINGBOT_CAMERAS[0]] = np.ones((12,16,3), np.float32)
    elif corruption == 'empty_camera': obs[LINGBOT_CAMERAS[0]] = np.ones((0,16,3), np.uint8)
    elif corruption == 'state_shape': obs['observation.state'] = np.ones((1,14), np.float32)
    elif corruption == 'state_nan': obs['observation.state'][0] = np.nan
    else: obs['undeclared'] = np.ones(1)
    path = tmp_path/'obs.npz'
    np.savez(path, **obs)
    with pytest.raises(ConfigurationError): load_startup_observation(path, 'pick up cup')


@pytest.mark.parametrize('backbone', ['lingbot_vla', 'lingbot_vla_v2'])
@pytest.mark.parametrize('fault', [None, 'native', 'weight_dtype', 'empty_weights',
                                  'missing_group', 'vision_dtype', 'no_capture', 'no_replay'])
def test_fp8_receipt_requires_weights_native_vision_and_execution(backbone, fault):
    weights = [] if fault == 'empty_weights' else [NS(dtype=(
        torch.bfloat16 if fault == 'weight_dtype' else torch.float8_e4m3fn))]
    frontend = NS(_l_qkv_w=weights, _e_qkv_w=weights,
                  _moe={'gateup_fp8': weights, 'down_fp8': weights})
    if fault == 'missing_group':
        frontend._e_qkv_w = []
        frontend._moe['down_fp8'] = []
    vision = NS(model=NS(parameters=lambda: iter([NS(dtype=(
        torch.float16 if fault == 'vision_dtype' else torch.bfloat16))])))
    generator = NS(frontend=frontend, vision=vision,
                   engine=NS(frontend=frontend, vision=vision))
    loop = NS(_server=NS(vla=NS(model=generator)), graph_stats={
        'captured': fault != 'no_capture', 'replays': 0 if fault == 'no_replay' else 8})
    runtime = NS(precision='native' if fault == 'native' else 'fp8', _backend=NS(_loop=loop))
    if fault:
        with pytest.raises(ConfigurationError): fp8_execution_receipt(runtime, backbone)
    else:
        result = fp8_execution_receipt(runtime, backbone)
        assert result['verified_e4m3_weight_tensors'] == 2
        assert result['vision_dtype'] == 'bfloat16'
