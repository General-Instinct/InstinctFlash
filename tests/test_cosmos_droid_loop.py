import threading
from types import SimpleNamespace

import numpy as np
import pytest

from instinctflash.runtime.cosmos_droid import CosmosDROIDLoop


class Service:
    def __init__(self):
        self.cfg = SimpleNamespace(seed=29, action_chunk_size=32, num_steps=4,
                                   guidance=3.0, conditioning_fps=15)
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(29)
        self.last = None

    def infer(self, obs):
        self.last = obs
        return {'action': self._rng.random((32, 8))}


def test_state_mapping_preserves_gripper_for_native_conversion():
    service = Service()
    loop = CosmosDROIDLoop(service)
    loop.reset(prompt='pick')
    state = np.arange(16, dtype=np.float32).reshape(2, 8)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    observation = dict(image=image, state=state)
    first = loop.predict(observation, executed_action=np.ones((32, 8)))['action']
    assert service.last['prompt'] == 'pick'
    np.testing.assert_array_equal(service.last['observation/joint_position'], state[:, :7])
    np.testing.assert_array_equal(service.last['observation/gripper_position'], state[:, 7:])
    assert service.last['observation/image'] is image
    assert set(observation) == {'image', 'state'}
    assert not np.array_equal(first, loop.predict(observation)['action'])
    loop.reset(prompt='pick')
    np.testing.assert_array_equal(first, loop.predict(observation)['action'])


def test_native_multiview_and_per_call_prompt_pass_through():
    service = Service()
    loop = CosmosDROIDLoop(service, {'recipe': 'test'})
    obs = {'prompt': 'move', 'observation/wrist_image_left': np.zeros((4, 4, 3)),
           'observation/exterior_image_1_left': np.zeros((4, 4, 3)),
           'observation/exterior_image_2_left': np.zeros((4, 4, 3)),
           'observation/joint_position': np.zeros(7),
           'observation/gripper_position': 0.3}
    loop.predict(obs)
    assert service.last.keys() == obs.keys()
    assert service.last['observation/gripper_position'] == 0.3
    assert loop.backend_stats()['precision'] == 'fp8'


def test_invalid_state_and_closed_loop_refused():
    loop = CosmosDROIDLoop(Service())
    loop.reset(prompt='pick')
    for state in (np.zeros(7), np.full(8, np.nan)):
        with pytest.raises(ValueError, match='state must'):
            loop.predict({'state': state})
    with pytest.raises(ValueError, match='not both'):
        loop.predict({'state': np.zeros(8), 'observation/gripper_position': 0.5})
    loop.close()
    loop.close()
    with pytest.raises(RuntimeError, match='closed'):
        loop.predict({})
    with pytest.raises(RuntimeError, match='closed'):
        loop.reset()


def test_numeric_stats_and_cleanup_follow_service_lifetime():
    service = Service()
    closed = []
    service._ifl_numeric_attention = SimpleNamespace(
        report=lambda: {'backend': 'cudnn', 'task_quality_certified': False},
        close=lambda: closed.append('attention'))
    loop = CosmosDROIDLoop(service)
    assert loop.backend_stats()['numeric_attention']['backend'] == 'cudnn'
    loop.close()
    loop.close()
    assert closed == ['attention']
