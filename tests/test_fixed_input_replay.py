import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.vla.fixed_input import compare_actions, replay
from benchmarks.vla.policy_trace import PolicyTrace
from benchmarks.vla.util import ConfigurationError
from instinctflash.serving.msgpack_numpy import Packer


class FixedReplayTests(unittest.TestCase):
    identity = dict(model_id='native', model_revision='pinned', checkpoint_sha256='bytes', protocol='native-v1')

    def trace(self, root, *, noise=True, inference=True):
        trace = PolicyTrace(root, self.identity)
        packer = Packer()
        trace.append(packer.pack(dict(reset=True, prompt='task', benchmark_seed=7)), packer.pack({}))
        if inference:
            response = {'action': np.array([[1., -0.]])}
            if noise:
                response['benchmark_noise'] = np.array([[[0.25, -0.5]]], dtype=np.float32)
            trace.append(packer.pack({'image': np.ones((2, 2, 3), dtype=np.uint8)}), packer.pack(response))
        trace.close()

    def test_actual_noise_and_inputs_are_reused_for_each_repetition(self):
        class Endpoint:
            instances = []

            def __init__(self, *args, **kwargs):
                self.resets = []; self.requests = []; self.closed = False
                self.instances.append(self)

            def reset_episode(self, prompt, seed):
                self.resets.append((prompt, seed))

            def infer(self, request):
                self.requests.append(request)
                return {'action': np.array([[1., -0.]]), 'benchmark_noise': request['benchmark_noise']}

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory, patch('benchmarks.vla.fixed_input.RemotePolicy', Endpoint):
            root = Path(directory); self.trace(root / 'trace')
            result = replay([root / 'trace'], 'ws://test', self.identity, root / 'result.json', repeats=3)
            self.assertTrue(result['all_noise_identical'])
            self.assertTrue(result['all_repeats_identical'])
            self.assertTrue(result['all_recorded_actions_identical'])
            self.assertEqual(len(result['rows']), 3)
            endpoint = Endpoint.instances[0]
            self.assertEqual(endpoint.resets, [('task', 7)] * 3)
            self.assertTrue(endpoint.closed)
            for request in endpoint.requests:
                np.testing.assert_array_equal(request['benchmark_noise'], [[[0.25, -0.5]]])
                np.testing.assert_array_equal(request['image'], np.ones((2, 2, 3), dtype=np.uint8))
            with self.assertRaises(ConfigurationError):
                replay([root / 'trace'], 'ws://test', self.identity, root / 'result.json')

    def test_invalid_evidence_refused_before_connecting(self):
        with tempfile.TemporaryDirectory() as directory, patch('benchmarks.vla.fixed_input.RemotePolicy') as endpoint:
            root = Path(directory)
            self.trace(root / 'valid'); self.trace(root / 'no-noise', noise=False)
            self.trace(root / 'reset-only', inference=False)
            cases = [([], self.identity), ([root / 'valid'], {}),
                     ([root / 'valid'], dict(self.identity, checkpoint_sha256='different')),
                     ([root / 'no-noise'], self.identity), ([root / 'reset-only'], self.identity)]
            for traces, identity in cases:
                with self.assertRaises(ConfigurationError):
                    replay(traces, 'ws://test', identity, root / 'result.json')
            endpoint.assert_not_called()

    def test_action_bytes_preserve_signed_zero_and_reject_invalid_values(self):
        self.assertFalse(compare_actions([0.], [-0.])['identical'])
        for a, b in [([], []), ([1], [[1]]), ([float('nan')], [1])]:
            with self.assertRaises(ConfigurationError):
                compare_actions(a, b)


if __name__ == '__main__':
    unittest.main()
