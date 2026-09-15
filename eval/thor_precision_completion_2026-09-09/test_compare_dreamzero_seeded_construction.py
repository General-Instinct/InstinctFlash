"""Synthetic verifier controls only; not DreamZero execution evidence."""
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from compare_dreamzero_seeded_construction import compare, sha


class Controls(unittest.TestCase):
    def test_evidence_and_difference_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in (1, 2):
                path = root / str(i)
                path.mkdir()
                stem = 'dreamzero-native-seeded_construction'
                (path/'source.json').write_text(json.dumps({'example.py': 'synthetic'}))
                state = {'construction_seed':9173, 'construction_rng':{'cpu':'x', 'cuda':['y']},
                         'model_state':{'weight':{'shape':[1], 'dtype':'float32', 'sha256':'z'}}}
                (path/f'{stem}-state.json').write_text(json.dumps(state))
                np.savez(path/f'{stem}-actions.npz', actions=np.zeros((6,24,8), dtype=np.float32))
                report = dict(ok=True, precision='native', construction_seed=9173, run_id=str(i), pid=i,
                    frame_positions=[3,5,7,3,5,7], scheduler_steps=16, dit_step_mask=[True]*8+[False]*8,
                    load_receipt={'verified_dit_tensors':1317}, executed_fp8_projections={},
                    checkpoint='fixture', probe_sha256='fixture', input_sha256='fixture', boot_id='fixture',
                    torch_version='fixture', device_name='fixture', matmul_tf32=False, cudnn_tf32=False,
                    cudnn_benchmark=False, source_sha256=sha(path/'source.json'),
                    state_sha256=sha(path/f'{stem}-state.json'), actions_sha256=sha(path/f'{stem}-actions.npz'))
                (path/f'{stem}.json').write_text(json.dumps(report))
            left, right = root/'1', root/'2'
            result = compare(left, right, 'native')
            self.assertTrue(result['action_byte_equal'])
            self.assertEqual(result['persistent_state_differences'], [])
            with self.assertRaisesRegex(ValueError, 'Same directory'):
                compare(left, left, 'native')
            receipt = right/f'{stem}.json'
            original = json.loads(receipt.read_text())
            changed = dict(original, input_sha256='different')
            receipt.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Unmatched input'):
                compare(left, right, 'native')
            receipt.write_text(json.dumps(original))
            np.savez(right/f'{stem}-actions.npz', actions=np.ones((6,24,8), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'Corrupted actions'):
                compare(left, right, 'native')
            original['actions_sha256'] = sha(right/f'{stem}-actions.npz')
            receipt.write_text(json.dumps(original))
            result = compare(left, right, 'native')
            self.assertFalse(result['action_byte_equal'])
            self.assertEqual(result['action_max_abs'], 1.0)
            state_path = right/f'{stem}-state.json'
            state = json.loads(state_path.read_text())
            state['model_state']['weight']['sha256'] = 'different'
            state_path.write_text(json.dumps(state))
            original['state_sha256'] = sha(state_path)
            receipt.write_text(json.dumps(original))
            self.assertEqual(compare(left, right, 'native')['persistent_state_differences'], ['weight'])
            state['construction_rng']['cuda'] = ['changed']
            state_path.write_text(json.dumps(state))
            original['state_sha256'] = sha(state_path)
            receipt.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'construction RNG'):
                compare(left, right, 'native')


if __name__ == '__main__':
    unittest.main()
