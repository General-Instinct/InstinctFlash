"""Trajectory evidence must not confuse absent evidence with equal actions."""
import copy
import hashlib
import struct
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.vla.report import _closed_loop_action_evidence


class ClosedLoopActionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.jobs = []
        results = []
        for arm in ('stock', 'candidate'):
            self.jobs.append({'request': {
                'arm': {'id': arm}, 'model_id': 'model', 'suite_id': 'suite',
                'pair_id': 'pair', 'task': 'task', 'suite': {'kind': 'closed_loop',
                'protocol': {'bridge': 'wan-va-libero-paused-v1'}}}})
            results.append({'resolved_seed': 3, 'provenance': {'scene_sha256': 'a'*64},
                            'metrics': {'finite': True, 'executed_steps': 2,
                                        'action_digest': 'b'*64, 'success': False}})
        self.pairs = [(self.jobs[0], results[0], self.jobs[1], results[1])]

    def evidence(self):
        return _closed_loop_action_evidence(self.pairs, self.jobs, 'stock')[0]

    def test_equal_failed_episodes_are_equal_actions_not_quality_success(self):
        r = self.evidence()
        self.assertEqual(r['verdict'], 'PASS')
        self.assertEqual(r['matching_pairs'], 1)
        self.assertEqual(r['control_successes'], 0)
        self.assertFalse(r['quality_gate'])

    def test_signed_zero_difference_and_step_count_mismatch_fail(self):
        for side, x in ((1, 0.0), (3, -0.0)):
            self.pairs[0][side]['metrics']['action_digest'] = hashlib.sha256(struct.pack('d', x)).hexdigest()
        self.assertEqual(self.evidence()['verdict'], 'FAIL')
        self.pairs[0][3]['metrics']['action_digest'] = self.pairs[0][1]['metrics']['action_digest']
        self.pairs[0][3]['metrics']['executed_steps'] = 3
        self.assertEqual(self.evidence()['verdict'], 'FAIL')

    def test_missing_digest_empty_or_nonfinite_is_incomplete(self):
        original = copy.deepcopy(self.pairs)
        for key, value in [('action_digest', None), ('executed_steps', 0),
                           ('executed_steps', True), ('finite', False)]:
            self.pairs = copy.deepcopy(original)
            for side in (1, 3):
                self.pairs[0][side]['metrics'][key] = value
            self.assertEqual(self.evidence()['verdict'], 'INCOMPLETE')

    def test_absent_pair_cannot_pass(self):
        self.pairs = []
        r = self.evidence()
        self.assertEqual(r['verdict'], 'INCOMPLETE')
        self.assertEqual(r['expected_pairs'], 1)
        self.assertEqual(r['missing_pair_ids'], ['pair'])

    def test_scene_and_protocol_mismatch_are_not_equivalence(self):
        self.pairs[0][3]['provenance']['scene_sha256'] = 'c'*64
        self.assertEqual(self.evidence()['verdict'], 'INCOMPLETE')
        self.pairs[0][3]['provenance']['scene_sha256'] = 'a'*64
        self.jobs[1]['request']['suite']['protocol']['bridge'] = 'other'
        self.assertEqual(self.evidence()['verdict'], 'INCOMPLETE')

    def test_unknown_digest_contract_not_assumed_raw_bytes(self):
        for job in self.jobs:
            job['request']['suite']['protocol']['bridge'] = 'unknown'
        self.assertEqual(self.evidence()['verdict'], 'NOT_APPLICABLE')
        for job in self.jobs:
            job['request']['suite']['protocol']['bridge'] = 'wan-va-robotwin-paused-v1'
        r = self.evidence()
        self.assertEqual(r['verdict'], 'PASS')
        self.assertIn('float64', r['encoding'])


if __name__ == '__main__':
    unittest.main()
