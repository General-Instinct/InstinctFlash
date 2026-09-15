"""A graph must not manufacture an exact pass by overwriting eager output storage."""
import unittest
try:
    import torch
except ImportError:
    torch = None
from instinctflash.runtime.capture_self_check import run_capture_self_check,_verdict_line

@unittest.skipIf(torch is None, 'torch is required for capture tensor checks')
class ReferenceIntegrity(unittest.TestCase):
    def test_replay_overwriting_reference_is_not_an_exact_pass(self):
        shared=torch.tensor([1.,2.])
        def eager():shared.copy_(torch.tensor([1.,2.]));return shared
        def replay():shared.add_(1);return shared
        result=run_capture_self_check(family='test',cases=[('shared-buffer',eager,replay)])
        self.assertFalse(result['passed']);self.assertEqual(result['max_abs_delta'],1.)
    def test_equal_reused_buffer_still_passes(self):
        shared=torch.tensor([1.,2.])
        result=run_capture_self_check(family='test',cases=[('same',lambda:shared,lambda:shared)])
        self.assertTrue(result['passed']);self.assertTrue(result['bitexact'])
    def test_cross_domain_guard_is_not_presented_as_calibration(self):
        result=run_capture_self_check(family='test',cases=[('staged',lambda:torch.tensor([0.]),lambda:torch.tensor([.01]))],
            tolerance=.02,comparison_domain='velocity',tolerance_domain='action',tolerance_provenance='legacy')
        self.assertTrue(result['passed']);self.assertEqual(result['calibration_status'],'legacy_cross_domain_guard')
        self.assertIn('legacy cross-domain guard',_verdict_line(result))
        self.assertNotIn('recorded envelope',_verdict_line(result))
    def test_later_replay_failure_is_not_hidden_by_the_first_pass(self):
        values=iter([0.,0.,.08])
        result=run_capture_self_check(family='test',
            cases=[('staged',lambda:torch.tensor([0.]),lambda:torch.tensor([next(values)]))],
            tolerance=.05,repeats=3)
        self.assertFalse(result['passed']);self.assertEqual(result['comparisons'],3)
        self.assertEqual(len(result['cases'][0]['repeat_checks']),3)
    def test_invalid_repeat_count_is_refused(self):
        for count in (0,-1,True,1.5):
            with self.assertRaises(ValueError):run_capture_self_check(family='test',cases=[],repeats=count)
    def test_nonfinite_and_empty_inputs_do_not_pass(self):
        for value in (float('nan'),float('inf')):
            result=run_capture_self_check(family='test',cases=[('bad',lambda:torch.tensor([value]),lambda:torch.tensor([value]))])
            self.assertFalse(result['passed'])
        self.assertFalse(run_capture_self_check(family='test',cases=[])['passed'])

if __name__=='__main__':unittest.main()
