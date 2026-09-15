import unittest
from benchmarks.vla.screening import paired_success_summary


def case(index, control=True, candidate=True):
    req = {'model_id':'m','suite_id':'s','suite':{'kind':'closed_loop'},'arm':{'id':'stock'},'pair_id':str(index),'task':'task'}
    job = {'request':req}
    return job, {'metrics':{'success':control}}, {}, {'metrics':{'success':candidate}}


class ScreeningSummaryTests(unittest.TestCase):
    def test_equal_successes_have_nonzero_uncertainty_and_no_verdict(self):
        pairs = [case(i) for i in range(20)]
        row = paired_success_summary(pairs, [p[0] for p in pairs], 'stock')[0]
        self.assertEqual(row['delta'], 0)
        self.assertLess(row['tango_central95'][0], 0)
        self.assertGreater(row['tango_central95'][1], 0)
        self.assertNotIn('verdict', row)
        self.assertFalse(row['quality_gate'])
    def test_discordance_and_missing_pairs_are_visible(self):
        pairs = [case(0,True,False),case(1,False,False)]
        row = paired_success_summary(pairs, [p[0] for p in pairs]+[case(2)[0]], 'stock')[0]
        self.assertEqual(row['delta'], -.5)
        self.assertEqual(row['control_only_successes'], 1)
        self.assertEqual(row['missing_pair_ids'], ['2'])
        self.assertFalse(row['complete'])
    def test_no_observed_pairs_is_not_zero_error(self):
        row = paired_success_summary([], [case(0)[0]], 'stock')[0]
        self.assertIsNone(row['delta'])
        self.assertIsNone(row['tango_central95'])
if __name__=='__main__':unittest.main()
