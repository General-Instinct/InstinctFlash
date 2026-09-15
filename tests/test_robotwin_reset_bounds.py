import unittest
from benchmarks.vla.robotwin_driver import fresh_clutter_bounds

class ResetBoundsTests(unittest.TestCase):
    def test_repeated_resets_do_not_accumulate_table_offsets(self):
        class Env:
            def get_cluttered_table(self, count=10, xlim=[-.59,.59], ylim=[-.34,.34], zlim=[.741]):
                xlim[0]+=.3;xlim[1]+=.3
                return list(xlim)
        env=Env();original=env.get_cluttered_table
        with fresh_clutter_bounds(env):
            first=env.get_cluttered_table();second=env.get_cluttered_table()
        self.assertEqual(first,second)
        self.assertEqual(original.__defaults__[1],[-.59,.59])
        self.assertEqual(env.get_cluttered_table.__func__,original.__func__)
    def test_explicit_bounds_are_copied_and_geometry_preserved(self):
        class Env:
            def get_cluttered_table(self, xlim=[0,1], ylim=[0,1], zlim=[0]):
                xlim[0]+=.3
                return xlim,ylim,zlim
        env=Env();bounds=[2,3]
        with fresh_clutter_bounds(env):result=env.get_cluttered_table(xlim=bounds)
        self.assertEqual(bounds,[2,3]);self.assertEqual(result[0],[2.3,3])

if __name__=='__main__':unittest.main()
