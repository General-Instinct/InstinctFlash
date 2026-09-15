import unittest
from benchmarks.vla.simulator_doctor import inspect_simulator

class SimulatorDoctorTest(unittest.TestCase):
    def test_h100_blocks_isaac_renderer(self):
        self.assertEqual(inspect_simulator('robolab',['NVIDIA H100 80GB HBM3']*8)['status'],'blocked_local_renderer')
    def test_h100_does_not_block_libero_cpu_rendering(self):
        self.assertEqual(inspect_simulator('libero',['NVIDIA H100'])['status'],'requires_runtime_validation')
    def test_unknown_or_mixed_hardware_is_not_certified(self):
        for names in [[],['Unknown'],['NVIDIA H100','NVIDIA RTX 6000 Ada']]:
            self.assertEqual(inspect_simulator('robolab',names)['status'],'requires_runtime_validation')
if __name__=='__main__':unittest.main()
