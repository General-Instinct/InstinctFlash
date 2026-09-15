import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import save_file
from benchmarks.vla.pi05_weights import verify_loaded_weights
from benchmarks.vla.util import ConfigurationError

class WeightsTests(unittest.TestCase):
    def test_missing_alias_only_allowed_for_actual_shared_storage(self):
        m=torch.nn.Module();m.a=torch.nn.Parameter(torch.ones(2));m.b=m.a
        with tempfile.TemporaryDirectory() as t:
            save_file({'a':torch.ones(2)},str(Path(t)/'model.safetensors'))
            r=verify_loaded_weights(m,t)
            self.assertEqual(r['verified_aliases'],{'b':'a'})
            m.b=torch.nn.Parameter(torch.ones(2))
            with self.assertRaisesRegex(ConfigurationError,'unloaded'):verify_loaded_weights(m,t)
    def test_swallowed_partial_load_detected_by_actual_values(self):
        m=torch.nn.Module();m.a=torch.nn.Parameter(torch.zeros(2))
        with tempfile.TemporaryDirectory() as t:
            save_file({'a':torch.ones(2)},str(Path(t)/'model.safetensors'))
            with self.assertRaisesRegex(ConfigurationError,'differs'):verify_loaded_weights(m,t)

if __name__=='__main__':unittest.main()
