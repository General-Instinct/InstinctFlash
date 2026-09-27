"""The VLA2 generator keeps the native sample_actions contract (masks, grids, noise)."""
import unittest
from types import SimpleNamespace
import torch
from instinctflash.runtime.vla2_engine import Vla2ActionGenerator


class GeneratorTests(unittest.TestCase):
    def test_native_masks_and_noise_are_preserved(self):
        class Engine:
            def set_prompt(self, ids):self.ids=ids
            def infer_staged(self, images, state, noise):
                self.noise=noise
                return {'actions':torch.zeros(50,55)}
        engine=Engine()
        generator=Vla2ActionGenerator(engine,SimpleNamespace(num_steps=10,n_action_steps=50,max_action_dim=55))
        noise=torch.ones(1,50,55,dtype=torch.bfloat16)
        args=(torch.zeros(768,1536),torch.ones(1,3,dtype=torch.bool),
              torch.tensor([[7,8,0]]),torch.tensor([[True,True,False]]),
              torch.zeros(1,55,dtype=torch.bfloat16))
        result=generator.sample_actions(*args,noise=noise,image_grid_thw=torch.tensor([[1,16,16]]*3))
        self.assertEqual(result.shape,(1,50,55))
        self.assertEqual(engine.ids,[7,8])
        self.assertIs(engine.noise,noise)
        with self.assertRaises(ValueError):
            generator.sample_actions(*args,noise=torch.ones(1,10,55))
        with self.assertRaises(ValueError):
            generator.sample_actions(*args,image_grid_thw=torch.tensor([[1,32,32]]*3))


if __name__ == '__main__':unittest.main()
