"""Calibration must retain early-step maxima and reject incomplete coverage."""
import unittest
from types import SimpleNamespace
import torch
from instinctflash.runtime.vla2_engine import ExpertScaleAccumulator,Vla2ActionGenerator


class ScaleTests(unittest.TestCase):
    def test_layer_local_maxima_across_steps(self):
        expert = torch.ones(2,4)
        moe = torch.ones(2)
        acc = ExpertScaleAccumulator(expert,moe,steps=2)
        for step in range(2):
            for layer in range(2):
                # All scales are below one: untouched initial slots must not
                # accidentally floor the calibration to the initial value.
                expert[layer].fill_(0.4 if step == layer else 0.1)
                moe[layer] = 0.3 if step == layer else 0.05
                acc(layer,step)
        acc.commit()
        self.assertTrue(torch.equal(expert,torch.full_like(expert,0.4)))
        self.assertTrue(torch.equal(moe,torch.full_like(moe,0.3)))

    def test_incomplete_and_nonfinite_rejected(self):
        expert,moe = torch.ones(1,4),torch.ones(1)
        acc = ExpertScaleAccumulator(expert,moe,steps=2)
        acc(0,0)
        with self.assertRaises(RuntimeError):acc.commit()
        with self.assertRaises(ValueError):acc(0,0)
        expert[0,2] = float('nan')
        acc(0,1)
        with self.assertRaises(RuntimeError):acc.commit()


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
