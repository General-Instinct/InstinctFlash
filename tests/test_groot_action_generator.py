"""Preserve native masks/embodiment and RTC behavior across the FP8 handoff."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from instinctflash.runtime.groot_engine import GrootActionGenerator


class Batch(dict):
    __getattr__=dict.__getitem__


class GeneratorTests(unittest.TestCase):
    def make(self):
        config=SimpleNamespace(action_horizon=40,max_action_dim=132,state_history_length=1,
                               add_pos_embed=True,use_alternate_vl_dit=True)
        self.rtc=[];self.infer=[];self.vlsa=[]
        head=SimpleNamespace(config=config,num_inference_timesteps=4,num_timestep_buckets=1000,
            dtype=torch.bfloat16,state_encoder=lambda s,e:torch.zeros(1,1,1536),
            get_action_with_features=lambda *args:self.rtc.append(args) or 'native RTC result')
        fe=SimpleNamespace(_embodiment_id=24,
            infer=lambda state,initial_noise:self.infer.append((state,initial_noise)) or torch.zeros(1,40,132))
        runner=lambda *args:self.vlsa.append(args) or torch.zeros(1,3,2048)
        return GrootActionGenerator(head,fe,runner)

    def test_rtc_preserves_options_and_masks_without_engine_denoise(self):
        generator=self.make()
        backbone=Batch(backbone_features=torch.ones(1,3,2048),image_mask=torch.tensor([[False,True,False]]),
                       backbone_attention_mask=torch.tensor([[True,True,False]]))
        action=Batch(state=torch.ones(1,1,132),embodiment_id=torch.tensor([24]),action=torch.zeros(1,40,132))
        options={'rtc_overlap_steps':3,'rtc_frozen_steps':1,'rtc_ramp_rate':2,'action_horizon':40}
        module=SimpleNamespace(BatchFeature=lambda data:Batch(data))
        with patch.dict(sys.modules,{'transformers.feature_extraction_utils':module}):
            result=generator.get_action(backbone,action,options)
        self.assertEqual(result,'native RTC result');self.assertFalse(self.infer)
        self.assertIs(self.vlsa[0][2],backbone.backbone_attention_mask)
        self.assertIs(self.rtc[0][-1],options)
        self.assertIs(self.rtc[0][-2],action)
        self.assertEqual(self.rtc[0][0].dtype,torch.bfloat16)

    def test_wrong_embodiment_is_rejected_before_vlsa(self):
        generator=self.make();action=Batch(state=torch.ones(1,1,132),embodiment_id=torch.tensor([2]))
        with patch.dict(sys.modules,{'transformers.feature_extraction_utils':SimpleNamespace(BatchFeature=Batch)}):
            with self.assertRaises(ValueError):generator.get_action(Batch(),action)
        self.assertFalse(self.vlsa)


if __name__=='__main__':unittest.main()
