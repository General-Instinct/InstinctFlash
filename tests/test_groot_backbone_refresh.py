"""Current observation KV must replace values, while graph addresses stay live."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'serving'))
from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor


class RefreshTests(unittest.TestCase):
    def test_weight_cast_cache_invalidates_on_mutation_and_replacement(self):
        fr=GrootN17TorchFrontendThor.__new__(GrootN17TorchFrontendThor)
        fr.weights=[torch.arange(8,dtype=torch.bfloat16)]
        cached=fr._fp32_weight('weights',0)
        self.assertIs(cached,fr._fp32_weight('weights',0))
        fr.weights[0].add_(1)
        updated=fr._fp32_weight('weights',0)
        self.assertIsNot(cached,updated)
        self.assertTrue(torch.equal(updated,fr.weights[0].float()))
        fr.weights[0]=torch.zeros(8,dtype=torch.bfloat16)
        replaced=fr._fp32_weight('weights',0)
        self.assertIsNot(updated,replaced)
        self.assertTrue(torch.equal(replaced,fr.weights[0].float()))

    def test_inference_tensor_without_version_is_not_cached(self):
        fr=GrootN17TorchFrontendThor.__new__(GrootN17TorchFrontendThor)
        with torch.inference_mode():
            fr.weight=torch.ones(8,dtype=torch.bfloat16)
        first=fr._fp32_weight('weight')
        with torch.inference_mode():
            fr.weight.mul_(3)
        self.assertTrue(torch.equal(first,torch.ones(8)))
        self.assertTrue(torch.equal(fr._fp32_weight('weight'),torch.full((8,),3.)))

    def test_dit_spec_preserves_native_bf16_values(self):
        from flash_rt.frontends.torch._groot_n17_thor_spec import _dit_block
        from flash_rt.executors.torch_weights import Quant
        block=_dit_block()
        self.assertEqual(block.num_layers,32)
        self.assertEqual(len(block.items),14)
        for item in block.items:
            self.assertIsNone(item.scale_into)
            self.assertFalse(any(isinstance(op,Quant) for op in item.transforms))
            source=torch.tensor([[1.03125,-2.0625],[0.5,3.125]],dtype=torch.bfloat16)
            if item.name.endswith('_b'):source=source[0]
            output=source.clone()
            for op in item.transforms:output=op.apply(output,None)
            expected=source.T.contiguous() if item.name.endswith('_w') else source
            self.assertTrue(torch.equal(output,expected))
            self.assertEqual(output.dtype,torch.bfloat16)

    def make_frontend(self):
        fr=GrootN17TorchFrontendThor.__new__(GrootN17TorchFrontendThor)
        fr.device='cpu'
        for name in ('_dit_k_w','_dit_v_w'):
            setattr(fr,name,[torch.ones(2048,2)*0.001 for _ in range(32)])
        for name in ('_dit_k_b','_dit_v_b'):
            setattr(fr,name,[torch.zeros(2) for _ in range(32)])
        return fr

    def test_refresh_keeps_addresses_and_geometry_change_invalidates(self):
        fr=self.make_frontend();x=torch.ones(1,4,2048);mask=torch.tensor([False,True,True,False])
        fr.update_backbone_features(x,mask)
        ptrs=[v.data_ptr() for v in fr._dit_cross_K+fr._dit_cross_V]
        before=fr._dit_cross_K[1].clone()
        graphs=fr._dit_graphs=object();fr._dit_attn=object()
        x[:,mask]*=2;fr.update_backbone_features(x,mask)
        self.assertEqual(ptrs,[v.data_ptr() for v in fr._dit_cross_K+fr._dit_cross_V])
        self.assertIs(fr._dit_graphs,graphs)
        self.assertFalse(torch.equal(before,fr._dit_cross_K[1]))
        fr.update_backbone_features(torch.ones(1,5,2048),torch.tensor([False,True,True,False,False]))
        self.assertFalse(hasattr(fr,'_dit_graphs'));self.assertFalse(hasattr(fr,'_dit_attn'))
        self.assertEqual(fr._dit_cross_K[0].shape,(3,2))

    def test_bad_features_do_not_replace_current_observation(self):
        fr=self.make_frontend();x=torch.ones(1,4,2048);mask=torch.tensor([False,True,True,False])
        fr.update_backbone_features(x,mask);old=fr._backbone_features
        with self.assertRaises(ValueError):fr.update_backbone_features(x*float('inf'),mask)
        with self.assertRaises(ValueError):fr.update_backbone_features(x,mask.float())
        fr._dit_k_w[0].fill_(float('inf'))
        with self.assertRaises(RuntimeError):fr.update_backbone_features(x*2,mask)
        self.assertIs(fr._backbone_features,old)


if __name__=='__main__':unittest.main()
