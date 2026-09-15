import random,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/lingbot_vla_v2'))
from lingbot_vla_v2_iwm.adapter import _LingBotVLAV2Loop,_release_prefix_graphs_on_fail,LingBotVLAV2Adapter

@unittest.skipIf(torch is None, 'torch/numpy are required for RNG fallback checks')
class CompleteFallback(unittest.TestCase):
    def test_unplanned_numeric_preprocessing_is_not_installed(self):
        server=SimpleNamespace(vla=SimpleNamespace(model=object()))
        with patch.dict('os.environ',{'IFL_VLA2_GPU_PREPROCESS':'1','IFL_VLA2_CUDA_KERNELS':'0',
                                    'IFL_VLA2_MOE_KERNEL':'0','IFL_VLA2_RMSNORM_KERNEL':'0'}):
            self.assertIsNone(LingBotVLAV2Adapter().install(server,SimpleNamespace(results=[]),device='cuda:0'))
        self.assertFalse(hasattr(server,'_instinctflash_gpu_preprocess_pending'))
    def test_rejection_closes_preprocessing_and_prefix(self):
        closed=[];server=SimpleNamespace(_instinctflash_gpu_preprocess=SimpleNamespace(close=lambda:closed.append('images')),
            _instinctflash_prefix_capture=SimpleNamespace(close=lambda:closed.append('prefix')),
            _instinctflash_gpu_preprocess_pending=('cuda:0','processor'))
        _release_prefix_graphs_on_fail(lambda r:None,server)({'passed':False})
        self.assertEqual(closed,['images','prefix']);self.assertIsNone(server._instinctflash_gpu_preprocess)
        self.assertFalse(hasattr(server,'_instinctflash_gpu_preprocess_pending'))
    def test_failed_inflight_call_is_discarded_without_consuming_rng_or_counter(self):
        driver=SimpleNamespace(rejected=False,graph=None)
        def draw():return torch.rand(1).item()+random.random()+np.random.rand()
        def seed():torch.manual_seed(13);random.seed(14);np.random.seed(15)
        seed();expected=draw();seed()
        class Server:
            global_step=0;last_action_chunk=None;last_normalized_action_chunk=None;calls=0
            def infer(self,obs):
                self.calls+=1;self.global_step+=1;value=draw()
                if self.calls==1:driver.rejected=True;value+=100
                self.last_action_chunk=np.array([value]);return {'action':self.last_action_chunk}
        server=Server();loop=object.__new__(_LingBotVLAV2Loop)
        loop._server=server;loop._driver=driver;loop._prompt='test'
        actual=loop.predict({'prompt':'test'})['action'][0]
        self.assertEqual(actual,expected);self.assertEqual(server.global_step,1);self.assertEqual(server.calls,2)
    def test_experimental_kernel_rejection_cannot_claim_native_fallback(self):
        server=SimpleNamespace(_instinctflash_moe_kernel=object())
        with self.assertRaisesRegex(RuntimeError,'automatic native fallback'):
            _release_prefix_graphs_on_fail(lambda r:None,server)({'passed':False})
        self.assertTrue(server._instinctflash_fallback_refused)
        loop=object.__new__(_LingBotVLAV2Loop);loop._server=server;loop._prompt='test'
        with self.assertRaisesRegex(RuntimeError,'reload'):loop.predict({})

if __name__=='__main__':unittest.main()
