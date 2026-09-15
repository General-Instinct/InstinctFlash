import tempfile,unittest
from pathlib import Path
import numpy as np
from benchmarks.vla.policy_trace import PolicyTrace,read_trace,V2Noise
from benchmarks.vla.util import ConfigurationError
from instinctflash.serving.msgpack_numpy import Packer
class TraceTests(unittest.TestCase):
 def test_roundtrip_and_changed_bytes_refuse(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'trace';p=Packer();t=PolicyTrace(path,{'model':'real'})
   t.append(p.pack({'image':np.ones((2,2,3),dtype=np.uint8)}),p.pack({'action':np.array([-.0])}));t.close()
   _,calls=read_trace(path);self.assertTrue(np.signbit(calls[0][1]['action'][0]))
   (path/'00000.request.msgpack').write_bytes(b'changed')
   with self.assertRaises(ConfigurationError):read_trace(path)
 def test_noise_hook_preserves_native_draw_and_replays_actual_tensor(self):
  import torch
  from types import SimpleNamespace
  server=SimpleNamespace(vla=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(n_action_steps=2,max_action_dim=3))))
  server.sample_actions_fn=lambda *args,noise=None,**kwargs:noise.clone()
  hook=V2Noise(SimpleNamespace(_server=server));state=torch.zeros((1,3))
  torch.manual_seed(17);expected=torch.randn((1,2,3));expected_next=torch.randn((2,))
  torch.manual_seed(17);actual=server.sample_actions_fn(None,None,None,None,state)
  self.assertTrue(torch.equal(expected,actual));self.assertTrue(torch.equal(expected_next,torch.randn((2,))))
  hook.pending=hook.last.copy();actual2=server.sample_actions_fn(None,None,None,None,state)
  self.assertTrue(torch.equal(actual,actual2))
if __name__=='__main__':unittest.main()
