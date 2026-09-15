import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import unittest
from types import SimpleNamespace
import numpy as np
from benchmarks.vla.pi05_scenes import checked_reset, observation_digest, state_record
from benchmarks.vla.util import ConfigurationError

class Vec:
    def __init__(self):
        self.envs=[SimpleNamespace(unwrapped=SimpleNamespace(_init_states=[[0.,1.]],task_description='pick'))]
        self.calls=0
        self.obs={'pixels':{'image':np.ones((1,3,3,3),dtype=np.uint8)},'state':np.zeros((1,8))}
    def reset(self,**kwargs):self.calls+=1;return copy.deepcopy(self.obs),{}

class SceneTests(unittest.TestCase):
    def scene(self,v):return {'seed':1,**state_record(v,1),'initial_observation_sha256':observation_digest(v.obs)}
    def test_checked_reset_consumes_exactly_one_real_reset(self):
        v=Vec();original=v.reset
        with checked_reset(v,self.scene(v)):v.reset(seed=[1])
        self.assertEqual(v.calls,1);self.assertEqual(v.reset,original)
    def test_image_and_proprioception_drift_refused(self):
        for key in ('pixels','state'):
            v=Vec();s=self.scene(v)
            if key=='pixels':v.obs[key]['image'][0,0,0,0]=2
            else:v.obs[key][0,0]=0.1
            with self.assertRaisesRegex(ConfigurationError,'initial observation'):
                with checked_reset(v,s):v.reset(seed=[1])
    def test_state_seed_and_missing_reset_refused(self):
        v=Vec();s=self.scene(v);v.envs[0].unwrapped._init_states=[[4.,5.]]
        with self.assertRaisesRegex(ConfigurationError,'state'):
            with checked_reset(v,s):pass
        v=Vec();s=self.scene(v)
        with self.assertRaisesRegex(ConfigurationError,'seed'):
            with checked_reset(v,s):v.reset(seed=[2])
        with self.assertRaisesRegex(ConfigurationError,'did not consume'):
            with checked_reset(v,s):pass
    def test_nonfinite_or_black_observation_refused(self):
        v=Vec();v.obs['state'][0,0]=np.nan
        with self.assertRaises(ConfigurationError):observation_digest(v.obs)
        v=Vec();v.obs['pixels']['image'][:]=0
        with self.assertRaises(ConfigurationError):observation_digest(v.obs)

if __name__=='__main__':unittest.main()
