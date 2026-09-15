"""Checks for native controller conversion and frozen observation enforcement."""
import hashlib
import unittest
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from benchmarks.vla import groot_libero_driver as driver
from benchmarks.vla.util import ConfigurationError

class Env:
    def __init__(self):
        self._env=SimpleNamespace(step=self.raw_step)
        self.calls=0
    def reset(self,seed=None):
        self.calls=0
        return {'video.image':np.ones((2,2,3),np.uint8),'state.x':[0.]},{}
    def raw_step(self,action):
        self.calls+=1
        return {},0,False,{}
    def step(self,action):
        applied=np.concatenate([action['action.'+k] for k in driver.KEYS])
        applied[-1]=-np.sign(2*applied[-1]-1)
        self._env.step(applied)
        return {},0,False,False,{'success':True}

class Remote:
    def reset_episode(self,*args):pass
    def infer(self,obs):return {'action':np.full((16,7),.25)}

class TestGrootLibero(unittest.TestCase):
    def scene(self,env):
        return {'resolved_seed':0,'prompt':'test','initial_observation_sha256':driver.observation_digest(env.reset()[0])}
    def test_digest_tracks_applied_gripper_and_stops_at_success(self):
        env=Env();original=env._env.step
        result=driver.rollout(env,Remote(),self.scene(env))
        expected=[.25]*6+[1.]
        self.assertEqual(result['executed_steps'],1)
        self.assertEqual(result['action_values'],expected)
        self.assertEqual(result['action_digest'],hashlib.sha256(np.asarray(expected,dtype='>f8').tobytes()).hexdigest())
        self.assertEqual(env._env.step,original)
    def test_frozen_observation_mismatch_refused(self):
        env=Env();scene=self.scene(env);scene['initial_observation_sha256']='0'*64
        with self.assertRaises(ConfigurationError):driver.rollout(env,Remote(),scene)
        self.assertEqual(env.calls,0)
    def test_nonfinite_action_refused(self):
        env=Env();remote=Remote();remote.infer=lambda _: {'action':np.full((16,7),np.nan)}
        with self.assertRaises(ConfigurationError):driver.rollout(env,remote,self.scene(env))
        self.assertEqual(env.calls,0)

if __name__=='__main__':unittest.main()


def test_source_binding_rejects_different_import_and_assets(tmp_path):
    base=tmp_path/'libero/libero'
    module=ModuleType('libero.libero')
    module.benchmark=SimpleNamespace(__file__=str(base/'benchmark/__init__.py'))
    paths={'assets':base/'assets','init_states':base/'init_files','bddl_files':base/'bddl_files'}
    module.get_libero_path=lambda key:str(paths[key])
    with patch.dict(sys.modules,{'libero.libero':module}):
        driver.validate_libero_binding(tmp_path)
        module.benchmark.__file__=str(tmp_path/'other/benchmark/__init__.py')
        with unittest.TestCase().assertRaisesRegex(ConfigurationError,'imported LIBERO code'):
            driver.validate_libero_binding(tmp_path)
        module.benchmark.__file__=str(base/'benchmark/__init__.py')
        paths['assets']=tmp_path/'other/assets'
        with unittest.TestCase().assertRaisesRegex(ConfigurationError,'assets'):
            driver.validate_libero_binding(tmp_path)


def test_wrapper_binding_rejects_cached_import_from_other_checkout(tmp_path):
    name='gr00t.eval.sim.LIBERO.libero_env'
    module=ModuleType(name)
    module.__file__=str(tmp_path/'gr00t/eval/sim/LIBERO/libero_env.py')
    with patch.dict(sys.modules,{name:module}), patch.object(sys,'path',list(sys.path)):
        assert driver.load_groot_wrapper(tmp_path) is module
        module.__file__=str(tmp_path/'other/gr00t/eval/sim/LIBERO/libero_env.py')
        with unittest.TestCase().assertRaisesRegex(ConfigurationError,'GR00T simulator wrapper'):
            driver.load_groot_wrapper(tmp_path)
