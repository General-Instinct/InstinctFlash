import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmarks.vla.joint_robotwin_driver import rollout
from benchmarks.vla.joint_policy_server import Policy
from benchmarks.vla.robotwin_driver import initial_state
from benchmarks.vla.util import ConfigurationError,sha256_json

class Env:
    def __init__(self):self.take_action_cnt=0;self.step_lim=10;self.eval_success=False;self.actions=[]
    def setup_demo(self,**kw):self.take_action_cnt=0
    def set_instruction(self,instruction):pass
    def get_obs(self):return {'observation':{k:{'rgb':np.ones((2,2,3),dtype=np.uint8)} for k in ('head_camera','left_camera','right_camera')},'joint_action':{'vector':np.zeros(14)},'endpose':{k:np.zeros(7) for k in ('left_endpose','right_endpose','left_gripper','right_gripper')}}
    def take_action(self,a,action_type):
        assert action_type=='qpos';self.actions.append(a);self.take_action_cnt+=1;self.eval_success=self.take_action_cnt==3
class Remote:
    def reset_episode(self,*args):pass
    def infer(self,obs):return {'action':np.ones((25,14))}
class Tests(unittest.TestCase):
    def test_terminal_chunk_only_hashes_executed_actions(self):
        env=Env();scene={'resolved_seed':1,'prompt':'test','initial_state_sha256':initial_state(env.get_obs())}
        r=rollout(env,{},scene,Remote(),[25,14])
        self.assertTrue(r['success']);self.assertEqual(r['executed_steps'],3);self.assertEqual(len(r['action_values']),42)
    def test_bad_initial_observation_refused_before_inference(self):
        with self.assertRaisesRegex(ConfigurationError,'initial'):
            rollout(Env(),{}, {'resolved_seed':1,'prompt':'test','initial_state_sha256':'bad'},Remote(),[25,14])
    def test_model_server_requires_seeded_reset_and_checks_shape(self):
        class Arm:
            def new_episode(self,p):pass
            def predict(self,o):return np.zeros(3)
        identity={'execution':{'action_shape':[25,14]}}
        p=Policy(Arm(),identity,lambda seed:None)
        with self.assertRaises(ConfigurationError):p.infer({})
        p.infer({'reset':True,'prompt':'test','benchmark_seed':1,'benchmark_identity_sha256':sha256_json(identity)})
        with self.assertRaises(ConfigurationError):p.infer({})

if __name__=='__main__':unittest.main()
