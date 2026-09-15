from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from types import SimpleNamespace
from benchmarks.vla import wan_va_libero_driver as d
from benchmarks.vla.util import ConfigurationError
from run_tests import run_module_tests

class Env:
    def __init__(self,success=34): self.env=self; self.success=success; self.timestep=0
    def seed(self,seed): pass
    def reset(self): self.timestep=0
    def set_init_state(self,state): pass
    def step(self,action):
        self.timestep+=1
        obs={k:np.ones((3,3,3),dtype=np.uint8) for k in ['agentview_image','robot0_eye_in_hand_image']}
        return obs,0,self.timestep>=self.success,{}

class Remote:
    def __init__(self,bad=False): self.calls=[]; self.bad=bad
    def reset_episode(self,prompt,seed): self.calls.append(('reset',seed))
    def infer(self,obs):
        if obs.get('compute_kv_cache'):
            self.calls.append(('commit',len(obs['obs']))); return {}
        self.calls.append(('infer',0))
        return {'action':np.full((7,4,4),np.nan if self.bad else 0.25,dtype=np.float32)}


def scene():
    env=Env(); obs=d.initialize(env,[0],7)
    return {'init_state':[0],'resolved_seed':7,'prompt':'test','initial_observation_sha256':d.observation_digest(obs)}


def test_libero_history_and_terminal_semantics():
    remote=Remote(); metrics,trace=d.rollout(Env(),remote,scene())
    assert metrics['success'] and metrics['executed_steps']==29
    assert [x[1] for x in remote.calls if x[0]=='commit']==[12,16]
    assert remote.calls[-1][0]=='infer'  # no successful-terminal commit
    assert len(trace)==29


def test_nan_actions_fail_before_robot_execution():
    try: d.rollout(Env(),Remote(True),scene())
    except ConfigurationError as e: assert 'finite' in str(e)
    else: raise AssertionError('invalid actions accepted')


def test_initial_observation_mismatch_fails_before_model_reset():
    s=scene();s['initial_observation_sha256']='bad';remote=Remote()
    try: d.rollout(Env(),remote,s)
    except ConfigurationError: assert not remote.calls
    else: raise AssertionError('scene drift accepted')

if __name__=='__main__': raise SystemExit(run_module_tests(globals()))
