"""Action scheduling and processor boundaries, independent of GPU kernels."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from instinctflash.runtime.pi05_engine import Pi05EngineLoop


def fixture(dim=7, bad_shape=False):
    calls=[]
    class Engine:
        def set_prompt(self, ids):
            calls.append(('tokens',ids.tolist()))
        def infer(self, obs):
            calls.append(('images',[float(x[0,0,0]) for x in obs['images']]))
            width=7 if bad_shape else 32
            return {'actions':np.arange(3*width,dtype=np.float32).reshape(3,width)}
    def pre(batch):
        state=int(batch['observation.state'][0].item())
        return {'observation.language.tokens':torch.tensor([[state,8,999]]),
                'observation.language.attention_mask':torch.tensor([[True,True,False]])}
    def images(batch):
        return ([torch.full((1,3,4,4),v) for v in (0.25,0.5,0.75)],
                [torch.tensor([v]) for v in (True,False,True)])
    def build(views):
        calls.append(('build',views));return Engine()
    cfg=SimpleNamespace(output_features={'action':SimpleNamespace(shape=[dim])},
                        chunk_size=3,n_action_steps=2)
    loop=Pi05EngineLoop(cfg,pre,lambda x:2*x+1,images,build,'cpu')
    return loop,calls


def test_native_postprocessor_and_queue_preserve_declared_dimensions():
    for dim in (7,32):
        loop,calls=fixture(dim)
        first=loop.predict({'observation.state':np.array([4],dtype=np.float32)})
        second=loop.predict({'observation.state':np.array([5],dtype=np.float32)})
        assert np.array_equal(first['action'],2*np.arange(dim)+1)
        assert np.array_equal(second['action'],2*np.arange(32,32+dim)+1)
        assert calls==[('build',2),('tokens',[4,8]),('images',[0.25,0.75])]
        loop.predict({'observation.state':np.array([6],dtype=np.float32)})
        assert calls[-2]==('tokens',[6,8])
        loop.close()


def test_reset_drops_pending_actions_and_refreshes_processed_state():
    loop,calls=fixture()
    loop.predict({'observation.state':np.array([4],dtype=np.float32)})
    loop.reset(prompt='new episode')
    result=loop.predict({'observation.state':np.array([9],dtype=np.float32)})
    assert np.array_equal(result['action'],2*np.arange(7)+1)
    assert calls[-2]==('tokens',[9,8])
    assert sum(k=='build' for k,_ in calls)==1
    loop.close()


def test_truncated_engine_output_is_refused_before_robot_decoding():
    loop,_=fixture(bad_shape=True)
    try:
        loop.predict({'observation.state':np.array([4],dtype=np.float32)})
    except RuntimeError as e:
        assert 'unexpected action shape' in str(e)
    else:
        raise AssertionError('engine truncation was accepted')
    loop.close()


if __name__=='__main__':
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))
