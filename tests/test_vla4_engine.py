"""Native sample_actions boundary checks without loading model weights."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from instinctflash.runtime.vla4_engine import Vla4ActionGenerator


def test_native_masks_state_noise_and_prompt_changes_reach_engine():
    class Frontend:
        def __init__(self):self.prompts=[];self.calls=[]
        def set_prompt(self,ids):self.prompts.append(ids)
        def infer_staged(self,images,state,noise):
            self.calls.append((images.clone(),state.clone(),noise.clone()))
            return {'actions':np.ones((50,75),dtype=np.float32)}
    fe=Frontend();generator=Vla4ActionGenerator(fe)
    images=torch.arange(3,dtype=torch.float32)[:,None,None].expand(3,256,1176)
    state=torch.zeros(1,75)
    noise=torch.ones(1,50,75)
    args=(images,torch.ones(3,dtype=torch.bool),torch.tensor([[3,9,999]]),
          torch.tensor([[True,True,False]]),state)
    a=generator.sample_actions(*args,noise=noise,num_steps=10)
    assert a.shape==(1,50,75) and fe.prompts==[[3,9]]
    generator.sample_actions(*args[:-1],state+2,noise=noise*3,num_steps=10)
    assert len(fe.prompts)==1
    assert torch.equal(fe.calls[-1][0],images)
    assert torch.equal(fe.calls[-1][1],state+2)
    assert torch.equal(fe.calls[-1][2],noise*3)
    generator.sample_actions(images,args[1],torch.tensor([[4,8]]),
        torch.ones(1,2,dtype=torch.bool),state,noise=noise,num_steps=10)
    assert fe.prompts==[[3,9],[4,8]]


def test_missing_camera_is_not_silently_included_in_attention():
    generator=Vla4ActionGenerator(None)
    try:
        generator.sample_actions(torch.zeros(3,256,1176),torch.tensor([True,False,True]),
            torch.tensor([[3]]),torch.tensor([[True]]),torch.zeros(1,75),num_steps=10)
    except ValueError as e:
        assert 'three active cameras' in str(e)
    else:
        raise AssertionError('unsupported masked camera reached engine')


if __name__=='__main__':
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))
