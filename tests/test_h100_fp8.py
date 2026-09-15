"""H100 precision permission, real packed GEMMs, and native feedback ownership."""
from types import SimpleNamespace
import pytest
import torch
from instinctflash.runtime.h100_fp8 import H100Loop, H100FP8Linear, maybe_install_h100_fp8


def test_native_plan_never_replaces_projections():
    model=torch.nn.Linear(16,16)
    maybe_install_h100_fp8(model,SimpleNamespace(results=[]),'pi05')
    assert not hasattr(model,'_h100_fp8_recipe')


def test_deferred_commit_uses_executed_action_and_reset():
    events=[]
    class Native:
        def predict(self,obs):events.append(('predict',obs));return {'action':'predicted'}
        def commit(self,obs,action):events.append(('commit',action))
        def reset(self,**kw):events.append(('reset',kw))
    loop=H100Loop(Native(),{})
    loop.predict({'frame':1},executed_action='actually executed')
    loop.predict({'frame':2})
    loop.reset(prompt='next')
    assert events==[('predict',{'frame':1}),('commit','actually executed'),('predict',{'frame':2}),('commit','predicted'),('reset',{'prompt':'next'})]


def test_unsupported_feedback_refused_before_policy_call():
    class Native:
        def validate_executed_action(self,action):
            if action is not None:raise ValueError('unsupported feedback')
        def predict(self,obs):raise AssertionError('must not call policy')
    with pytest.raises(ValueError,match='unsupported feedback'):H100Loop(Native(),{}).predict({},executed_action=[1])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0),reason='H100 required')
def test_real_h100_fp8_and_graph_replay():
    torch.manual_seed(7)
    source=torch.nn.Linear(256,128,device='cuda',dtype=torch.bfloat16)
    layer=H100FP8Linear(source)
    assert layer.weight_fp8.dtype==torch.float8_e4m3fn
    x=torch.randn(32,256,device='cuda',dtype=torch.bfloat16)
    expected=layer(x)
    assert torch.isfinite(expected).all()
    # A coarse numerical sanity bound, not a model-quality certificate.
    assert (expected.float()-source(x).float()).abs().mean()<.1
    for _ in range(3):layer(x)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):actual=layer(x)
    g.replay();torch.cuda.synchronize()
    assert torch.equal(expected,actual)
    with pytest.raises(ValueError,match='BF16'):layer(x.float())
