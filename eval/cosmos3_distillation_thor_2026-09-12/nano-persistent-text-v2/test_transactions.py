"""CPU lifecycle tests, not substitutes for actual cached-model action comparison."""
from collections import OrderedDict
import torch
import pytest
from persistent_cache import PersistentTransactions, model_signature

class Harness(PersistentTransactions):
    def __init__(self):
        self.model=torch.nn.Linear(2,2).eval()
        self.slots=OrderedDict();self.seen=set();self.checks=[]
        self.request=self.disabled=self.closed=False
        self.active=None;self.filling=self.validating=False
        self.stats=dict(bypasses=0,requests=0)
        self.persistent_init()
    def fill(self,key):
        self.slots[key]={'complete':True};self.seen.add(key)

def run(cache,callback,**kwargs):
    with torch.no_grad():return cache._generate(callback,**kwargs)

def test_success_publishes_and_exception_discards_partial_storage():
    c=Harness();run(c,lambda:c.fill('A'));assert c.published=={'A'}
    def failed():
        assert c.seen=={'A'}
        c.fill('B');raise RuntimeError('generation failed')
    with pytest.raises(RuntimeError,match='generation failed'):run(c,failed)
    assert not c.published and not c.slots and not c.request and not c.seen
    run(c,lambda:c.fill('A'));assert c.published=={'A'}
    assert c.persistent_stats==dict(commits=2,aborts=1,invalidations=0)

def test_evicted_key_is_not_published_or_reused():
    c=Harness();run(c,lambda:c.fill('A'));c.slots.pop('A')
    def next_generation():
        assert 'A' not in c.seen
        c.fill('B')
    run(c,next_generation);assert c.published=={'B'}

def test_changed_weight_forces_refill():
    c=Harness();run(c,lambda:c.fill('A'))
    with torch.no_grad():c.model.weight.add_(1)
    def refill():
        assert not c.seen and not c.slots
        c.fill('A')
    run(c,refill);assert c.persistent_stats['invalidations']==1

def test_reentrant_generation_aborts_outer_transaction():
    c=Harness();run(c,lambda:c.fill('A'))
    with pytest.raises(RuntimeError,match='Reentrant'):
        run(c,lambda:run(c,lambda:None))
    assert not c.slots and not c.published and not c.request

def test_bypassed_custom_generation_cannot_publish():
    c=Harness();run(c,lambda:c.fill('A'))
    run(c,lambda **kw:None,velocity_postprocess_builder=object())
    assert not c.published and not c.slots

def test_untracked_inference_parameters_rejected():
    with torch.inference_mode():model=torch.nn.Linear(2,2)
    with pytest.raises(ValueError,match='tracked parameter versions'):model_signature(model)

def test_changed_guidance_invalidates_existing_slots():
    c=Harness();run(c,lambda **kw:c.fill('A'),guidance=1)
    def changed(**kw):
        assert not c.seen and not c.slots
        c.fill('B')
    run(c,changed,guidance=4)
    assert c.published=={'B'} and c.persistent_stats['invalidations']==1

class OneSlotParent:
    def velocity(self, original, **kwargs):
        key=(tuple(tuple(x) for x in kwargs['text_tokens']),False)
        if key not in self.slots:
            if self.slots:self.slots.popitem(last=False)
            self.slots[key]={}
        hit=key in self.seen
        self.seen.add(key)
        return hit

class OneSlotHarness(Harness,OneSlotParent):
    pass

def test_eviction_and_reappearance_within_one_generation_refills():
    c=OneSlotHarness()
    def requests():
        return [c.velocity(lambda:None,text_tokens=[[token]]) for token in (1,2,1)]
    assert run(c,requests)==[False,False,False]
    assert c.published=={(((1,),),False)}
    assert run(c,lambda:c.velocity(lambda:None,text_tokens=[[1]])) is True

def test_changed_buffer_forces_refill():
    c=Harness();c.model.register_buffer('condition_scale',torch.ones(1))
    run(c,lambda:c.fill('A'))
    c.model.condition_scale.add_(1)
    def refill():
        assert not c.seen and not c.slots
        c.fill('B')
    run(c,refill);assert c.published=={'B'}

def test_graph_pool_belongs_to_slot_and_retired_stats_remain_visible(monkeypatch):
    import persistent_cache as implementation
    pools=[]
    def allocate():
        pool=object();pools.append(pool);return pool
    class FakeGraph:
        def __init__(self,original,stats,name,*,pool):
            self.original=original;self.pool=pool;self.stats=stats
        def __call__(self,value):
            self.stats['replays']+=1
            return self.original(value)
    monkeypatch.setattr(torch.cuda,'graph_pool_handle',allocate)
    monkeypatch.setattr(implementation,'LayerGraph',FakeGraph)
    c=implementation.PersistentConditioningCache.__new__(implementation.PersistentConditioningCache)
    c.disabled=c.filling=c.validating=False;c.graph_history=[]
    new_slot=lambda:dict(graphs={},graph_stats={})
    c.active=new_slot()
    first=c.active
    layer0=c.wrap_layer(lambda x:x+1,lambda x:-1,0)
    layer1=c.wrap_layer(lambda x:x*2,lambda x:-1,1)
    assert layer0(2)==3 and layer1(2)==4
    assert len(pools)==1
    assert first['graphs'][0].pool is first['graphs'][1].pool
    c.active=new_slot();assert layer0(3)==4
    assert len(pools)==2 and c.active['pool'] is not first['pool']
    c.active=first;assert layer0(4)==5 and len(pools)==2
    c.active=None;assert layer0(3)==-1
    del first
    assert len(c.graph_history)==3 and sum(g['replays'] for g in c.graph_history)==4
