"""Small Thor lifecycle/ownership tests using real CUDA Graphs and BF16 tensors."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from cosmos3_iwm.conditioning_cache import ConditioningCache
from cosmos3_iwm import conditioning_cache as implementation

output = Path(sys.argv[1])
assert not output.exists()
lock = open('/tmp/thor_gpu.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
torch.manual_seed(913)
torch.backends.cuda.matmul.allow_tf32 = False


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, torch.nn.Linear(16, 16, bias=False))
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = torch.nn.Linear(16, 16, bias=False)
        self.fail = False

    def forward(self, text, noisy):
        x = self.input_layernorm(text)
        a = self.self_attn
        q = a.q_norm(a.q_proj(x))
        k = a.k_norm(a.k_proj(x))
        v = a.v_proj(x)
        if self.fail:
            raise RuntimeError('injected partial prefill failure')
        text_out = a.o_proj(q + k + v)
        text_out = self.mlp(self.post_attention_layernorm(text_out))
        return text_out + noisy


class Model:
    def __init__(self):
        self.config = SimpleNamespace(joint_attn_implementation='two_way',
            video_temporal_causal=False, sound_gen=False)
        self.parallel_dims = None
        layers = torch.nn.ModuleList([Layer().cuda().to(torch.bfloat16)])
        self.net = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
        self.payload = None

    @property
    def layer(self):
        return self.net.language_model.model.layers[0]


def bind(model, verify=False):
    cache = ConditioningCache(model, verify=verify, max_slots=2)
    def velocity(**kw):
        return model.layer(model.payload, kw['noise_x'][0])
    def generate(prompts, base, *, drift=False):
        outputs = []
        for step in range(4):
            for tokens in prompts:
                model.payload = torch.full((4, 16), base + sum(tokens)/16,
                    device='cuda', dtype=torch.bfloat16)
                if drift and step == 1:
                    model.payload.add_(1)
                noisy = torch.full_like(model.payload, step/8)
                outputs.append(cache.velocity(velocity, text_tokens=[tokens],
                    sequence_plans=[None], noise_x=[noisy], skip_text_tokens=False))
        return outputs
    return cache, lambda *a, **kw: cache.generate(lambda payload: generate(*payload[0], **payload[1]), (a, kw))


def expected(model, prompts, base):
    return [model.layer(torch.full((4,16), base + sum(tokens)/16,
                device='cuda', dtype=torch.bfloat16),
            torch.full((4,16), step/8, device='cuda', dtype=torch.bfloat16))
        for step in range(4) for tokens in prompts]


def equal(left, right):
    return all(bool(torch.isfinite(a).all()) and
        torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
        for a, b in zip(left, right)) and len(left) == len(right)


with torch.inference_mode():
    model = Model()
    reference = copy.deepcopy(model)
    cache, run = bind(model)
    retained = []
    # Same token identity with changed conditioning verifies payload refresh;
    # reversed CFG order and LRU eviction exercise branch/graph ownership.
    for prompts, base in [([(1,2),(3,4)], 0), ([(3,4),(1,2)], 1),
                          ([(5,6),(7,8)], 2), ([(1,2),(3,4)], 3)]:
        outputs = run(prompts, base)
        assert equal(outputs, expected(reference, prompts, base))
        for live, snapshot in retained:
            assert equal(live, snapshot), 'A later replay/refill overwrote retained outputs'
        retained.append((outputs, [x.clone() for x in outputs]))
    model.layer.mlp.weight.add_(0.03125)
    reference.layer.mlp.weight.add_(0.03125)
    assert equal(run([(1,2),(3,4)], 4), expected(reference, [(1,2),(3,4)], 4))
    model.layer.mlp.weight = torch.nn.Parameter(model.layer.mlp.weight.clone() * 1.125)
    reference.layer.mlp.weight = torch.nn.Parameter(reference.layer.mlp.weight.clone() * 1.125)
    assert equal(run([(1,2),(3,4)], 5), expected(reference, [(1,2),(3,4)], 5))
    for live, snapshot in retained:
        assert equal(live, snapshot), 'Weight changes overwrote a retained output'
    assert cache.stats['evictions'] > 0
    assert sum(g['replays'] for g in cache.report()['graph_stats']) > 0
    assert not any(g['rejected'] for g in cache.report()['graph_stats'])

    check_model = Model()
    check_reference = copy.deepcopy(check_model)
    checked, checked_run = bind(check_model, verify=True)
    drifted = checked_run([(1,2),(3,4)], 0, drift=True)
    assert checked.disabled and checked.stats['rejected']
    # The request that fails admission must return the original computations,
    # including the deliberately changed conditioning at step 1.
    expected_drift = expected(check_reference, [(1,2),(3,4)], 0)
    for i, tokens in enumerate([(1,2),(3,4)]):
        expected_drift[2+i] = check_reference.layer(
            torch.full((4,16), 1 + sum(tokens)/16, device='cuda', dtype=torch.bfloat16),
            torch.full((4,16), 1/8, device='cuda', dtype=torch.bfloat16))
    assert equal(drifted, expected_drift)
    assert not checked.request and checked.active is None and not checked.seen
    assert equal(checked_run([(1,2),(3,4)], 1), expected(check_reference, [(1,2),(3,4)], 1))
    check_model = Model()
    check_reference = copy.deepcopy(check_model)
    checked, checked_run = bind(check_model, verify=True)
    check_model.layer.fail = True
    try:
        checked_run([(1,2),(3,4)], 2)
    except RuntimeError as error:
        assert 'partial prefill' in str(error)
    else:
        raise AssertionError('Injected failure was swallowed')
    assert not checked.request and checked.active is None and not checked.seen
    check_model.layer.fail = False
    assert equal(checked_run([(1,2),(3,4)], 3), expected(check_reference, [(1,2),(3,4)], 3))

    graph_report = cache.report()
    cache.close()
    cache.close()
    assert cache.closed and not cache.slots and not cache.patches
    assert equal(expected(model, [(1,2),(3,4)], 6), expected(reference, [(1,2),(3,4)], 6))

result = dict(ok=True, checks=['changed conditioning between requests', 'CFG branch order',
    'LRU eviction', 'retained output ownership', 'in-place weight update',
    'parameter replacement', 'invariance violation returns original output',
    'partial prefill failure', 'successful retry after failure', 'idempotent close restores forwards'],
    torch=torch.__version__, graphs=graph_report, closed=cache.closed, verification=checked.report(),
    sources={'probe.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'conditioning_cache.py': hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()})
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
