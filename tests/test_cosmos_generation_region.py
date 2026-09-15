from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/cosmos3_policy'))
from cosmos3_iwm.adapter import Cosmos3PolicyAdapter
from cosmos3_iwm.generation_region import install
from instinctflash.planners.planner import Tier
from instinctflash.runtime.cosmos_droid import CosmosDROIDLoop


def test_numeric_permission_is_required_before_checkpoint_or_cuda(monkeypatch):
    monkeypatch.setenv('IFL_COSMOS3_GEN_REGIONS', '1')
    with pytest.raises(ValueError, match='numeric'):
        Cosmos3PolicyAdapter()._build_droid(None, device=None, nfe=None, precision='native',
                                          plan=SimpleNamespace(tier_ceiling=Tier.BITEXACT))


def test_numeric_permission_does_not_make_fp8_eligible(monkeypatch):
    monkeypatch.setenv('IFL_COSMOS3_GEN_REGIONS', '1')
    with pytest.raises(ValueError, match='native precision'):
        Cosmos3PolicyAdapter()._build_droid(None, device=None, nfe=None, precision='fp8',
                                          plan=SimpleNamespace(tier_ceiling=Tier.NUMERIC))


def test_unqualified_cache_rejected_before_compilation():
    with pytest.raises(ValueError, match='admitted conditioning'):
        install(SimpleNamespace(), SimpleNamespace(tier_ceiling=Tier.NUMERIC))


def test_lifecycle_closes_regions_before_prompt_cache():
    order = []
    loop = CosmosDROIDLoop.__new__(CosmosDROIDLoop)
    loop._service = SimpleNamespace(
        _ifl_generation_regions=SimpleNamespace(close=lambda: order.append('regions')),
        _ifl_conditioning_cache=SimpleNamespace(close=lambda: order.append('conditioning')))
    loop.close()
    loop.close()
    assert order == ['regions', 'conditioning']


def test_split_prefill_requires_regions_before_model_loading(monkeypatch):
    monkeypatch.setenv('IFL_COSMOS3_SPLIT_PREFILL', '1')
    monkeypatch.setenv('IFL_COSMOS3_GEN_REGIONS', '0')
    with pytest.raises(ValueError, match='requires Cosmos GEN regions'):
        Cosmos3PolicyAdapter()._build_droid(None, device=None, nfe=None, precision='native')


@pytest.mark.parametrize('bad_key', [False, True])
def test_split_prefill_gate_and_changed_request_refresh(monkeypatch, bad_key):
    import contextlib
    import torch
    import cosmos3_iwm.generation_region as module
    monkeypatch.setattr(module, 'sdpa_kernel', lambda *args: contextlib.nullcontext())
    owner = module.GenerationRegions.__new__(module.GenerationRegions)
    cache = SimpleNamespace(active={}, disabled=False, filling=True, validating=False)
    owner.cache = cache
    owner.stats = dict(rejected=[], prefill_checks=0, eager_checks=0,
                       compiled_calls=0, compiled_prefill_calls=0)
    positions = ({'causal_seq': torch.ones(1, 2), 'full_only_seq': torch.ones(2, 2)},) * 2
    pack = dict(causal_seq=torch.ones(1, 2), full_only_seq=torch.ones(2, 2),
                _causal_indices=torch.tensor([0]), _full_indices=torch.tensor([1, 2]),
                sample_offsets=torch.tensor([0, 3]))

    def prefill(hidden, cos, sin):
        return hidden + 1, hidden * 2, hidden * 3

    def compute(hidden, key, value, *metadata):
        return hidden + key.sum() + value.sum()

    def native(p, mask, pos):
        und, key, value = prefill(p['causal_seq'], None, None)
        cache.active.setdefault('gen_regions', {}).setdefault(0, {}).update(
            key=key + int(bad_key), value=value)
        return dict(p, causal_seq=und, full_only_seq=compute(p['full_only_seq'], key, value)), {}, None

    compiled = []

    def region(*args):
        compiled.append(args[1].clone())
        return compute(*args)

    forward = owner._forward(0, compute, region, prefill)(native)
    if bad_key:
        with pytest.raises(ValueError, match='UND prefill differs'):
            forward(pack, None, positions)
        assert compiled == []
        assert not cache.active['gen_regions'][0].get('prefill_checked', False)
        return
    forward(pack, None, positions)
    assert owner.stats['prefill_checks'] == 1
    assert compiled == []
    # Existing decode extraction admission is separate from prefill admission.
    cache.active['gen_regions'][0]['checked'] = True
    changed = dict(pack, causal_seq=pack['causal_seq'] * 7, all_seq=torch.tensor([-999.]))
    output = forward(changed, None, positions)[0]
    assert torch.equal(compiled[0], changed['causal_seq'] * 2)
    assert torch.equal(output['causal_seq'], changed['causal_seq'] + 1)
    assert torch.equal(output['full_only_seq'], torch.full((2, 2), 71.))
    assert 'all_seq' not in output
    assert owner.stats['compiled_prefill_calls'] == 1


def test_understanding_prefill_keeps_native_causal_key_separate_from_gen_key():
    import torch
    from cosmos3_iwm.understanding_prefill import UnderstandingPrefill
    identity = torch.nn.Identity()
    a = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1, head_dim=2,
                        q_proj=identity, k_proj=identity, v_proj=identity,
                        q_norm=identity, k_norm=identity, k_norm_und_for_gen=lambda x: x * 3,
                        o_proj=identity,
                        _apply_rotary_pos_emb=lambda q, k, cos, sin, **kw: (q, k))
    layer = SimpleNamespace(self_attn=a, input_layernorm=identity,
                            post_attention_layernorm=identity, mlp=identity)
    seen = []
    def attention(q, k, v):
        seen.append(k.clone())
        return v
    hidden = torch.tensor([[1., 2.], [3., 4.]])
    output, gen_key, value = UnderstandingPrefill(layer, attention=attention)(hidden, None, None)
    assert torch.equal(seen[0].squeeze(0).flatten(1), hidden)
    assert torch.equal(gen_key.flatten(1), hidden * 3)
    assert torch.equal(value.flatten(1), hidden)
    assert torch.equal(output, hidden * 4)


def test_contiguous_layout_rechecks_values_even_at_same_address():
    import torch
    from cosmos3_iwm.generation_region import validate_contiguous_layout
    ci, fi = torch.tensor([0, 1]), torch.tensor([2, 3, 4])
    validate_contiguous_layout(ci, fi)
    validate_contiguous_layout(ci.clone(), fi.clone())
    ci[1], fi[0] = 2, 1
    with pytest.raises(ValueError, match='UND tokens followed'):
        validate_contiguous_layout(ci, fi)
    with pytest.raises(ValueError, match='one-dimensional'):
        validate_contiguous_layout(torch.tensor([0.]), torch.tensor([1.]))


@pytest.mark.parametrize('und_tokens', [0, 3])
def test_contiguous_kv_eager_layer_matches_indexed_layer(und_tokens):
    import torch
    from cosmos3_iwm.generation_region import GenerationCompute, validate_contiguous_layout
    torch.manual_seed(310)
    nn = torch.nn
    a = SimpleNamespace(num_attention_heads=2, num_key_value_heads=1, head_dim=2,
        q_proj_moe_gen=nn.Linear(4, 4), k_proj_moe_gen=nn.Linear(4, 2),
        v_proj_moe_gen=nn.Linear(4, 2), o_proj_moe_gen=nn.Linear(4, 4),
        q_norm_moe_gen=nn.Identity(), k_norm_moe_gen=nn.Identity(),
        _apply_rotary_pos_emb=lambda q, k, cos, sin, **kw: (q, k))
    layer = SimpleNamespace(self_attn=a, input_layernorm_moe_gen=nn.Identity(),
        post_attention_layernorm_moe_gen=nn.Identity(),
        mlp_moe_gen=SimpleNamespace(up_proj=nn.Linear(4, 8), down_proj=nn.Linear(8, 4), act_fn=nn.SiLU()))
    ci, fi = torch.arange(und_tokens), torch.arange(und_tokens, und_tokens + 5)
    validate_contiguous_layout(ci, fi)
    args = (torch.randn(5, 4), torch.randn(und_tokens, 1, 2), torch.randn(und_tokens, 1, 2),
            torch.ones(5, 2), torch.zeros(5, 2), ci, fi)
    with torch.inference_mode():
        reference = GenerationCompute(layer)(*args)
        candidate = GenerationCompute(layer, contiguous_kv=True)(*args)
    assert torch.equal(reference.view(torch.uint8), candidate.view(torch.uint8))


def test_contiguous_kv_requires_regions_before_model_loading(monkeypatch):
    monkeypatch.setenv('IFL_COSMOS3_CONTIGUOUS_KV', '1')
    monkeypatch.setenv('IFL_COSMOS3_GEN_REGIONS', '0')
    with pytest.raises(ValueError, match='requires Cosmos GEN regions'):
        Cosmos3PolicyAdapter()._build_droid(None, device=None, nfe=None, precision='native')


def test_fused_linear_requires_regions_before_model_loading(monkeypatch):
    monkeypatch.setenv('IFL_BF16_LINEAR_RELU2', '1')
    monkeypatch.setenv('IFL_COSMOS3_GEN_REGIONS', '0')
    with pytest.raises(ValueError, match='fusion requires Cosmos GEN'):
        Cosmos3PolicyAdapter()._build_droid(None, device=None, nfe=None, precision='native')
