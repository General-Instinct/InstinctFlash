"""Permission and isolation checks for native numerical attention."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/cosmos3_policy'))
from cosmos3_iwm.adapter import _numeric_attention_requested
from cosmos3_iwm.numeric_attention import NumericAttention
from instinctflash.planners.planner import Tier


def test_numeric_permission_does_not_change_precision(monkeypatch):
    monkeypatch.delenv('IFL_COSMOS3_ATTENTION', raising=False)
    strict = SimpleNamespace(tier_ceiling=Tier.BITEXACT)
    numeric = SimpleNamespace(tier_ceiling=Tier.NUMERIC)
    assert not _numeric_attention_requested('native', strict, True)
    assert _numeric_attention_requested('native', numeric, True)
    assert not _numeric_attention_requested('fp8', numeric, False)
    assert not _numeric_attention_requested('native', numeric, False)
    monkeypatch.setenv('IFL_COSMOS3_ATTENTION', 'native')
    assert not _numeric_attention_requested('native', numeric, True)
    monkeypatch.setenv('IFL_COSMOS3_ATTENTION', 'cudnn')
    with pytest.raises(ValueError, match='tier_ceiling'):
        _numeric_attention_requested('native', strict, True)
    with pytest.raises(ValueError, match='native Thor'):
        _numeric_attention_requested('fp8', numeric, False)


def test_dispatch_isolation_and_owned_cleanup():
    import torch
    # Independent namespace models the vendor's global dispatcher without loading it.
    namespace = {}
    exec('def attention(q, k, v, **kw): return q + 7\n'
         'def two_way_attention(q, k, v): return attention(q, k, v)\n'
         'def dispatch_attention(q, k, v): return two_way_attention(q, k, v)\n', namespace)
    module = SimpleNamespace(**namespace)
    original = module.dispatch_attention
    first = SimpleNamespace(dispatch_attention_fn=original)
    second = SimpleNamespace(dispatch_attention_fn=original)
    installed = NumericAttention(module, [first])
    tensor = torch.ones(1)  # CPU/unsupported shapes must retain vendor semantics.
    assert torch.equal(first.dispatch_attention_fn(tensor, tensor, tensor), tensor + 7)
    assert installed.report()['fallback_python_calls'] == 1
    assert second.dispatch_attention_fn is original
    assert module.dispatch_attention is original
    assert namespace['two_way_attention'] is module.two_way_attention
    installed.close()
    installed.close()
    assert first.dispatch_attention_fn is original
    installed = NumericAttention(module, [first])
    replacement = lambda *a: None
    first.dispatch_attention_fn = replacement
    installed.close()
    assert first.dispatch_attention_fn is replacement


def test_build_time_installation_is_not_reported_as_unoptimized(capsys):
    from instinctflash.runtime.execution import InProcessBackend
    adapter = SimpleNamespace()
    checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone='cosmos3_policy'))
    plan = SimpleNamespace(results=[SimpleNamespace(name='cosmos3_cudnn_attention', applies=True)])
    backend = InProcessBackend(adapter, checkpoint, plan)
    backend._report_unapplied()
    output = capsys.readouterr().out
    assert 'backend statistics' in output
    assert 'NONE' not in output and 'runs unoptimized' not in output
