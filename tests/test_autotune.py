#!/usr/bin/env python3
"""The autotune runner, exercised with a stubbed bench. No GPU, no torch.

What is pinned here, and why it matters:

  * cache round-trip -- the second load of the same (device, model, site, shape) must not
    re-bench, because a fleet re-timing what it already knows is the cost the cache exists
    to remove.
  * overrides -- IFL_AUTOTUNE=0 and IFL_AUTOTUNE_<SITE> are the operator's escape hatches, and
    an escape hatch that half-works (forced choice leaking into the cache, or forcing past the
    tier ceiling silently) is worse than none.
  * tier surfacing -- a NUMERIC swap must make the plan NUMERIC by the existing max() rule, and
    a BITEXACT-equivalent swap must not move the tier. This is the pass discipline applied to
    autotune, and it is the part that keeps 'autotune' from becoming 'silent numerics change'.
  * determinism -- same stubbed timings, same winner, every time, including ties.

    python tests/test_autotune.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instinctflash.autotune import (  # noqa: E402
    Candidate, Decision, Site, autotune, cache_path, record_decision, register_site,
)
from instinctflash.passes.contract import DeviceProfile, Tier  # noqa: E402
from instinctflash.planners.planner import Plan  # noqa: E402


DEV = DeviceProfile(name="StubGPU-9000", capability=(9, 0), total_memory=80 << 30,
                    features=frozenset({"cuda", "cudnn"}))


def _site(name="stub_site", numeric=False):
    return Site(
        name=name,
        candidates=(
            Candidate("stock", Tier.BITEXACT, "the incumbent; selecting it changes nothing"),
            Candidate("fast", Tier.NUMERIC if numeric else Tier.BITEXACT,
                      "stub evidence for the test"),
        ),
        baseline="stock",
        shape_signature="1x160x8x128x160",
    )


class _Bench:
    """Deterministic stub: fixed ms per candidate, counts calls."""

    def __init__(self, ms):
        self.ms, self.calls = dict(ms), 0

    def __call__(self, cand):
        self.calls += 1
        return self.ms[cand.name]


class _env:
    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _tmp_cache():
    return tempfile.NamedTemporaryFile(suffix=".json", delete=False).name


def test_cache_round_trip():
    cache = _tmp_cache()
    os.unlink(cache)
    with _env(IFL_AUTOTUNE_CACHE=cache, IFL_AUTOTUNE=None, IFL_AUTOTUNE_STUB_SITE=None):
        site = _site()
        bench = _Bench({"stock": 2.0, "fast": 1.0})
        d1 = autotune(site, bench, model_id="m", device=DEV)
        assert d1.source == "measured", d1
        assert d1.chosen == "fast" and d1.over == "stock", d1
        assert abs(d1.speedup - 2.0) < 1e-9, d1.speedup
        first_calls = bench.calls
        assert first_calls > 0
        assert cache_path() == Path(cache)
        doc = json.loads(Path(cache).read_text())
        assert len(doc) == 1 and list(doc.values())[0]["chosen"] == "fast", doc

        d2 = autotune(site, bench, model_id="m", device=DEV)
        assert d2.source == "cache", d2
        assert d2.chosen == "fast" and bench.calls == first_calls, "cache hit must not re-bench"
        assert d2.timings_ms == {"stock": 2.0, "fast": 1.0}

        # a DIFFERENT model_id is a different key: it must measure again, not reuse
        d3 = autotune(site, bench, model_id="other", device=DEV)
        assert d3.source == "measured" and bench.calls > first_calls


def test_override_forced_and_disabled():
    cache = _tmp_cache()
    site = _site()
    bench = _Bench({"stock": 2.0, "fast": 1.0})
    with _env(IFL_AUTOTUNE_CACHE=cache, IFL_AUTOTUNE="0"):
        d = autotune(site, bench, model_id="m", device=DEV)
        assert d.source == "disabled" and d.chosen == "stock" and bench.calls == 0
    with _env(IFL_AUTOTUNE_CACHE=cache, IFL_AUTOTUNE=None, IFL_AUTOTUNE_STUB_SITE="fast"):
        d = autotune(site, bench, model_id="m", device=DEV)
        assert d.source == "forced" and d.chosen == "fast" and bench.calls == 0
        # a forced choice is a preference, not a measurement: it must not enter the cache
        assert json.loads(Path(cache).read_text() or "{}") == {} if Path(cache).exists() else True
    with _env(IFL_AUTOTUNE_CACHE=cache, IFL_AUTOTUNE=None, IFL_AUTOTUNE_STUB_SITE="typo"):
        try:
            autotune(site, bench, model_id="m", device=DEV)
        except KeyError as e:
            assert "typo" in str(e) and "stock" in str(e)
        else:
            raise AssertionError("unknown forced candidate must raise, with the names")


def test_forcing_past_the_ceiling_raises():
    site = _site(numeric=True)
    with _env(IFL_AUTOTUNE_CACHE=_tmp_cache(), IFL_AUTOTUNE=None, IFL_AUTOTUNE_STUB_SITE="fast"):
        try:
            autotune(site, _Bench({"stock": 2.0, "fast": 1.0}), model_id="m", device=DEV,
                     tier_ceiling=Tier.BITEXACT)
        except RuntimeError as e:
            assert "ceiling" in str(e).lower(), e
        else:
            raise AssertionError("forcing a NUMERIC candidate under a BITEXACT ceiling must raise")


def test_tier_ceiling_excludes_and_surfaces():
    site = _site("stub_ceiling", numeric=True)
    bench = _Bench({"stock": 2.0, "fast": 1.0})
    with _env(IFL_AUTOTUNE_CACHE=_tmp_cache(), IFL_AUTOTUNE=None, IFL_AUTOTUNE_STUB_CEILING=None):
        d = autotune(site, bench, model_id="m", device=DEV, tier_ceiling=Tier.BITEXACT)
        assert d.chosen == "stock" and d.source == "unopposed", d
        assert bench.calls == 0, "a candidate above the ceiling must not even be benched"
        assert "ceiling" in d.reason and "fast" in d.reason, d.reason


def test_plan_surfacing_and_tier_discipline():
    # NUMERIC swap: the plan must become NUMERIC, and explain() must carry the autotuned line.
    site = register_site(_site("stub_numeric_site", numeric=True))
    bench = _Bench({"stock": 2.0, "fast": 1.0})
    with _env(IFL_AUTOTUNE_CACHE=_tmp_cache(), IFL_AUTOTUNE=None,
              IFL_AUTOTUNE_STUB_NUMERIC_SITE=None):
        d = autotune(site, bench, model_id="m", device=DEV, tier_ceiling=Tier.NUMERIC)
    plan = Plan("m", [])
    record_decision(plan, d)
    assert plan.tier() is Tier.NUMERIC, plan.tier()
    text = plan.explain()
    assert "autotuned: stub_numeric_site chose fast over stock (2.00x), equivalence NUMERIC" \
        in text, text
    assert plan.results[0].params["evidence"] == "stub evidence for the test"

    # BITEXACT-equivalent swap: the plan tier must NOT move.
    site2 = register_site(_site("stub_bitexact_site", numeric=False))
    with _env(IFL_AUTOTUNE_CACHE=_tmp_cache(), IFL_AUTOTUNE=None,
              IFL_AUTOTUNE_STUB_BITEXACT_SITE=None):
        d2 = autotune(site2, _Bench({"stock": 2.0, "fast": 1.0}), model_id="m", device=DEV)
    plan2 = Plan("m", [])
    record_decision(plan2, d2)
    assert plan2.results[0].applies and plan2.tier() is Tier.BITEXACT, plan2.explain()

    # kept baseline: recorded with applies=False, never omitted
    with _env(IFL_AUTOTUNE_CACHE=_tmp_cache(), IFL_AUTOTUNE=None,
              IFL_AUTOTUNE_STUB_BITEXACT_SITE=None):
        d3 = autotune(site2, _Bench({"stock": 1.0, "fast": 2.0}), model_id="m", device=DEV)
    plan3 = Plan("m", [])
    record_decision(plan3, d3)
    assert not plan3.results[0].applies and "autotune:stub_bitexact_site" in plan3.explain()

    # re-tuning replaces the entry rather than stacking a second one
    record_decision(plan3, d3)
    assert len([r for r in plan3.results if r.name == "autotune:stub_bitexact_site"]) == 1


def test_winner_determinism_given_a_stubbed_bench():
    site = _site("stub_det")
    with _env(IFL_AUTOTUNE=None, IFL_AUTOTUNE_STUB_DET=None):
        winners = set()
        for _ in range(5):
            with _env(IFL_AUTOTUNE_CACHE=_tmp_cache()):
                d = autotune(site, _Bench({"stock": 1.5, "fast": 1.0}), model_id="m", device=DEV)
            winners.add(d.chosen)
        assert winners == {"fast"}, winners
        # exact tie: declaration order decides, and the baseline is declared first
        with _env(IFL_AUTOTUNE_CACHE=_tmp_cache()):
            d = autotune(site, _Bench({"stock": 1.0, "fast": 1.0}), model_id="m", device=DEV)
        assert d.chosen == "stock", "a tie must go to the earlier declaration (the baseline)"


def test_crashing_candidate_loses_loudly():
    site = _site("stub_crash")

    def bench(c):
        if c.name == "fast":
            raise RuntimeError("no such kernel on this device")
        return 3.0

    with _env(IFL_AUTOTUNE_CACHE=_tmp_cache(), IFL_AUTOTUNE=None, IFL_AUTOTUNE_STUB_CRASH=None):
        d = autotune(site, bench, model_id="m", device=DEV)
    assert d.chosen == "stock" and "FAILED" in d.reason and "no such kernel" in d.reason, d.reason


def test_verify_refuses_a_failed_bitexact_claim():
    site = _site("stub_verify")  # 'fast' claims BITEXACT equivalence
    bench = _Bench({"stock": 2.0, "fast": 1.0})
    with _env(IFL_AUTOTUNE_CACHE=_tmp_cache(), IFL_AUTOTUNE=None, IFL_AUTOTUNE_STUB_VERIFY=None):
        d = autotune(site, bench, model_id="m", device=DEV, verify=lambda name: 6.25e-2)
    assert d.chosen == "stock" and d.source == "verify-failed", d
    assert "max|delta| = 6.250e-02" in d.reason and bench.calls == 0, d.reason


def test_stale_cache_entry_is_remeasured_not_trusted():
    cache = _tmp_cache()
    site = _site("stub_stale", numeric=True)
    bench = _Bench({"stock": 2.0, "fast": 1.0})
    with _env(IFL_AUTOTUNE_CACHE=cache, IFL_AUTOTUNE=None, IFL_AUTOTUNE_STUB_STALE=None):
        d = autotune(site, bench, model_id="m", device=DEV, tier_ceiling=Tier.NUMERIC)
        assert d.chosen == "fast" and d.source == "measured"
        # same cache, tighter ceiling: the cached NUMERIC winner is not legal now
        d2 = autotune(site, bench, model_id="m", device=DEV, tier_ceiling=Tier.BITEXACT)
        assert d2.chosen == "stock", "a cached winner the ceiling forbids must not serve"


def test_baseline_must_be_bitexact():
    try:
        Site("bad", (Candidate("a", Tier.NUMERIC, "x"),), baseline="a")
    except ValueError as e:
        assert "incumbent" in str(e)
    else:
        raise AssertionError("a NUMERIC baseline must be refused at declaration time")


if __name__ == "__main__":
    from run_tests import run_module_tests
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(run_module_tests(globals()))
