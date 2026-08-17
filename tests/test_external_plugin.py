#!/usr/bin/env python3
"""A second model family must be reachable without editing InstinctWM.

These are the invariants the external-author audit turned into regressions. No GPU, no weights.
The full runnable integration is examples/external_plugin/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def test_discovery_is_by_entry_point():
    print("\n=== 1. plugins are discovered from installed metadata, not import order ===")
    from instinctwm.runtime.loader import ENTRY_POINT_GROUP, discover_plugins
    check(ENTRY_POINT_GROUP == "instinctwm.adapters", "the group is stable and documented",
          ENTRY_POINT_GROUP)
    check(callable(discover_plugins), "discovery runs on lookup, so no user import is needed")
    # A broken third-party plugin must not take the runtime down with it.
    import instinctwm.runtime.loader as L
    L._DISCOVERED = False
    problems = L.discover_plugins()
    check(isinstance(problems, list), "discovery reports failures instead of raising", str(problems))


def test_lingbot_passes_do_not_fire_on_other_models():
    print("\n=== 2. LingBot's passes decline models they cannot rewrite ===")
    # These patch the LingBot server object. Applied to anything else they were claiming, in the
    # plan, that a toy GRU would get "measured 1.75x standalone on LingBot-VA".
    from instinctwm.passes.lingbot.substrate import (
        AllocatorChurnElision, DebugDumpElision, FSDPElision,
    )
    for cls in (FSDPElision, AllocatorChurnElision, DebugDumpElision):
        need = frozenset(getattr(cls, "requires_capabilities", ()) or ())
        check("backbone:wan_va" in need, f"{cls.name} declares its backbone requirement", str(set(need)))


def test_the_impl_contract_uses_the_public_verb():
    print("\n=== 3. an external impl implements predict(), not LingBot's infer() ===")
    src = (ROOT / "instinctwm" / "runtime" / "execution.py").read_text()
    check('getattr(impl, "predict", None)' in src, "predict is preferred")
    check('getattr(impl, "infer", None)' in src, "infer remains as compatibility")
    check('getattr(impl, "commit", None)' in src,
          "commit is an OPTIONAL adapter hook, so predict stays loopable")


def test_lifecycle_verbs_are_the_decided_set():
    print("\n=== 4. the public lifecycle is model / episode / cycle -- and nothing else ===")
    from instinctwm import Episode, Runtime
    for name in ("from_pretrained", "predict", "reset", "episode", "close"):
        check(hasattr(Runtime, name), f"Runtime.{name}")
    for name in ("step", "commit", "warmup", "flush"):
        check(not hasattr(Runtime, name), f"Runtime has NO {name}()")
    for name in ("predict", "close", "steps"):
        check(hasattr(Episode, name), f"Episode.{name}")
    check(not hasattr(Episode, "commit"), "Episode has NO commit() -- phases stay private")
    check(hasattr(Episode, "__enter__") and hasattr(Runtime, "__enter__"),
          "both lifetimes are context managers")


def test_declaration_carries_no_model_specific_knowledge():
    print("\n=== 5. model knowledge stays in the adapter, not in instinctwm.json ===")
    import json
    p = ROOT / "examples" / "external_plugin" / "my-world-model" / "instinctwm.json"
    if not p.exists():
        print("  (built artifact absent; run examples/external_plugin/build_checkpoint.py)")
        return
    ex = json.loads(p.read_text())["execution"]
    for k in ("vocab", "dim", "history", "hidden_size", "architecture"):
        check(k not in ex, f"execution declares no {k!r}")


def test_a_second_model_family_plans_correctly():
    print("\n=== 6. a VLA plans on its own declared merits, not LingBot's ===")
    # examples/pi05_vla declares lerobot/pi05_base: one stream, CHUNK-lifetime prefix K/V, no
    # guidance, no commit phase, no decode modules -- structurally unlike the world model this
    # framework was built around. If the abstraction is real, the generic pass fires and the
    # model-specific ones do not silently claim to.
    import sys as _s
    p = str(ROOT / "examples" / "pi05_vla")
    if p not in _s.path:
        _s.path.insert(0, p)
    from pi05_iwm.adapter import Pi05Adapter
    from instinctwm import Optimizer

    spec = Pi05Adapter().spec()
    check(spec.forwards_breakdown() == "prefix=1 + action=10",
          "the VLA declares its own phase structure", spec.forwards_breakdown())
    text = Optimizer().compile(spec).explain()

    def line(name):
        return next((l for l in text.splitlines() if name in l), "")

    check("APPLY" in line("conditioning_prefill") and "chunk" in line("conditioning_prefill"),
          "conditioning_prefill fires on a CHUNK-scope prefix it has never seen")
    check("skip" in line("cfg_branch_elision"), "cfg_branch_elision declines: no CFG declared")
    check("skip" in line("obs_decode_elision"), "obs_decode_elision declines: a VLA has no decoder")
    for nm in ("fsdp_elision", "allocator_churn_elision", "debug_dump_elision"):
        check("APPLICABILITY UNCHECKED" in line(nm),
              f"{nm} is not silently endorsed for another family")
    # and with capabilities supplied, the gate filters rather than annotates
    filtered = Optimizer().compile(spec, capabilities=frozenset({"servable", "backbone:pi05"}))
    ftext = filtered.explain()
    fline = next((l for l in ftext.splitlines() if "fsdp_elision" in l), "")
    check("skip" in fline and "does not declare" in fline,
          "a declared checkpoint filters the pass out entirely")


def test_shape_stability_is_derived_not_declared():
    print("\n=== 7. shape stability across control cycles is derived from the declaration ===")
    # The property that decides whether whole-cycle graph capture can pay. It generalises the one
    # structural difference three model families made visible: a stream that outlives a control cycle
    # accumulates, so shapes change and a captured graph is invalidated. LingBot-VA measured capture
    # at 1.43x SLOWER for exactly that reason; VLAs with no persistent stream are shape-stable, which
    # is why hand-tuned engines capture a whole VLA forward and win. Deriving it means the next family
    # does not have to discover it by building the pass and measuring the regression.
    import sys as _s
    for sub in ("pi05_vla",):
        p_ = str(ROOT / "examples" / sub)
        if p_ not in _s.path:
            _s.path.insert(0, p_)
    from instinctwm import load
    from act_iwm.adapter import ACTAdapter
    from pi05_iwm.adapter import Pi05Adapter

    ok_wm, why_wm = load("wan_va").spec().shapes_static_across_cycles()
    check(not ok_wm, "a world model with EPISODE streams is NOT shape-stable", why_wm[:60])
    check("outlive" in why_wm, "and the reason names the growing streams")
    ok_v, why_v = Pi05Adapter().spec().shapes_static_across_cycles()
    check(ok_v, "a VLA with a CHUNK-lifetime prefix IS shape-stable", why_v[:60])
    ok_a, why_a = ACTAdapter().spec().shapes_static_across_cycles()
    check(ok_a, "a policy with no persistent stream IS shape-stable", why_a[:60])


def test_a_no_refinement_model_plans_honestly():
    print("\n=== 8. a model with no refinement loop gets an honest, empty plan ===")
    import sys as _s
    p_ = str(ROOT / "examples" / "pi05_vla")
    if p_ not in _s.path:
        _s.path.insert(0, p_)
    from act_iwm.adapter import ACTAdapter
    from instinctwm import Optimizer
    spec = ACTAdapter().spec()
    check(spec.total_forwards() == 1, "ACT is one forward per control step, no denoise loop")
    text = Optimizer().compile(spec).explain()
    cp = next((l for l in text.splitlines() if "conditioning_prefill" in l), "")
    check("skip" in cp, "conditioning_prefill declines: there is no loop to hoist out of", cp[-58:])
    for nm in ("cfg_branch_elision", "obs_decode_elision"):
        check("skip" in next((l for l in text.splitlines() if nm in l), ""), f"{nm} declines")


def test_weights_may_be_supplied_by_reference():
    print("\n=== 9. a declaration can adopt an upstream checkpoint without vendoring it ===")
    # Found by declaring a LeRobot ACT policy: every byte lives in the upstream repo, so the only
    # sane package is a declaration plus a pointer. `base_weights` was already an execution fact,
    # but validate_package demanded local weight files, so adopting somebody else's checkpoint meant
    # copying their gigabytes first.
    import json
    import tempfile
    from instinctwm.descriptors.package import publishability, validate_package
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "declared"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "instinctwm.json").write_text(json.dumps({
            "instinctwm_schema": 1,
            "execution": {"model_id": "example-org/declared", "backbone": "act",
                          "servable": True, "base_weights": "upstream-org/some-policy"},
        }))
        r = validate_package(d)
        check(r.ok, "a pointer-only package is servable", "; ".join(r.missing) or "no missing")
        check(any("base_weights" in n for n in r.notes),
              "and the report says where the weights actually live")
        check(publishability(d)[0], "it is publishable too")

        # neither local weights nor a pointer is still incomplete
        (d / "instinctwm.json").write_text(json.dumps({
            "instinctwm_schema": 1,
            "execution": {"model_id": "example-org/nothing", "backbone": "act", "servable": True},
        }))
        r2 = validate_package(d)
        check(not r2.ok and any("weights" in m for m in r2.missing),
              "but a package with neither is refused", "; ".join(r2.missing))


def main() -> int:
    test_discovery_is_by_entry_point()
    test_lingbot_passes_do_not_fire_on_other_models()
    test_the_impl_contract_uses_the_public_verb()
    test_lifecycle_verbs_are_the_decided_set()
    test_declaration_carries_no_model_specific_knowledge()
    test_a_second_model_family_plans_correctly()
    test_shape_stability_is_derived_not_declared()
    test_a_no_refinement_model_plans_honestly()
    test_weights_may_be_supplied_by_reference()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: a second model family integrates by entry point, and the lifecycle is closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
