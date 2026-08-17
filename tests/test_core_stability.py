#!/usr/bin/env python3
"""The core must get MORE stable as model support grows, and that is measurable.

The architecture's central claim is that a new model family enters through checkpoints, adapters,
capabilities and backends without making the runtime more model-specific. That is falsifiable, so it
is tested rather than asserted.

Measured before this test existed: exactly ONE model-name reference in executable code anywhere in the
generic layers -- `planners/planner.py` doing `from instinctwm.passes.lingbot import default_passes`,
which made one world model's pass set the default for every family in the ecosystem and meant a new
family could not add a pass without editing the planner. It is a registry now, and the count is zero.

The distinction the ratchet encodes: a REGISTRY is allowed to know what ships (`runtime/loader.py`
names the adapters it bundles, `passes/registry.py` names the passes). A PLANNER, a DESCRIPTOR, an
EXECUTOR or a VERIFIER is not allowed to know a model. Prose may name models freely -- concrete
examples are what make an abstract contract readable -- so only executable lines are counted.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILED: list[str] = []

#: Layers that must stay model-agnostic in their executable code.
GENERIC_DIRS = ("planners", "descriptors", "executors", "verify")
GENERIC_FILES = ("runtime/facade.py", "runtime/execution.py", "runtime/loader.py",
                 "passes/contract.py", "adapters/base.py")
MODEL_NAMES = re.compile(r"\b(wan_va|lingbot|LingBot|pi05|Pi05|gridworld|cosmos3|Cosmos3)\b")

#: The ratchet. Lower it when a leak is removed; raising it means accepting a branch, which is the
#: thing the architecture exists to prevent -- so raising it needs an argument, not a commit.
MAX_MODEL_REFERENCES = 0


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def _generic_paths():
    for d in GENERIC_DIRS:
        yield from sorted((ROOT / "instinctwm" / d).rglob("*.py"))
    for f in GENERIC_FILES:
        p = ROOT / "instinctwm" / f
        if p.exists():
            yield p


def _executable_lines(src: str):
    """Line numbers that are neither inside a string literal nor a comment."""
    doc = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            doc.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    for i, line in enumerate(src.splitlines(), 1):
        if i in doc:
            continue
        yield i, line.split("#", 1)[0]


def test_generic_layers_name_no_model():
    print("\n=== 1. the ratchet: model names in executable code, generic layers ===")
    found = []
    for p in _generic_paths():
        if "__pycache__" in str(p):
            continue
        for i, code in _executable_lines(p.read_text()):
            if MODEL_NAMES.search(code):
                found.append(f"{p.relative_to(ROOT)}:{i}  {code.strip()[:70]}")
    for f in found:
        print(f"    {f}")
    check(len(found) <= MAX_MODEL_REFERENCES,
          f"at most {MAX_MODEL_REFERENCES} model-name reference(s)", f"found {len(found)}")


def test_the_core_never_branches_on_model_identity():
    print("\n=== 2. no conditional on which model is being served ===")
    pat = re.compile(r"if .*(==|!=|\bin\b).*[\"'](wan_va|lingbot|pi05|act|cosmos3)[\"']")
    hits = []
    for p in _generic_paths():
        if "__pycache__" in str(p):
            continue
        for i, code in _executable_lines(p.read_text()):
            if pat.search(code):
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    check(not hits, "the core never branches on model identity", str(hits))


def test_passes_are_discovered_like_adapters():
    print("\n=== 3. passes come from a registry, symmetric with adapters ===")
    from instinctwm.passes.registry import (
        ENTRY_POINT_GROUP, default_passes, discover, providers, register_passes,
    )
    check(ENTRY_POINT_GROUP == "instinctwm.passes", "the group is stable", ENTRY_POINT_GROUP)
    check(discover() == [], "discovery reports no problems", str(discover()))
    names = providers()
    check("lingbot" in names, "the in-tree passes register as a provider", str(names))
    base = [getattr(p, "name", type(p).__name__) for p in default_passes()]
    check(len(base) >= 6, "and the default set is non-empty", str(len(base)))

    # an external provider can add passes without touching the core
    class Fake:
        name = "external_probe_pass"
        def evaluate(self, spec, deployment):                    # pragma: no cover
            raise NotImplementedError
    register_passes("external_probe", lambda: [Fake()])
    after = [getattr(p, "name", type(p).__name__) for p in default_passes()]
    check("external_probe_pass" in after, "an external provider's pass appears in the default set")
    register_passes("external_probe", lambda: [])                 # withdraw it again
    check("external_probe_pass" not in
          [getattr(p, "name", type(p).__name__) for p in default_passes()],
          "and can be withdrawn")


def test_a_broken_provider_does_not_break_planning():
    print("\n=== 4. one bad pass package cannot stop an unrelated model planning ===")
    from instinctwm.passes.registry import default_passes, register_passes

    def explodes():
        raise RuntimeError("third-party pass package is broken")

    register_passes("broken_probe", explodes)
    try:
        got = default_passes()
        check(len(got) >= 6, "the other providers still supply their passes", str(len(got)))
    finally:
        register_passes("broken_probe", lambda: [])


def test_the_observation_contract_is_declared_not_guessed():
    print("\n=== 5. every family declares what predict() expects ===")
    # numpy is needed only to BUILD an example, and the ratchet above must keep running without it.
    # Importing it at module scope made the whole file skip in the torch-free runner, which silently
    # switched off the architecture gate -- a skipped ratchet is a ratchet that ratchets nothing.
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("  SKIP: needs numpy to build an example observation "
              "(the ratchet in tests 1-2 still ran)")
        return
    # The CLI used to branch on notes["family"] == "vla" and then hardcode camera names, tensor shapes
    # and a history of 8. Three families want three different things -- and a fourth, VQ-BeT, declares
    # a five-observation window, which that branch would have got wrong three ways. So it is declared.
    import sys as _s
    ex = str(ROOT / "examples" / "pi05_vla")
    if ex not in _s.path:
        _s.path.insert(0, ex)
    from instinctwm import load
    from act_iwm.adapter import ACTAdapter
    from pi05_iwm.adapter import Pi05Adapter

    cases = {"wan_va": load("wan_va").spec(), "pi05": Pi05Adapter().spec(),
             "act": ACTAdapter().spec()}
    for name, spec in cases.items():
        o = spec.observation
        check(bool(o.fields), f"{name} declares its observation fields", str(len(o.fields)))
        ex_obs = o.example()
        check(bool(ex_obs), f"{name} can build an example from its declaration")
        for c in o.conditioning:
            check(c in ex_obs, f"{name}: conditioning key {c!r} is present")
        if o.frames_key:
            check(isinstance(ex_obs[o.frames_key], list) and
                  len(ex_obs[o.frames_key]) == o.history,
                  f"{name}: {o.history} frames under {o.frames_key!r}")
        else:
            check(all(f.key in ex_obs for f in o.fields), f"{name}: flat keys present")

    # the contracts must actually DIFFER, or the abstraction is decorative
    sigs = {n: (s.observation.history, s.observation.frames_key, s.observation.batched,
                tuple(sorted(f.key for f in s.observation.fields))) for n, s in cases.items()}
    check(len(set(sigs.values())) == len(sigs), "all three contracts are distinct",
          str({n: v[:3] for n, v in sigs.items()}))

    # and the CLI must not name a model to build one
    cli = (ROOT / "instinctwm" / "cli.py").read_text()
    body = cli[cli.index("def _zero_observation"):]
    body = body[:body.index("def ", 10)] if "def " in body[10:] else body
    for banned in ("family", "cam_high", "observation.images.top", "240, 320"):
        check(banned not in body.split('"""')[-1],
              f"the CLI's observation builder does not mention {banned!r}")


def main() -> int:
    test_generic_layers_name_no_model()
    test_the_core_never_branches_on_model_identity()
    test_passes_are_discovered_like_adapters()
    test_a_broken_provider_does_not_break_planning()
    test_the_observation_contract_is_declared_not_guessed()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the generic layers name no model, and passes are discovered like adapters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
