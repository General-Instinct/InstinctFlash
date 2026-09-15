#!/usr/bin/env python3
"""The platform claim, made checkable: many checkpoints, one runtime, no branching on training method.

CHECKPOINTS.md states the rule. `test_runtime_boundary.py` enforces the *static* half — no runtime
module imports a training package or names a provenance key. This file enforces the *behavioural*
half, which is the part that actually matters to a checkpoint author:

    two checkpoints with identical execution blocks and wildly different provenance
    must produce a byte-identical plan.

If that holds, "the training recipe never influences planning" is a measured property rather than a
convention someone might forget. The interesting case is deliberately adversarial: same weights,
same capabilities, one labelled PDD with a full training record and the other labelled with a recipe
that does not exist.

No GPU, no torch, no model.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instinctflash.descriptors.checkpoint import (  # noqa: E402
    FORBIDDEN_IN_EXECUTION, SCHEMA_VERSION, load_declaration, provenance_of,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


EXECUTION = {
    "model_id": "example/wm-blockheads-2v4a",
    "backbone": "wan_va",
    "servable": True,
    "guidance": {"video": "cfg", "action": "positive_only"},
    "nfe": {"video": 2, "action": 4},
    "output_projection": {
        "kind": "per_interval_velocity_heads",
        "n_intervals": 8, "block": 4,
        "velocity_convention": "sigma_descending", "foldable": True,
    },
}

# Same capabilities. Utterly different training stories. The runtime must not be able to tell.
PROVENANCE_A = {
    "training_method": "parallel_decoding_distillation",
    "teacher": "wan_va_50step", "solver": "flow_match", "dataset": "robotwin_2.0",
    "optimizer": "adamw", "coverage_gate_pass": True, "endpoint_rmse": 0.0041,
    "min_updates_per_head": 900, "paper": "arXiv:2501.00001",
}
PROVENANCE_B = {
    "training_method": "a_recipe_that_does_not_exist_yet",
    "teacher": None, "notes": "trained by a third party who told us nothing",
}


def _write(d: Path, execution: dict, provenance: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "instinctflash.json").write_text(json.dumps(
        {"instinctflash_schema": SCHEMA_VERSION, "execution": execution, "provenance": provenance},
        indent=2))
    (d / "config.json").write_text(json.dumps({"architectures": ["WanTransformer3DModel"]}))
    (d / "model.safetensors").write_bytes(b"\x00")     # presence is what the layout checks
    return d


def test_declaration_never_returns_provenance():
    print("\n=== 1. load_declaration() cannot hand the runtime a training fact ===")
    with tempfile.TemporaryDirectory() as td:
        d = _write(Path(td) / "a", EXECUTION, PROVENANCE_A)
        decl = load_declaration(d)
        blob = json.dumps({"model_id": decl.model_id, "backbone": decl.backbone,
                           "servable": decl.servable, "guidance": dict(decl.guidance),
                           "nfe": dict(decl.nfe), "extra": dict(decl.extra)})
        for key in ("parallel_decoding_distillation", "robotwin_2.0", "adamw", "arXiv"):
            check(key not in blob, f"execution declaration does not carry {key!r}")
        for key in FORBIDDEN_IN_EXECUTION:
            check(key not in dict(decl.extra), f"execution.extra does not carry {key!r}")
        check(provenance_of(d)["training_method"] == "parallel_decoding_distillation",
              "provenance_of() still gives tools the full record")


def test_forbidden_keys_are_refused_at_the_boundary():
    print("\n=== 2. a mis-stamped checkpoint fails loudly, not quietly ===")
    with tempfile.TemporaryDirectory() as td:
        bad = dict(EXECUTION, training_method="pdd")
        d = _write(Path(td) / "bad", bad, {})
        try:
            load_declaration(d)
            check(False, "a provenance key in `execution` is rejected")
        except RuntimeError as e:
            check("training_method" in str(e) and "provenance" in str(e),
                  "a provenance key in `execution` is rejected, and the error says where to move it")


def test_planning_is_invariant_to_provenance():
    print("\n=== 3. THE PLATFORM CLAIM: identical execution -> identical plan ===")
    from instinctflash.descriptors.package import from_pretrained
    with tempfile.TemporaryDirectory() as td:
        a = from_pretrained(_write(Path(td) / "a", EXECUTION, PROVENANCE_A))
        b = from_pretrained(_write(Path(td) / "b", EXECUTION, PROVENANCE_B))
        check(a.capabilities() == b.capabilities(),
              "capabilities are identical", f"{len(a.capabilities())} tokens")
        blob = " ".join(sorted(a.capabilities()))
        for word in ("pdd", "distill", "teacher", "dataset", "recipe", "adamw"):
            check(word not in blob.lower(), f"no capability token mentions {word!r}")

        from instinctflash.adapters.base import AdapterSpec
        from instinctflash.planners.planner import Optimizer
        spec = AdapterSpec(model_id=a.model_id, param_bytes=1, streams=(), phases=(), guidance={})
        opt = Optimizer(passes=[])
        pa = opt.compile(spec, capabilities=a.capabilities()).explain()
        pb = opt.compile(spec, capabilities=b.capabilities()).explain()
        check(pa == pb, "the two plans are byte-identical")


def test_a_pass_is_admitted_by_capability_not_by_recipe():
    print("\n=== 4. capability gating: declared, or the pass is skipped ===")
    from instinctflash.adapters.base import AdapterSpec
    from instinctflash.planners.planner import Optimizer, PassResult, Tier

    class NeedsFoldableHeads:
        name = "needs_foldable_heads"
        requires_capabilities = frozenset({"output_projection:foldable"})

        def evaluate(self, spec, deployment):
            return PassResult(name=self.name, applies=True, tier=Tier.BITEXACT, reason="declared")

    spec = AdapterSpec(model_id="m", param_bytes=1, streams=(), phases=(), guidance={})
    opt = Optimizer(passes=[NeedsFoldableHeads()])
    yes = opt.compile(spec, capabilities=frozenset({"output_projection:foldable"}))
    no = opt.compile(spec, capabilities=frozenset({"servable"}))
    check(yes.results[0].applies, "admitted when the capability is declared")
    check(not no.results[0].applies, "skipped when it is not")
    check("CAPABILITY decision" in no.results[0].reason,
          "and the reason says it is a capability decision")
    unfiltered = opt.compile(spec)
    check(unfiltered.results[0].applies,
          "capabilities=None does not filter -- every existing pass composes with every checkpoint")


def test_publishable_without_training_internals():
    print("\n=== 5. a checkpoint can be published with provenance stripped ===")
    from instinctflash.descriptors.package import publishability, validate_package
    with tempfile.TemporaryDirectory() as td:
        d = _write(Path(td) / "pub", EXECUTION, PROVENANCE_A)
        ok, findings = publishability(d)
        check(ok, "publishable: the runtime can serve it with provenance removed", str(findings))
        stripped = _write(Path(td) / "stripped", EXECUTION, {})
        rep = validate_package(stripped)
        check(rep.ok, "and a package with provenance={} validates", rep.explain().replace("\n", " | "))
        check(load_declaration(stripped).servable, "and is still servable")


def test_legacy_is_a_compatibility_layer_only():
    print("\n=== 6. delta.json still serves, and migrates ===")
    from instinctflash.descriptors.package import migrate_legacy, validate_package
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "legacy"
        d.mkdir()
        (d / "delta.json").write_text(json.dumps({
            "model_id": "legacy/m", "backbone": "wan_va", "n_intervals": 8, "block": 4,
            "coverage_gate_pass": True, "recipe": "pdd", "endpoint_rmse": 0.004}))
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\x00")
        decl = load_declaration(d)
        check(decl.legacy and decl.servable, "legacy checkpoint still loads and is servable")
        check(decl.output_projection is not None and decl.output_projection.n_intervals == 8,
              "and its capability survives the mapping")
        rep = validate_package(d)
        check(any("legacy" in n for n in rep.notes), "validate_package flags it as legacy")
        from instinctflash.descriptors.package import publishability
        ok, _ = publishability(d)
        check(not ok, "and it is NOT publishable -- one flat namespace cannot hide training keys")
        doc = migrate_legacy(d)
        check("recipe" not in doc["execution"] and doc["provenance"].get("recipe") == "pdd",
              "migrate_legacy puts the recipe in provenance, where the runtime cannot reach it")




def test_the_shipped_example_package_validates():
    """The example in examples/ is a real artifact, validated here rather than a fixture in the test.

    A layout documented in prose drifts from the layout the code accepts. This is the same reason
    tests/test_shipped_config.py derives from released.py instead of restating it.
    """
    print("\n=== 7. examples/checkpoint/wm-blockheads-2v4a validates ===")
    from instinctflash.descriptors.package import from_pretrained, publishability, validate_package
    d = ROOT / "examples" / "checkpoint" / "wm-blockheads-2v4a"
    check(d.is_dir(), "the example package exists")
    rep = validate_package(d)
    check(rep.ok, "it validates as a servable package", "; ".join(rep.problems + rep.missing))
    ok, findings = publishability(d)
    check(ok, "and is publishable with provenance stripped", str(findings))
    ck = from_pretrained(d)
    check(ck.model_id == "example-org/wm-blockheads-2v4a", "from_pretrained loads it", ck.model_id)
    caps = ck.capabilities()
    check("output_projection:foldable" in caps and "servable" in caps,
          "and its capabilities are the documented ones", str(sorted(caps)))
    readme = (d / "README.md").read_text()
    for tok in sorted(caps):
        check(tok in readme, f"the model card documents capability {tok}")




def training_recipe_decisions(source, relative_path):
    """Find recipe-dependent control flow, allowing architecture-specific execution.

    Importing DreamZero's action head or GR00T's adapter is family dispatch, not
    a training-recipe decision. The deprecated --pdd-heads CLI spelling is the
    sole compatibility allowance: it normalizes to the same block-heads path.
    Provenance-field reads and transitive training imports are checked separately
    in test_runtime_boundary, including within family-specific modules.
    """
    import ast
    import re
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    names = {"pdd", "dmd2", "lcm", "rcm", "scm"}
    roots = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            roots.append(node.test)
        elif isinstance(node, ast.comprehension):
            roots.extend(node.ifs)
        elif isinstance(node, ast.Match):
            roots.append(node.subject)
            for case in node.cases:
                roots.append(case.pattern)
                if case.guard:
                    roots.append(case.guard)

    def legacy_alias(node):
        if relative_path != "instinctflash/runtime/lingbot_worker.py":
            return False
        if isinstance(node, ast.Attribute):
            return node.attr == "pdd_heads" and isinstance(node.value, ast.Name) and node.value.id == "args"
        parent = parents.get(node)
        return (isinstance(node, ast.Constant) and node.value == "pdd_heads"
                and isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                and parent.func.id == "getattr" and len(parent.args) in (2, 3)
                and isinstance(parent.args[0], ast.Name) and parent.args[0].id == "args"
                and parent.args[1] is node)

    offenders = set()
    for root in roots:
        for node in ast.walk(root):
            value = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                     else node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else "")
            if (names & set(re.split(r"[^a-z0-9]+", value.lower()))
                    or value.lower() in {"training_method", "training_recipe"}) and not legacy_alias(node):
                offenders.add((node.lineno, value))
    return sorted(offenders)


def test_no_runtime_code_selects_a_training_recipe():
    print("\n=== 8. architecture dispatch is allowed; training-recipe decisions are not ===")
    dirs = ("runtime", "planners", "executors", "passes", "backends", "descriptors", "adapters")
    offenders = []
    for sub in dirs:
        for f in sorted((ROOT / "instinctflash" / sub).rglob("*.py")):
            if "__pycache__" in str(f):
                continue
            rel = f.relative_to(ROOT).as_posix()
            offenders.extend(f"{rel}:{line}: training-recipe decision on {name!r}"
                             for line, name in training_recipe_decisions(f.read_text(), rel))
    for o in offenders:
        print(f"       {o}")
    check(not offenders, "runtime/planning control flow does not select a training recipe",
          f"{len(offenders)} found")

    # and the one allowance is a real quarantine, not a loophole
    from instinctflash.runtime.state import manifests as M
    check(set(M.REGISTRY) <= {"lingbot-va", "cosmos3-edge"},
          "REGISTRY contains only supported models", str(sorted(M.REGISTRY)))
    unval = set(getattr(M, "UNVALIDATED_DESIGNS", {}))
    check(unval and not (unval & set(M.REGISTRY)),
          "unvalidated sketches are segregated and never in REGISTRY", str(sorted(unval)))


def test_recipe_decision_check_detects_real_branches_and_scopes_the_cli_alias():
    path = "instinctflash/runtime/dreamzero_fp8.py"
    for source in ('if recipe == "pdd":\n    serve()\n',
                   'selected = a if training_method == method else b\n',
                   'if checkpoint.training_recipe:\n    serve()\n',
                   'selected = [x for x in values if x == "dmd2"]\n'):
        check(bool(training_recipe_decisions(source, path)), "a training-recipe branch is detected")
    check(not training_recipe_decisions('if backbone == "dreamzero":\n    install_dreamzero_fp8(head)\n', path),
          "family-specific execution dispatch does not imply training provenance")
    alias = 'if getattr(args, "pdd_heads", None):\n    block_heads = args.pdd_heads\n'
    check(not training_recipe_decisions(alias, "instinctflash/runtime/lingbot_worker.py"),
          "the deprecated CLI alias remains a narrowly scoped compatibility spelling")
    check(bool(training_recipe_decisions(alias, path)),
          "the CLI compatibility exception does not apply in model execution code")


def main() -> int:
    test_declaration_never_returns_provenance()
    test_forbidden_keys_are_refused_at_the_boundary()
    test_planning_is_invariant_to_provenance()
    test_a_pass_is_admitted_by_capability_not_by_recipe()
    test_publishable_without_training_internals()
    test_legacy_is_a_compatibility_layer_only()
    test_the_shipped_example_package_validates()
    test_no_runtime_code_selects_a_training_recipe()
    test_recipe_decision_check_detects_real_branches_and_scopes_the_cli_alias()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: one runtime, many checkpoints, and planning cannot see the recipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
