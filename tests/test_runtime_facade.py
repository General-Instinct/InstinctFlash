#!/usr/bin/env python3
"""The public facade: one handle, and transport is not part of the model abstraction.

No GPU, no weights. Everything here is about the SHAPE of the API and the ORDER of the load, both of
which are the product. The real 10.2 GB path is exercised by `tools/lingbot_end_to_end.py`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


def _pkg(d: Path, backbone="tiny-wam", servable=True, extra=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    ex = {"model_id": "example-org/x", "backbone": backbone, "servable": servable,
          "guidance": {"action": "positive_only"}, "nfe": {"action": 2}}
    ex.update(extra or {})
    (d / "instinctflash.json").write_text(json.dumps(
        {"instinctflash_schema": 1, "execution": ex,
         "provenance": {"training_method": "secret", "teacher": "also secret"}}))
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    return d


def test_public_api_is_small():
    print("\n=== 1. the public API is five names, not eighteen ===")
    import instinctflash
    for n in ("Runtime", "from_pretrained", "describe"):
        check(hasattr(instinctflash, n), f"instinctflash.{n} exists")
    check(instinctflash.__all__[0] == "Runtime", "Runtime leads __all__", instinctflash.__all__[0])
    src = (ROOT / "instinctflash" / "runtime" / "facade.py").read_text()
    for word in ("socket", "websocket", "port", "subprocess"):
        # transport may be MENTIONED in prose; it must not be in the public signatures
        sigs = [ln for ln in src.splitlines()
                if ln.strip().startswith("def ") or ln.strip().startswith("    def ")]
        check(not any(word in ln for ln in sigs), f"no public signature mentions {word!r}")


def test_describe_reads_no_provenance_and_no_weights():
    print("\n=== 2. describe() returns execution facts only ===")
    from instinctflash import describe
    with tempfile.TemporaryDirectory() as td:
        d = _pkg(Path(td) / "p")
        got = describe(d)
        blob = json.dumps(got)
        check("secret" not in blob, "no provenance value leaks into describe()")
        check(got["has_provenance"] is True, "but it reports that provenance EXISTS")
        check(got["servable"] and got["backbone"] == "tiny-wam", "and returns the execution facts")


def test_unknown_backbone_teaches():
    print("\n=== 3. an unregistered backbone produces an error that teaches ===")
    from instinctflash import Runtime
    from instinctflash.runtime.facade import UnknownBackboneError
    with tempfile.TemporaryDirectory() as td:
        d = _pkg(Path(td) / "p", backbone="not-registered-anywhere")
        try:
            Runtime.from_pretrained(d)
            check(False, "raises for an unknown backbone")
        except UnknownBackboneError as e:
            m = str(e)
            check("not-registered-anywhere" in m, "names the declared backbone")
            check("Registered backbones:" in m, "lists what IS registered")
            check("instinctflash.register(" in m, "shows the exact fix")
            check("examples/tiny_wam/adapter.py" in m, "points at a worked example")


def test_unservable_is_refused():
    print("\n=== 4. servable=false is refused, without asking why ===")
    from instinctflash import Runtime
    with tempfile.TemporaryDirectory() as td:
        d = _pkg(Path(td) / "p", servable=False)
        try:
            Runtime.from_pretrained(d)
            check(False, "refuses an unservable checkpoint")
        except RuntimeError as e:
            check("servable" in str(e).lower(), "refuses it, citing servable")
            check("provenance" in str(e).lower(), "and says the reason lives in provenance")


def test_placement_is_a_deployment_choice_not_a_model_property():
    print("\n=== 5. placement is chosen, not declared ===")
    from instinctflash.runtime.execution import can_host_in_process, choose_backend
    from instinctflash import load
    ok, why = can_host_in_process(load('wan_va'))
    print(f"  this interpreter: {why}")
    decl = (ROOT / "instinctflash" / "descriptors" / "checkpoint.py").read_text()
    for w in ("websocket", "socket", "worker", "in_process", "placement"):
        check(w not in decl, f"the declaration schema has no notion of {w!r}")
    check(callable(choose_backend), "the runtime chooses placement at load time")


def test_no_fast_quality_presets():
    print("\n=== 6. no Fast/Quality preset table lives in the runtime ===")
    src = (ROOT / "instinctflash" / "runtime" / "facade.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    check("operating_point" not in code.split('"""')[-1],
          "no operating_point parameter in the implementation")
    check("nfe" in src, "nfe override IS available -- an explicit override of a declared field")
    import inspect
    from instinctflash import Runtime
    params = inspect.signature(Runtime.from_pretrained).parameters
    check("operating_point" not in params, "from_pretrained has no operating_point argument")
    check("nfe" in params, "from_pretrained takes nfe=", str(list(params)))


def test_composed_tree_never_lands_inside_the_package():
    print("\n=== 7. the composed tree is a cache artifact, not a package member ===")
    # REGRESSION. materialize() used to write `<pkg>/.instinctflash_composed/`. A package directory can
    # be a shared Hugging Face snapshot, and `hf upload` of one that had been loaded once published
    # a SECOND 10 GB copy of the transformer -- the symlinks resolved to real bytes on the way up.
    import os
    from instinctflash.adapters.lingbot_va import LingBotVA
    with tempfile.TemporaryDirectory() as td:
        pkg = _pkg(Path(td) / "pkg", backbone="wan_va")
        base = Path(td) / "base"
        for comp in LingBotVA.FROZEN_COMPONENTS:
            (base / comp).mkdir(parents=True, exist_ok=True)
        cache = Path(td) / "cache"
        os.environ["IFL_CACHE"] = str(cache)
        os.environ["LINGBOT_CKPT"] = str(base)
        try:
            class _Ck:
                path = str(pkg)
                model_id = "example-org/x"
                class execution:                       # noqa: N801
                    extra = {"base_weights": str(base)}
            composed = Path(LingBotVA.materialize(_Ck()))
            # idempotent: loading twice must not raise on the links the first call created
            again = Path(LingBotVA.materialize(_Ck()))
        finally:
            os.environ.pop("IFL_CACHE", None)
            os.environ.pop("LINGBOT_CKPT", None)

        check(not (pkg / ".instinctflash_composed").exists(),
              "nothing was written inside the package")
        check(cache in composed.parents, "the composed tree lives under the cache", str(composed))
        check((composed / "transformer" / "config.json").is_symlink(),
              "the trainable side is symlinked, not copied")
        check(set(p.name for p in pkg.iterdir()) ==
              {"instinctflash.json", "config.json", "model.safetensors"},
              "the package directory is byte-for-byte what it was before loading")
        check(again == composed, "materialize() is idempotent")


def main() -> int:
    test_public_api_is_small()
    test_describe_reads_no_provenance_and_no_weights()
    test_unknown_backbone_teaches()
    test_unservable_is_refused()
    test_placement_is_a_deployment_choice_not_a_model_property()
    test_no_fast_quality_presets()
    test_composed_tree_never_lands_inside_the_package()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: one handle, a teaching error, no presets, and no transport in the API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
