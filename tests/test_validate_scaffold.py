#!/usr/bin/env python3
"""`--validate.scaffold`: the one command from "training output" to "a package validate can judge".

The contract under test, per the design that shipped it:

  * scaffold-from-base copies the base's built-in declaration and goes field by field:
    INFERRED facts carry their evidence, INHERITED facts say so, and everything the checkpoint
    does not prove is written as the literal "FILL_ME" — never a guess. The wan_va geometry
    keys are the canary: inheriting the base's robotwin cameras onto a fine-tune is exactly the
    silent-wrong-conditioning failure the adapter's declare-or-fail rule exists to prevent.
  * `auto` fingerprints the checkpoint's own config.json; ambiguity is refused with the
    candidates listed, never resolved by picking one.
  * an existing declaration is never overwritten without --validate.force=true — the scaffold
    prints the full would-be document and a field diff instead.
  * the follow-up validate (and every later plain validate) flags each FILL_ME as a PROBLEM
    and exits non-zero until the last sentinel is replaced; a filled scaffold round-trips to
    exit 0.

No GPU, no torch, no weights — fake checkpoints in tmpdirs, the same fixture pattern as
test_checkpoint_platform.py.
"""
from __future__ import annotations

import contextlib
import io
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


def run(argv):
    from instinctflash.cli import main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = main(argv)
        except SystemExit as e:                                  # argparse
            rc = int(e.code or 0)
    return rc, buf.getvalue()


def wan_va_fixture(td: Path) -> Path:
    """A wan_va fine-tune the way a training run leaves it: transformer config + weights, no
    declaration. Flat package layout (the transformer contents at the root)."""
    d = td / "wanva-ft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"_class_name": "WanTransformer3DModel", "action_dim": 30, "num_layers": 30}))
    (d / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00" * 4096)
    return d


def pi05_fixture(td: Path) -> Path:
    """A lerobot pi05 fine-tune: config.json states type, input_features and the schedule."""
    d = td / "pi05-ft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "type": "pi05", "num_inference_steps": 10,
        "input_features": {
            "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.state": {"type": "STATE", "shape": [8]},
        }}))
    (d / "model.safetensors").write_bytes(b"\x00" * 2048)
    (d / "policy_preprocessor.json").write_text("{}")
    return d


def test_scaffold_from_named_base_infers_proves_and_refuses_to_guess():
    print("\n=== 1. scaffold from a named base: inherited / inferred(+evidence) / FILL_ME ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_fixture(Path(td))
        rc, out = run(["validate", str(d),
                       "--validate.scaffold=robbyant/lingbot-va-posttrain-robotwin"])
        f = d / "instinctflash.json"
        check(f.is_file(), "the declaration was written")
        doc = json.loads(f.read_text())
        ex = doc["execution"]
        check(ex["backbone"] == "wan_va", "backbone inferred", ex["backbone"])
        check("WanTransformer3DModel" in out and "action_dim" in out,
              "with the config fingerprint quoted as evidence")
        check(ex["param_bytes"] == 4096, "param_bytes measured from the weight file",
              str(ex["param_bytes"]))
        check(ex["guidance"] == {"video": "cfg", "action": "positive_only"},
              "guidance inherited from the base")
        for key in ("obs_cam_keys", "height", "width", "env_type"):
            check(ex[key] == "FILL_ME", f"{key} is FILL_ME, never the base's robotwin value")
        check("inference depth for backbone 'wan_va'" in out,
              "the per-family inference depth is stated in the output")
        check("inferred" in out and "inherited" in out and "FILL_ME" in out,
              "the report names all three field classes")
        reasons = doc["provenance"]["scaffold"]["fill_me"]
        check(set(reasons) == {"obs_cam_keys", "height", "width", "env_type"},
              "each FILL_ME carries a one-line explanation (in provenance, for humans)")
        check(rc == 1, "and the follow-up validate exits non-zero on the sentinels", str(rc))
        check("PROBLEM  execution.obs_cam_keys" in out, "flagging them as PROBLEM lines")


def test_auto_detects_the_base_per_family():
    print("\n=== 2. --validate.scaffold=auto fingerprints the checkpoint itself ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_fixture(Path(td))
        rc, out = run(["validate", str(d), "--validate.scaffold=auto"])
        check("auto-detected base: robbyant/lingbot-va-posttrain-robotwin" in out,
              "a wan_va fine-tune matches the LingBot-VA base")
    with tempfile.TemporaryDirectory() as td:
        d = pi05_fixture(Path(td))
        rc, out = run(["validate", str(d), "--validate.scaffold=auto"])
        check("auto-detected base: lerobot/pi05_base" in out,
              "a pi05 fine-tune matches lerobot/pi05_base")
        ex = json.loads((d / "instinctflash.json").read_text())["execution"]
        check(ex["obs_features"] == {"observation.images.image": [3, 256, 256],
                                     "observation.images.image2": [3, 256, 256],
                                     "observation.state": [8]},
              "obs_features read from the checkpoint's own input_features")
        check(ex["nfe"] == {"prefix": 1, "action": 10},
              "the denoise schedule read from num_inference_steps")
        check(ex["base_weights"] == str(d.resolve()),
              "the package itself is the loadable policy, so base_weights points at it")
        check(rc == 0, "pi05 inference is deep enough that nothing is left to fill", str(rc))


def test_ambiguity_is_refused_with_candidates_listed():
    print("\n=== 3. an ambiguous fingerprint lists the candidates and refuses to pick ===")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "cosmos-ft"
        d.mkdir()
        # cosmos3_omni whose text tower identifies neither Edge nor Nano
        (d / "config.json").write_text(json.dumps(
            {"model_type": "cosmos3_omni", "text_config": {"model_type": "mystery"}}))
        rc, out = run(["validate", str(d), "--validate.scaffold=auto"])
        check(rc == 2, "refusal is a config error, exit 2", str(rc))
        check("ambiguous" in out and "refuses to pick" in out, "and says so")
        check("Cosmos3-Edge-Policy-DROID" in out and "Cosmos3-Nano-Policy-DROID" in out,
              "listing both candidates")
        check(not (d / "instinctflash.json").exists(), "and writes nothing")


def test_wrong_or_unknown_base_is_refused():
    print("\n=== 4. a base that contradicts the checkpoint, or names nothing, is refused ===")
    with tempfile.TemporaryDirectory() as td:
        d = pi05_fixture(Path(td))
        rc, out = run(["validate", str(d),
                       "--validate.scaffold=robbyant/lingbot-va-posttrain-robotwin"])
        check(rc == 2 and "does not look like a wan_va checkpoint" in out,
              "a pi05 checkpoint cannot be scaffolded as wan_va", str(rc))
        rc, out = run(["validate", str(d), "--validate.scaffold=nobody/nothing"])
        check(rc == 2 and "lerobot/pi05_base" in out,
              "an unknown base id is refused with the known bases listed", str(rc))
        check(not (d / "instinctflash.json").exists(), "neither refusal writes anything")


def test_existing_declaration_is_never_overwritten_without_force():
    print("\n=== 5. the no-overwrite guard: print what would change, write only under force ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_fixture(Path(td))
        rc, _ = run(["validate", str(d), "--validate.scaffold=auto"])
        # fill the geometry so the existing file differs meaningfully from a re-scaffold
        doc = json.loads((d / "instinctflash.json").read_text())
        doc["execution"].update({"obs_cam_keys": ["observation.images.ego"],
                                 "height": 224, "width": 384, "env_type": "none"})
        (d / "instinctflash.json").write_text(json.dumps(doc, indent=2))
        before = (d / "instinctflash.json").read_bytes()

        rc, out = run(["validate", str(d), "--validate.scaffold=auto"])
        check((d / "instinctflash.json").read_bytes() == before, "the file is untouched")
        check("NOT WRITTEN" in out and "--validate.force=true" in out,
              "the refusal names the flag that overrides it")
        check("~ execution.obs_cam_keys" in out, "and shows the field-level diff")
        check("the full document the scaffold would write" in out, "plus the would-be document")
        check(rc != 0, "refusing to do the asked work is not a success exit", str(rc))

        rc, out = run(["validate", str(d), "--validate.scaffold=auto", "--validate.force=true"])
        check((d / "instinctflash.json").read_bytes() != before, "force overwrites")
        check("overwriting" in out, "and says it is overwriting")


def test_fill_me_is_flagged_by_every_later_plain_validate():
    print("\n=== 6. plain validate keeps failing until the last FILL_ME is replaced ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_fixture(Path(td))
        run(["validate", str(d), "--validate.scaffold=auto"])
        rc, out = run(["validate", str(d)])                       # no scaffold flag this time
        check(rc == 1, "a later plain validate still exits 1", str(rc))
        check(out.count('is "FILL_ME"') == 4, "flagging all four sentinels",
              str(out.count('is "FILL_ME"')))
        check("copy obs_cam_keys from your wan_va training config" in out,
              "each with the scaffold's explanation of what belongs there")


def test_round_trip_scaffold_fill_validate_pass():
    print("\n=== 7. round trip: scaffold + filled fields -> validate PASS ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_fixture(Path(td))
        run(["validate", str(d), "--validate.scaffold=auto"])
        doc = json.loads((d / "instinctflash.json").read_text())
        doc["execution"].update({"obs_cam_keys": ["observation.images.ego",
                                                  "observation.images.left_wrist"],
                                 "height": 224, "width": 384, "env_type": "none"})
        (d / "instinctflash.json").write_text(json.dumps(doc, indent=2))
        rc, out = run(["validate", str(d)])
        check(rc == 0, "the filled scaffold validates clean", str(rc) + " " + out)
        check("servable package: YES" in out, "as a servable package")
        # and the declaration actually resolves through the adapter's own geometry rule
        from instinctflash.descriptors.checkpoint import load_declaration
        from instinctflash.adapters.lingbot_va import resolve_observation_geometry
        geom, source = resolve_observation_geometry(load_declaration(d), environ={})
        check(geom["height"] == 224 and geom["obs_cam_keys"][0] == "observation.images.ego",
              "and the wan_va adapter reads the filled geometry from the declaration")
        check("declaration" in source, "as a declaration source", source)


def test_adapters_treat_fill_me_as_undeclared():
    print("\n=== 8. FILL_ME is a sentinel to the adapters, never a value ===")
    from types import SimpleNamespace

    from instinctflash.adapters.lingbot_va import resolve_observation_geometry
    ex = SimpleNamespace(extra={"obs_cam_keys": "FILL_ME", "height": "FILL_ME",
                                "width": "FILL_ME", "env_type": "FILL_ME"},
                         model_id="x/ft", source="pkg/instinctflash.json")
    try:
        resolve_observation_geometry(ex, environ={})
    except RuntimeError as e:
        check("no observation geometry" in str(e) and "obs_cam_keys" in str(e),
              "wan_va raises its loud declare-or-IFL_CFG error, not int('FILL_ME')")
    else:
        check(False, "wan_va must refuse FILL_ME geometry")

    sys.path.insert(0, str(ROOT / "examples" / "pi05_vla"))
    from pi05_iwm.adapter import Pi05Adapter
    ckpt = SimpleNamespace(path="pkg", execution=SimpleNamespace(
        extra={"obs_features": "FILL_ME"}, model_id="x/pi05-ft"))
    try:
        Pi05Adapter().observation_contract(ckpt)
    except RuntimeError as e:
        check("obs_features" in str(e), "pi05 raises its declare-obs_features error")
    else:
        check(False, "pi05 must refuse FILL_ME obs_features")


def main_() -> int:
    test_scaffold_from_named_base_infers_proves_and_refuses_to_guess()
    test_auto_detects_the_base_per_family()
    test_ambiguity_is_refused_with_candidates_listed()
    test_wrong_or_unknown_base_is_refused()
    test_existing_declaration_is_never_overwritten_without_force()
    test_fill_me_is_flagged_by_every_later_plain_validate()
    test_round_trip_scaffold_fill_validate_pass()
    test_adapters_treat_fill_me_as_undeclared()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the scaffold proves what it can, refuses what it cannot, and validate "
          "polices the difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
