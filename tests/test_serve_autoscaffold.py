#!/usr/bin/env python3
"""`instinctflash serve <local-dir>` with no declaration: the one-command path to a server.

The contract under test, per the design that shipped it:

  * a local directory with NO declaration is scaffolded inline — the same writer as
    `validate --validate.scaffold=auto`, announced field by field with the evidence, and the
    file is WRITTEN so the second serve is a plain fast path.
  * zero FILL_ME (pi05-class deep inference) -> serve continues straight into its normal
    preflight in the same command. One command, zero ceremony.
  * FILL_ME remaining -> serve STOPS before any download or load, with exactly the missing
    fields, one line each of why + where the value comes from, and the exact rerun. The same
    stop fires on EVERY later serve until the last sentinel is replaced — otherwise the second
    run would download the base's frozen stack and only then hit the adapter's refusal.
  * an unmerged LoRA adapter hits `unmerged_adapter_problem`'s wall — the exact merge command —
    BEFORE any preflight or load, declaration or not.
  * a directory that already carries a complete declaration is untouched: the dry_run output is
    byte-identical to the preflight alone (regression pin: this feature changes nothing for
    declared packages).

No GPU, no torch, no weights — fake checkpoints in tmpdirs, the same fixtures as
test_validate_scaffold.py. The loaded end of the same path (a real fine-tune dir to a finite
action in one command) is the GPU e2e, run out-of-tree.
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


def _register_pi05_adapter():
    """The pi05 adapter is an external plugin (examples/pi05_vla); serve's preflight resolves
    the backbone through the registry, so the source tree stands in for `pip install`."""
    import instinctflash
    sys.path.insert(0, str(ROOT / "examples" / "pi05_vla"))
    from pi05_iwm.adapter import Pi05Adapter
    try:
        instinctflash.register("pi05", Pi05Adapter)
    except Exception:                                            # noqa: BLE001 - already registered
        pass


def wan_va_fixture(td: Path) -> Path:
    d = td / "wanva-ft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"_class_name": "WanTransformer3DModel", "action_dim": 30, "num_layers": 30}))
    (d / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00" * 4096)
    return d


def pi05_fixture(td: Path) -> Path:
    d = td / "pi05-ft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "type": "pi05", "num_inference_steps": 10,
        "input_features": {
            "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.state": {"type": "STATE", "shape": [8]},
        }}))
    (d / "model.safetensors").write_bytes(b"\x00" * 2048)
    (d / "policy_preprocessor.json").write_text("{}")
    return d


def test_pi05_dir_scaffolds_and_proceeds_in_one_command():
    print("\n=== 1. no-declaration pi05 dir: scaffold inline, announce, continue to preflight ===")
    _register_pi05_adapter()
    with tempfile.TemporaryDirectory() as td:
        d = pi05_fixture(Path(td))
        rc, out = run(["serve", str(d), "--serve.dry_run=true"])
        check(rc == 0, "one command, exit 0 — the deep-inference family proceeds", str(rc))
        check("no declaration in" in out and "scaffolding one" in out,
              "the run announces what it is about to do")
        check("auto-detected base: lerobot/pi05_base" in out, "and names the detected base")
        check('config.json: type == "pi05"' in out, "with the fingerprint quoted as evidence")
        check("inferred" in out and "inherited" in out, "per-field provenance is printed")
        check("config.json num_inference_steps" in out and "measured: model.safetensors" in out,
              "with the evidence one-liners for the inferred facts")
        check(f"wrote {d / 'instinctflash.json'}" in out, "the write is announced")
        check("serve preflight" in out and "declaration-only" in out,
              "and the same command continues into the normal preflight")
        check("plan tier" in out or "InstinctFlash plan" in out, "through to the plan")
        ex = json.loads((d / "instinctflash.json").read_text())["execution"]
        check(ex["obs_features"] == {"observation.images.image": [3, 256, 256],
                                     "observation.state": [8]},
              "the written declaration carries the checkpoint's own obs_features")
        check("FILL_ME" not in out, "nothing was left to fill")


def test_wan_va_dir_stops_with_the_geometry_asks():
    print("\n=== 2. no-declaration wan_va dir: scaffold, then STOP with the four geometry asks ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_fixture(Path(td))
        rc, out = run(["serve", str(d), "--serve.dry_run=true"])
        check(rc == 1, "serve exits 1 on the sentinels", str(rc))
        check("SERVE STOPPED" in out and "before any download or load" in out,
              "and says it stopped before any download")
        for key in ("obs_cam_keys", "height", "width", "env_type"):
            check(f"execution.{key}" in out, f"asking for {key}")
        check("copy obs_cam_keys from your wan_va training config" in out,
              "each ask says where the value comes from")
        check(f"instinctflash serve {d}" in out, "and quotes the exact rerun")
        check((d / "instinctflash.json").is_file(),
              "the declaration WAS written, so the user edits a file, not a template")
        check("serve preflight" not in out, "no preflight ran — the stop is the whole answer")

        # without dry_run the same wall holds — this is the path that guards the download
        rc, out = run(["serve", str(d)])
        check(rc == 1 and "SERVE STOPPED" in out,
              "a non-dry serve stops at the same wall", str(rc))


def test_fill_me_stops_every_later_serve_until_filled():
    print("\n=== 3. the stop persists on rerun (file untouched), and filling releases it ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_fixture(Path(td))
        run(["serve", str(d), "--serve.dry_run=true"])
        before = (d / "instinctflash.json").read_bytes()
        rc, out = run(["serve", str(d), "--serve.dry_run=true"])
        check(rc == 1, "the second serve still stops", str(rc))
        check("scaffolding one" not in out, "without re-scaffolding")
        check("still carries" in out, "and says the existing declaration still carries FILL_ME")
        check((d / "instinctflash.json").read_bytes() == before, "the file is untouched")

        doc = json.loads((d / "instinctflash.json").read_text())
        doc["execution"].update({"obs_cam_keys": ["observation.images.ego"],
                                 "height": 224, "width": 384, "env_type": "none"})
        (d / "instinctflash.json").write_text(json.dumps(doc, indent=2))
        rc, out = run(["serve", str(d), "--serve.dry_run=true"])
        check(rc == 0, "the filled declaration serves (dry_run preflight passes)", str(rc))
        check("scaffold" not in out.lower(), "as a plain fast path — no scaffold vocabulary")


def test_pi05_second_serve_is_the_plain_fast_path():
    print("\n=== 4. second serve of the scaffolded pi05 dir: plain fast path, byte-stable ===")
    _register_pi05_adapter()
    with tempfile.TemporaryDirectory() as td:
        d = pi05_fixture(Path(td))
        rc1, out1 = run(["serve", str(d), "--serve.dry_run=true"])
        written = (d / "instinctflash.json").read_bytes()
        rc2, out2 = run(["serve", str(d), "--serve.dry_run=true"])
        check(rc1 == 0 and rc2 == 0, "both runs exit 0", f"{rc1},{rc2}")
        check("scaffolding one" not in out2 and "wrote" not in out2,
              "the second run does not scaffold or write")
        check((d / "instinctflash.json").read_bytes() == written, "the declaration is untouched")
        check(out1.endswith(out2),
              "the first run was exactly scaffold-report + the second run's preflight")


def test_existing_declaration_means_zero_behavior_change():
    print("\n=== 5. regression pin: a declared package preflights byte-identically ===")
    import instinctflash
    from instinctflash.cli import _serve_preflight
    from instinctflash.cli_config import RuntimeConfig
    from examples.tiny_wam.adapter import TinyWAMAdapter
    try:
        instinctflash.register("tiny-wam", TinyWAMAdapter)
    except Exception:                                            # noqa: BLE001 - already registered
        pass
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "p"
        pkg.mkdir()
        (pkg / "config.json").write_text("{}")
        (pkg / "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "example-org/x", "backbone": "tiny-wam", "servable": True,
                          "guidance": {"action": "positive_only"}, "nfe": {"action": 2},
                          "base_weights": "upstream/none"},
        }))
        (pkg / "model.safetensors").write_bytes(b"\x00")
        before = (pkg / "instinctflash.json").read_bytes()
        rc, out = run(["serve", str(pkg), "--serve.dry_run=true"])
        check(rc == 0, "dry_run exits 0", str(rc))
        _, preflight_text = _serve_preflight(str(pkg), RuntimeConfig())
        check(out == preflight_text.rstrip() + "\n",
              "the output is the preflight text alone, byte-for-byte — plan included")
        check("scaffold" not in out.lower(), "no scaffold vocabulary")
        check((pkg / "instinctflash.json").read_bytes() == before, "the declaration untouched")


def test_unmerged_adapter_hits_the_merge_command_wall():
    print("\n=== 6. an unmerged LoRA adapter dir: the merge-command wall, before anything ===")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pi05-lora-ft"
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "r": 8, "lora_alpha": 16,
            "base_model_name_or_path": "lerobot/pi05_base"}))
        (d / "adapter_model.safetensors").write_bytes(b"\x00" * 8192)
        (d / "config.json").write_text(json.dumps({"type": "pi05"}))
        rc, out = run(["serve", str(d), "--serve.dry_run=true"])
        check(rc == 2, "serve refuses (config error, exit 2)", str(rc))
        check("unmerged LoRA adapter" in out, "naming the layout")
        check("merge_and_unload" in out and f"'{d}'" in out,
              "with the exact merge command over this adapter path")
        check("get_policy_class('pi05')" in out and "'lerobot/pi05_base'" in out,
              "loader from the checkpoint's own config type, base from adapter_config")
        check(not (d / "instinctflash.json").exists(), "and writes nothing")

        # a garbage declaration written before the wall existed does not launder it
        (d / "instinctflash.json").write_text(json.dumps({
            "instinctflash_schema": 1,
            "execution": {"model_id": "x/lora-ft", "backbone": "pi05", "servable": True,
                          "base_weights": "lerobot/pi05_base", "param_bytes": 14467165872}}))
        rc, out = run(["serve", str(d), "--serve.dry_run=true"])
        check(rc == 2 and "unmerged LoRA adapter" in out,
              "a pre-existing declaration does not launder the adapter", str(rc))


def test_wrong_level_dir_stops_with_the_servable_path():
    print("\n=== 6b. serve at a lerobot-train run root: pointed at the real checkpoint ===")
    # The first REAL lerobot-train fine-tune served (2026-08-28): the user's natural argument is
    # the --output_dir they passed to lerobot-train, but the servable model is two levels down.
    # The old refusal ("no built-in declaration matches ... pass --validate.scaffold=<base>") was
    # true and useless; the wall must name the exact path and the rerun.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "pi05_libero_sft"
        step = root / "checkpoints" / "000400"
        pm = step / "pretrained_model"
        pm.mkdir(parents=True)
        (pm / "config.json").write_text(json.dumps({"type": "pi05"}))
        (pm / "model.safetensors").write_bytes(b"\x00" * 2048)
        (pm / "policy_preprocessor.json").write_text("{}")
        (step / "training_state").mkdir()
        (root / "checkpoints" / "last").symlink_to(step, target_is_directory=True)

        rc, out = run(["serve", str(root), "--serve.smoke=true"])
        check(rc == 2, "the run root stops as a config error", str(rc))
        check("lerobot-train output" in out, "naming the layout it recognized")
        check(f"instinctflash serve {pm.resolve()}" in out,
              "with the exact servable path as the rerun command", out.strip()[-200:])
        check(not (root / "instinctflash.json").exists(), "and writes nothing at the wrong level")

        rc2, out2 = run(["serve", str(step), "--serve.smoke=true"])
        check(rc2 == 2 and "pretrained_model" in out2,
              "the intermediate step dir points one level down too", str(rc2))


def test_json_mode_carries_the_stop_structurally():
    print("\n=== 7. --output.format=json: the stop is structured, not just prose ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_fixture(Path(td))
        rc, out = run(["serve", str(d), "--serve.dry_run=true", "--output.format=json"])
        payload = json.loads(out)
        check(rc == 1 and payload["ok"] is False, "exit 1, ok false", str(rc))
        check(sorted(payload["result"]["fill_me"]) == [
            "execution.env_type", "execution.height", "execution.obs_cam_keys",
            "execution.width"], "result.fill_me lists exactly the four asks")
        check(payload["result"]["scaffold"]["base"] == "robbyant/lingbot-va-posttrain-robotwin",
              "and the scaffold record rides along")


def main_() -> int:
    test_pi05_dir_scaffolds_and_proceeds_in_one_command()
    test_wan_va_dir_stops_with_the_geometry_asks()
    test_fill_me_stops_every_later_serve_until_filled()
    test_pi05_second_serve_is_the_plain_fast_path()
    test_existing_declaration_means_zero_behavior_change()
    test_unmerged_adapter_hits_the_merge_command_wall()
    test_wrong_level_dir_stops_with_the_servable_path()
    test_json_mode_carries_the_stop_structurally()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: one command from a bare training output to a serving preflight — announced, "
          "written, and stopped loudly exactly where proof runs out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
