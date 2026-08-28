#!/usr/bin/env python3
"""Evidence-based scaffold deepening: read what the checkpoint dir itself states, never guess.

The eight-family one-command UX campaign measured where `instinctflash serve <dir>` stops for
each family and shortened the distance ONLY where the checkpoint carries its own evidence:

  * wan_va — a training-config artifact shipped in the package (the upstream EasyDict format,
    e.g. va_fans_train_cfg.py) is parsed STATICALLY (ast literals, never imported/executed) for
    the four geometry keys and the two denoise-step counts. Conflicting or non-literal values
    stay FILL_ME: evidence that disagrees with itself proves nothing.
  * cosmos3_policy — checkpoint.json's policy block is upstream's own serving artifact; a key
    stated there (Edge: domain_name, action_chunk_size) is inferred with the citation, and a
    value that differs from the DROID base's measured protocol says so. Nano ships an empty
    checkpoint.json and honestly keeps all five FILL_ME.
  * lingbot_vla / _v2 — lingbotvla_cli.yaml (the file the fingerprint already trusts) provides
    norm_stats from data.norm_stats_file; `robot` is written only when data.data_name is
    corroborated by the stats filename (robotwin -> robotwin_50.json). The V2 release's
    data_name 'multi' is a training-data mix, not a serving profile, and is refused with the
    yaml value quoted.
  * groot_n17 — the native Isaac-GR00T release layout (model_type "Gr00tN1d7") fingerprints;
    embodiment_tag falls back to the sole statistics.json key, and a multi-head statistics.json
    stays FILL_ME with the checkpoint's own candidates listed.
  * unknown-backbone UX — a first-party family stops with the exact `pip install <package>`;
    an installed-but-broken adapter entry point stops with the import error instead of sending
    the user in an install circle; serve's scaffold announcement survives a failing preflight.

No GPU, no torch, no weights — fake checkpoints in tmpdirs, same fixtures style as
test_validate_scaffold.py.
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


def scaffold(d: Path):
    from instinctflash.descriptors.scaffold import run_scaffold
    result, text, wrote = run_scaffold(d, "auto")
    doc = json.loads((d / "instinctflash.json").read_text())
    return result, text, doc["execution"]


def wan_va_dir(td: Path, cfg_py: "str | None" = None, extra_py: "str | None" = None) -> Path:
    d = td / "wanva-ft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"_class_name": "WanTransformer3DModel", "action_dim": 30, "num_layers": 30}))
    (d / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00" * 4096)
    if cfg_py is not None:
        (d / "va_test_train_cfg.py").write_text(cfg_py)
    if extra_py is not None:
        (d / "second_cfg.py").write_text(extra_py)
    return d


FANS_STYLE_CFG = """
import os
from easydict import EasyDict
from .shared_config import va_shared_cfg

cfg = EasyDict(__name__='Config: VA test')
cfg.update(va_shared_cfg)
cfg.env_type = 'none'
cfg.height = 224
cfg.width = 384
cfg.obs_cam_keys = [
    'observation.images.ego', 'observation.images.left_wrist',
    'observation.images.right_wrist'
]
cfg.num_inference_steps = 25
cfg.video_exec_step = -1
cfg.action_num_inference_steps = 50
"""


def test_wan_va_training_cfg_artifact_fills_geometry_and_schedule():
    print("\n=== 1. wan_va: a shipped training-config artifact proves geometry + schedule ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_dir(Path(td), cfg_py=FANS_STYLE_CFG)
        _, text, ex = scaffold(d)
        check(ex["obs_cam_keys"] == ["observation.images.ego", "observation.images.left_wrist",
                                     "observation.images.right_wrist"],
              "obs_cam_keys read from the artifact")
        check(ex["height"] == 224 and ex["width"] == 384 and ex["env_type"] == "none",
              "height/width/env_type read from the artifact")
        check(ex["nfe"] == {"video": 25, "action": 50},
              "nfe is the artifact's schedule, not the base's operating point", str(ex["nfe"]))
        check("va_test_train_cfg.py" in text, "the citation names the artifact file")
        check("training-config artifact shipped in the package" in text,
              "and says what kind of evidence it is")
        check("0 to fill" in text and "FILL_ME" not in json.dumps(ex),
              "nothing left to fill")


def test_wan_va_conflicting_or_nonliteral_evidence_stays_fill_me():
    print("\n=== 2. wan_va: conflicting/non-literal artifact values prove nothing ===")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_dir(Path(td), cfg_py="cfg.height = 224\ncfg.width = 384\n",
                       extra_py="other.height = 256\n")
        _, text, ex = scaffold(d)
        check(ex["height"] == "FILL_ME", "a key assigned two different literals stays FILL_ME")
        check("conflicting values" in text, "and the stop message says why")
        check(ex["width"] == 384, "an uncontested key is still read")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_dir(Path(td), cfg_py=(
            "import os\ncfg.height = int(os.environ.get('H', 224))\n"
            "cfg.env_type = 3\ncfg.obs_cam_keys = []\n"))
        _, text, ex = scaffold(d)
        check(ex["height"] == "FILL_ME", "a computed (non-literal) value is never evaluated")
        check(ex["env_type"] == "FILL_ME", "a wrong-typed literal is not evidence")
        check(ex["obs_cam_keys"] == "FILL_ME", "an empty camera list is not evidence")
    with tempfile.TemporaryDirectory() as td:
        d = wan_va_dir(Path(td), cfg_py="cfg.num_inference_steps = 25\n")
        _, text, ex = scaffold(d)
        check(ex["nfe"] == {"video": 2, "action": 4},
              "one step count alone does not rewrite the schedule (both or neither)")


def cosmos3_dir(td: Path, checkpoint_json) -> Path:
    d = td / "cosmos3-ft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"model_type": "cosmos3_omni",
         "text_config": {"model_type": "cosmos3_edge_text"}}))
    (d / "model.safetensors").write_bytes(b"\x00" * 1024)
    if checkpoint_json is not None:
        (d / "checkpoint.json").write_text(json.dumps(checkpoint_json))
    return d


def test_cosmos3_checkpoint_json_policy_block_is_read():
    print("\n=== 3. cosmos3: checkpoint.json policy block is upstream's own serving artifact ===")
    with tempfile.TemporaryDirectory() as td:
        d = cosmos3_dir(Path(td), {"policy": {"action_chunk_size": 32, "conditioning_fps": 15.0,
                                              "domain_name": "droid_lerobot"}})
        _, text, ex = scaffold(d)
        check(ex["domain_name"] == "droid_lerobot", "domain_name inferred from the policy block")
        check(ex["action_chunk_size"] == 32, "action_chunk_size inferred from the policy block")
        check("checkpoint.json policy.domain_name" in text, "with the citation")
        check("differs from the DROID base declaration's measured 16" in text,
              "a value that disagrees with the base's measured protocol says so")
        for key in ("action_dim", "image_height", "image_width"):
            check(ex[key] == "FILL_ME", f"{key} is not in the artifact, stays FILL_ME")
    with tempfile.TemporaryDirectory() as td:
        d = cosmos3_dir(Path(td), {})    # the Nano release ships an empty checkpoint.json
        _, text, ex = scaffold(d)
        for key in ("domain_name", "action_dim", "action_chunk_size",
                    "image_height", "image_width"):
            check(ex[key] == "FILL_ME", f"empty artifact: {key} honestly stays FILL_ME")
    with tempfile.TemporaryDirectory() as td:
        d = cosmos3_dir(Path(td), {"policy": {"action_chunk_size": "32"}})
        _, text, ex = scaffold(d)
        check(ex["action_chunk_size"] == "FILL_ME", "a wrong-typed policy value is not evidence")


def lingbot_dir(td: Path, yaml_text: str) -> Path:
    d = td / "vla-ft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"type": "pi0"}))
    (d / "lingbotvla_cli.yaml").write_text(yaml_text)
    (d / "model.safetensors").write_bytes(b"\x00" * 1024)
    return d


def test_lingbot_yaml_corroboration_rule():
    print("\n=== 4. lingbot_vla: yaml facts only when the config corroborates itself ===")
    with tempfile.TemporaryDirectory() as td:
        d = lingbot_dir(Path(td), (
            "data:\n  data_name: robotwin\n"
            "  norm_stats_file: assets/norm_stats/robotwin_50.json\n"))
        _, text, ex = scaffold(d)
        check(ex["robot"] == "robotwin",
              "robot inferred: data_name corroborated by the stats filename")
        check(ex["norm_stats"] == "assets/norm_stats/robotwin_50.json",
              "norm_stats read from data.norm_stats_file")
        check("corroborated by" in text, "the citation quotes both yaml statements")
        check("verified at load" in text, "and says the path is still verified at load")
    with tempfile.TemporaryDirectory() as td:
        d = lingbot_dir(Path(td), "data:\n  data_name: multi\n  norm_stats_file: null\n")
        _, text, ex = scaffold(d)
        check(ex["robot"] == "FILL_ME", "an uncorroborated data_name is refused")
        check("data.data_name = 'multi'" in text, "with the yaml value quoted for the human")
        check("training data mix" in text, "and why it is not written")
        check(ex["norm_stats"] == "FILL_ME", "no stats path anywhere: FILL_ME")
    with tempfile.TemporaryDirectory() as td:
        # in-package stats file stays the strongest evidence and wins over the yaml
        d = lingbot_dir(Path(td), (
            "data:\n  data_name: robotwin\n"
            "  norm_stats_file: assets/norm_stats/robotwin_50.json\n"))
        (d / "assets" / "norm_stats").mkdir(parents=True)
        (d / "assets" / "norm_stats" / "myrobot.json").write_text("{}")
        _, text, ex = scaffold(d)
        check(ex["norm_stats"] == "assets/norm_stats/myrobot.json",
              "the file actually shipped in the package outranks the yaml pointer")


def groot_native_dir(td: Path, stats_keys) -> Path:
    d = td / "groot-native"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"model_type": "Gr00tN1d7", "architectures": ["Gr00tN1d7"],
         "num_inference_timesteps": 4, "action_horizon": 40}))
    (d / "model.safetensors").write_bytes(b"\x00" * 1024)
    (d / "statistics.json").write_text(json.dumps({k: {} for k in stats_keys}))
    return d


def test_groot_native_layout_fingerprint_and_statistics_evidence():
    print("\n=== 5. groot_n17: the native Isaac-GR00T layout fingerprints and lists heads ===")
    with tempfile.TemporaryDirectory() as td:
        d = groot_native_dir(Path(td), ["oxe_droid_relative_eef_relative_joint",
                                        "xdof_relative_eef_relative_joint"])
        _, text, ex = scaffold(d)
        check(ex["backbone"] == "groot_n17", "native layout detected")
        check('model_type == "Gr00tN1d7"' in text, "with the fingerprint quoted")
        check(ex["nfe"] == {"backbone": 1, "action": 4},
              "nfe read from config.json num_inference_timesteps")
        check(ex["embodiment_tag"] == "FILL_ME",
              "a multi-head statistics.json is ambiguous, so FILL_ME")
        check("2 heads to choose from" in text
              and "oxe_droid_relative_eef_relative_joint" in text,
              "and the checkpoint's own candidates are listed in the ask")
    with tempfile.TemporaryDirectory() as td:
        d = groot_native_dir(Path(td), ["libero_sim_tag"])
        _, text, ex = scaffold(d)
        check(ex["embodiment_tag"] == "libero_sim_tag",
              "a sole statistics key IS the embodiment, inferred")
        check("sole statistics key of statistics.json" in text, "with that evidence stated")


def test_unknown_backbone_names_the_first_party_package():
    print("\n=== 6. unknown backbone: a first-party family stops with the exact pip install ===")
    from instinctflash.runtime.facade import (
        _first_party_adapter_hint, _unknown_backbone_message,
    )

    hint = _first_party_adapter_hint("pi05")
    check("pip install" in hint and "examples/pi05_vla" in hint,
          "the hint is the install command, pointing at this checkout")
    check("instinctflash.adapters" in hint, "and names the entry-point mechanism")
    check(_first_party_adapter_hint("some_external_family") == "",
          "an unknown family gets no first-party claim")

    class _Ex:                                                    # minimal Checkpoint stand-ins
        model_id, backbone = "my-ft", "some_external_family"

    class _Ckpt:
        model_id, execution = "my-ft", _Ex()

    m = _unknown_backbone_message(_Ckpt(), ["wan_va"])
    check("instinctflash.register(" in m and "examples/tiny_wam" in m,
          "a genuinely external family still gets the write-an-adapter teaching")


def test_broken_installed_adapter_reports_the_import_error():
    print("\n=== 7. an installed-but-broken adapter reports its import error, not a circle ===")
    from instinctflash.runtime import loader
    from instinctflash.runtime.facade import _unknown_backbone_message

    class _Ex:
        model_id, backbone = "my-ft", "pi05"

    class _Ckpt:
        model_id, execution = "my-ft", _Ex()

    saved = dict(loader._DISCOVERY_PROBLEMS)
    try:
        loader._DISCOVERY_PROBLEMS["pi05"] = ("(pi05_iwm.adapter:Pi05Adapter): "
                                              "ModuleNotFoundError: No module named 'numpy'")
        m = _unknown_backbone_message(_Ckpt(), ["wan_va"])
        check("failed to import" in m, "the message says the adapter is installed but broken")
        check("No module named 'numpy'" in m, "and carries the actual import error")
        check("pip install " + str(ROOT / "examples/pi05_vla") not in m,
              "the install hint is NOT repeated — that would be a circle")
    finally:
        loader._DISCOVERY_PROBLEMS.clear()
        loader._DISCOVERY_PROBLEMS.update(saved)


def test_serve_scaffold_announcement_survives_a_failing_preflight():
    print("\n=== 8. serve: the scaffold announcement survives a failing preflight ===")
    from instinctflash.cli import main
    from instinctflash.runtime import loader

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pi05-ft"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "type": "pi05", "num_inference_steps": 10,
            "input_features": {
                "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.state": {"type": "STATE", "shape": [8]},
            }}))
        (d / "model.safetensors").write_bytes(b"\x00" * 2048)
        (d / "policy_preprocessor.json").write_text("{}")

        saved_registry = dict(loader._REGISTRY)
        saved_discovered = loader._DISCOVERED
        out, err = io.StringIO(), io.StringIO()
        try:
            loader._REGISTRY.pop("pi05", None)
            loader._DISCOVERED = True                             # freeze: no plugin rescue
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = main(["serve", str(d), "--serve.dry_run=true"])
        finally:
            loader._REGISTRY.clear()
            loader._REGISTRY.update(saved_registry)
            loader._DISCOVERED = saved_discovered
        both = out.getvalue() + err.getvalue()
        check(rc != 0, "the preflight failure still fails the command", str(rc))
        check(f"wrote {d / 'instinctflash.json'}" in both,
              "the user is told a declaration was just written into their directory")
        check("auto-detected base: lerobot/pi05_base" in both,
              "with the full field-by-field announcement")
        check("UnknownBackboneError" in both, "and then the preflight's own error")


def test_groot_adapter_module_import_stays_dependency_free():
    print("\n=== 9. the groot adapter module imports with no numpy (metadata layer stays light) ===")
    import ast
    src = (ROOT / "examples/groot_n17/groot_n17_iwm/adapter.py").read_text()
    top_imports = [n for n in ast.parse(src).body
                   if isinstance(n, (ast.Import, ast.ImportFrom))]
    heavy = [getattr(a, "name", getattr(n, "module", "")) or ""
             for n in top_imports for a in getattr(n, "names", [])
             if (getattr(a, "name", "") or "").split(".")[0] in ("numpy", "torch", "cv2")]
    modules = {(getattr(n, "module", "") or "").split(".")[0] for n in top_imports
               if isinstance(n, ast.ImportFrom)}
    check(not heavy and not (modules & {"numpy", "torch", "cv2"}),
          "no module-level numpy/torch/cv2 import — the registry can always load it",
          str(heavy or modules & {"numpy", "torch", "cv2"}))


def test_local_bundle_with_checkpoint_subdir_validates_like_its_hub_id():
    print("\n=== 10. a local nested bundle validates via execution.checkpoint_subdir ===")
    from instinctflash.descriptors.package import validate_package
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "v2-bundle"
        nested = d / "checkpoints" / "global_step_10" / "hf_ckpt"
        nested.mkdir(parents=True)
        (d / "lingbotvla_cli.yaml").write_text("data:\n  data_name: multi\n")
        (nested / "config.json").write_text(json.dumps({"vlm_family": "qwen3_vl"}))
        (nested / "model.safetensors").write_bytes(b"\x00" * 1024)
        scaffold(d)                                              # writes the declaration
        # the scaffold leaves robot=FILL_ME (uncorroborated 'multi'); fill it like a user would
        doc = json.loads((d / "instinctflash.json").read_text())
        doc["execution"]["robot"] = "robotwin"
        (d / "instinctflash.json").write_text(json.dumps(doc))
        rep = validate_package(d)
        check("config.json" not in rep.missing,
              "config.json under the declared checkpoint_subdir satisfies the requirement")
        check(any("checkpoint_subdir" in n for n in rep.notes),
              "and the report says where the adapter will load it from")
        check(rep.is_checkpoint and not rep.missing and not rep.problems,
              "the local copy of the bundle validates exactly like its hub id",
              f"missing={list(rep.missing)} problems={list(rep.problems)}")


def test_serve_stops_early_on_a_training_output_layout():
    print("\n=== 11. serve: a directory the loader would refuse stops BEFORE preflight ===")
    from instinctflash.cli import main
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "wanva-nested"
        (d / "transformer").mkdir(parents=True)
        (d / "transformer" / "config.json").write_text(json.dumps(
            {"_class_name": "WanTransformer3DModel", "action_dim": 30}))
        (d / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00" * 4096)
        (d / "va_cfg.py").write_text(
            "cfg.height = 224\ncfg.width = 384\ncfg.env_type = 'none'\n"
            "cfg.obs_cam_keys = ['observation.images.ego']\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["serve", str(d), "--serve.dry_run=true"])
        text = out.getvalue()
        check(rc == 1, "the geometry-complete but unloadable dir still stops", str(rc))
        check("cannot load as a package" in text, "saying the package would not load")
        check(f"mv {d / 'transformer'}" in text.replace("/*", "").replace("  ", " ")
              or "mv " in text and "transformer" in text,
              "with the exact mv from the layout hint")
        check("serve preflight" not in text,
              "and no preflight was printed — the stop comes first")
        # the user runs the mv and the same command proceeds to a clean preflight
        for f in (d / "transformer").iterdir():
            f.rename(d / f.name)
        out2 = io.StringIO()
        with contextlib.redirect_stdout(out2):
            rc2 = main(["serve", str(d), "--serve.dry_run=true"])
        check(rc2 == 0, "after the mv, the same command completes its preflight", str(rc2))
        check("plan tier" in out2.getvalue() or "InstinctFlash plan" in out2.getvalue(),
              "through to the plan")


def main_() -> int:
    test_wan_va_training_cfg_artifact_fills_geometry_and_schedule()
    test_wan_va_conflicting_or_nonliteral_evidence_stays_fill_me()
    test_cosmos3_checkpoint_json_policy_block_is_read()
    test_lingbot_yaml_corroboration_rule()
    test_groot_native_layout_fingerprint_and_statistics_evidence()
    test_unknown_backbone_names_the_first_party_package()
    test_broken_installed_adapter_reports_the_import_error()
    test_serve_scaffold_announcement_survives_a_failing_preflight()
    test_groot_adapter_module_import_stays_dependency_free()
    test_local_bundle_with_checkpoint_subdir_validates_like_its_hub_id()
    test_serve_stops_early_on_a_training_output_layout()
    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)}")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("PASS: the scaffold reads the checkpoint's own artifacts as evidence — with citations, "
          "conflicts refused, and every stop message actionable.")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
