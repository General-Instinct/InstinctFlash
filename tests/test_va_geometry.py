#!/usr/bin/env python3
"""Observation geometry is declared, or explicitly overridden — never silently defaulted.

THE FAILURE THIS PINS. The wan_va adapter used to pick its upstream config with
`VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]`: a custom checkpoint served without an env
var no document mentioned ran under robotwin's camera keys, resolution and T-shape compositing —
conditioning entirely wrong, with no warning — and `--serve.smoke` printed robotwin's camera
names as what the model "expects". Found by a fresh-user walkthrough importing a fans-robot
checkpoint (2026-08-26 usability journal, item 6).

The rule, per geometry key: declaration > IFL_CFG > FAIL LOUD with the fix in the message.
No GPU, no torch, no upstream tree — the resolver is a pure function and is tested as one.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instinctflash.adapters.lingbot_va import (  # noqa: E402
    GEOMETRY_KEYS, LingBotVA, resolve_observation_geometry,
)

FANS = {
    "obs_cam_keys": ["observation.images.ego", "observation.images.left_wrist",
                     "observation.images.right_wrist"],
    "height": 224, "width": 384, "env_type": "none",
}

ROBOTWIN_CFG = SimpleNamespace(
    obs_cam_keys=["observation.images.cam_high", "observation.images.cam_left_wrist",
                  "observation.images.cam_right_wrist"],
    height=256, width=320, env_type="robotwin_tshape")


def _execution(extra, model_id="example-org/fans-8000", source="/pkg/instinctflash.json"):
    return SimpleNamespace(extra=dict(extra), model_id=model_id, source=source)


def test_declaration_provides_geometry():
    geom, source = resolve_observation_geometry(_execution(FANS), environ={})
    assert geom == FANS, geom
    assert "declaration" in source and "instinctflash.json" in source, source


def test_declaration_outranks_ifl_cfg():
    geom, source = resolve_observation_geometry(
        _execution(FANS), environ={"IFL_CFG": "robotwin"},
        va_configs={"robotwin": ROBOTWIN_CFG})
    assert geom["obs_cam_keys"] == FANS["obs_cam_keys"], "declared cameras must win over IFL_CFG"
    assert (geom["height"], geom["width"]) == (224, 384)
    assert "declaration" in source, source


def test_partial_declaration_merges_over_env_config():
    partial = {"obs_cam_keys": FANS["obs_cam_keys"]}
    geom, source = resolve_observation_geometry(
        _execution(partial), environ={"IFL_CFG": "robotwin"},
        va_configs={"robotwin": ROBOTWIN_CFG})
    assert geom["obs_cam_keys"] == FANS["obs_cam_keys"]      # declared key wins
    assert (geom["height"], geom["width"], geom["env_type"]) == (256, 320, "robotwin_tshape")
    assert "IFL_CFG=robotwin" in source and "obs_cam_keys" in source, source


def test_env_provides_geometry():
    geom, source = resolve_observation_geometry(
        _execution({}), environ={"IFL_CFG": "robotwin"},
        va_configs={"robotwin": ROBOTWIN_CFG})
    assert geom["obs_cam_keys"] == ROBOTWIN_CFG.obs_cam_keys
    assert (geom["height"], geom["width"], geom["env_type"]) == (256, 320, "robotwin_tshape")
    assert source == "IFL_CFG=robotwin", source


def test_missing_geometry_fails_loud_with_the_fix_in_the_message():
    try:
        resolve_observation_geometry(_execution({}), environ={})
    except RuntimeError as e:
        m = str(e)
    else:
        raise AssertionError("missing geometry must raise, not default to robotwin")
    assert "example-org/fans-8000" in m, "names the checkpoint"
    for key in GEOMETRY_KEYS:
        assert key in m, f"names the missing key {key!r}: {m}"
    assert "instinctflash.json" in m and '"env_type"' in m, "shows the declaration snippet to add"
    assert "IFL_CFG" in m, "names the env override"
    assert "wins over IFL_CFG" in m, "states the resolution order"


def test_unknown_ifl_cfg_name_lists_the_known_ones():
    try:
        resolve_observation_geometry(
            _execution({}), environ={"IFL_CFG": "nope"}, va_configs={"robotwin": ROBOTWIN_CFG})
    except RuntimeError as e:
        assert "nope" in str(e) and "robotwin" in str(e), str(e)
    else:
        raise AssertionError("an unknown IFL_CFG name must be refused, not KeyError'd")


def test_builtin_known_declarations_carry_the_fields():
    # robbyant/* keeps working with neither a local declaration nor IFL_CFG because the built-in
    # declaration now states its geometry explicitly — the same rule, not a fourth source.
    from instinctflash.descriptors.known import lookup
    ex = lookup("robbyant/lingbot-va-posttrain-robotwin")["execution"]
    for key in GEOMETRY_KEYS:
        assert key in ex, f"built-in declaration must carry {key}"
    assert ex["env_type"] == "robotwin_tshape" and (ex["height"], ex["width"]) == (256, 320)
    # and the declaration still loads through the real reader (forbidden-key check included)
    import json
    import tempfile
    from instinctflash.descriptors.checkpoint import load_declaration
    from instinctflash.descriptors.known import lookup as lk
    with tempfile.TemporaryDirectory() as td:
        Path(td, "instinctflash.json").write_text(
            json.dumps(lk("robbyant/lingbot-va-posttrain-robotwin")))
        decl = load_declaration(td)
    got, source = resolve_observation_geometry(decl, environ={})
    assert got["obs_cam_keys"][0] == "observation.images.cam_high"
    assert "declaration" in source


def test_observation_contract_reflects_the_checkpoint():
    ckpt = SimpleNamespace(execution=_execution(FANS))
    spec, source = LingBotVA().observation_contract(ckpt)
    assert [f.key for f in spec.fields] == FANS["obs_cam_keys"]
    assert all(f.shape == (224, 384, 3) for f in spec.fields), [f.shape for f in spec.fields]
    assert spec.history == 8 and spec.frames_key == "obs", "non-geometry contract facts unchanged"
    assert "declaration" in source


def test_no_silent_default_survives_in_the_adapter():
    src = (ROOT / "instinctflash" / "adapters" / "lingbot_va.py").read_text()
    assert 'os.environ.get("IFL_CFG", "robotwin")' not in src, \
        "the silent robotwin default is back"
    cli = (ROOT / "instinctflash" / "cli.py").read_text()
    smoke = cli[cli.index("def _serve_smoke"):]
    smoke = smoke[:smoke.index("\ndef ", 10)]
    assert "observation_source" in smoke, "the smoke test must print which geometry source it used"


if __name__ == "__main__":
    from run_tests import run_module_tests

    raise SystemExit(run_module_tests(globals()))
