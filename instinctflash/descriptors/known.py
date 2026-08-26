"""Declarations for published checkpoints whose upstream repos do not carry one.

A checkpoint should carry its own `instinctflash.json`, and one found in the repo always wins.
This table exists for upstream releases we can serve but do not control: their authors publish
weights without a declaration, and republishing 9.5 GB of unmodified third-party weights to add
two kilobytes of JSON would be waste dressed as packaging. Pointing at the original repo id
just works instead.

Entries are keyed by the exact Hub repo id and hold a complete declaration document — the same
schema `load_declaration` enforces, forbidden-key checks included.
"""

from __future__ import annotations

import copy

KNOWN_DECLARATIONS: dict[str, dict] = {
    "robbyant/lingbot-va-posttrain-robotwin": {
        "instinctflash_schema": 1,
        "execution": {
            "model_id": "robbyant/lingbot-va-posttrain-robotwin",
            "backbone": "wan_va",
            "servable": True,
            "guidance": {"video": "cfg", "action": "positive_only"},
            "nfe": {"video": 2, "action": 4},
            "base_weights": "robbyant/lingbot-va-posttrain-robotwin",
            # Observation geometry, transcribed from wan_va/configs/va_robotwin_cfg.py. Declared
            # EXPLICITLY because the adapter refuses to guess these: a wan_va checkpoint states
            # its cameras and resolution or names an IFL_CFG entry, and the built-ins follow the
            # same rule rather than being a hidden fourth resolution source.
            "obs_cam_keys": ["observation.images.cam_high",
                             "observation.images.cam_left_wrist",
                             "observation.images.cam_right_wrist"],
            "height": 256,
            "width": 320,
            "env_type": "robotwin_tshape",
            "param_bytes": 10179017396,
        },
    },
}


def lookup(model_id: str) -> "dict | None":
    """The declaration for a known upstream release, or None. Exact repo-id match only."""
    doc = KNOWN_DECLARATIONS.get(str(model_id))
    return copy.deepcopy(doc) if doc is not None else None
