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
    "nvidia/GR00T-N1.7-3B": {
        "instinctflash_schema": 1,
        "execution": {
            "model_id": "nvidia/GR00T-N1.7-3B",
            "backbone": "groot_n17",
            "servable": True,
            "guidance": {"action": "none"},
            "nfe": {"backbone": 1, "action": 4},
            "base_weights": "nvidia/GR00T-N1.7-3B",
            "embodiment_tag": "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
            "param_bytes": 6910499416,
        },
    },
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
    # LeRobot publishes pi05 checkpoints without a declaration. The adapter package is
    # examples/pi05_vla (backbone "pi05"); execution facts below are read from each
    # checkpoint's own config.json (three 224x224 cameras, 32-dim state, chunk 50,
    # num_inference_steps 10), not guessed.
    "lerobot/pi05_base": {
        "instinctflash_schema": 1,
        "execution": {
            "model_id": "lerobot/pi05_base",
            "backbone": "pi05",
            "servable": True,
            "guidance": {"action": "none"},
            "nfe": {"prefix": 1, "action": 10},
            "base_weights": "lerobot/pi05_base",
            # Observation geometry, from the checkpoint's own config.json. Declared EXPLICITLY:
            # the adapter refuses to guess a fine-tune's cameras (a pi05 fine-tune renames and
            # reshapes them — see the v044 entry below), so every declaration states its own.
            "obs_features": {
                "observation.images.base_0_rgb": [3, 224, 224],
                "observation.images.left_wrist_0_rgb": [3, 224, 224],
                "observation.images.right_wrist_0_rgb": [3, 224, 224],
                "observation.state": [32],
            },
            "param_bytes": 14467165872,
        },
    },
    # The LIBERO fine-tune is bf16-stored — the realistic serving artifact, and the checkpoint
    # the README H100 row (206.7 -> 72.8 ms) was measured on. Its observation contract differs
    # from the base (train_config.json input_features): two 256x256 cameras, one empty 224
    # camera, an 8-dim state — which is exactly why obs geometry is declared, never assumed.
    "lerobot/pi05_libero_finetuned_v044": {
        "instinctflash_schema": 1,
        "execution": {
            "model_id": "lerobot/pi05_libero_finetuned_v044",
            "backbone": "pi05",
            "servable": True,
            "guidance": {"action": "none"},
            "nfe": {"prefix": 1, "action": 10},
            "base_weights": "lerobot/pi05_libero_finetuned_v044",
            "obs_features": {
                "observation.images.image": [3, 256, 256],
                "observation.images.image2": [3, 256, 256],
                "observation.images.empty_camera_0": [3, 224, 224],
                "observation.state": [8],
            },
            "param_bytes": 7473096344,
        },
    },
    # The 4B family. The upstream release is a flat checkpoint (safetensors +
    # lingbotvla_cli.yaml) served by its own deploy/lingbot_vla_policy.py, which the adapter
    # package (examples/lingbot_vla) wraps in-process. The action norm stats ship with the
    # upstream CHECKOUT, not the checkpoint, so they are declared here and verified at load.
    "robbyant/lingbot-vla-4b-posttrain-robotwin": {
        "instinctflash_schema": 1,
        "execution": {
            "model_id": "robbyant/lingbot-vla-4b-posttrain-robotwin",
            "backbone": "lingbot_vla",
            "servable": True,
            "guidance": {"action": "none"},
            "nfe": {"prefix": 1, "action": 10},
            "base_weights": "robbyant/lingbot-vla-4b-posttrain-robotwin",
            "tokenizer_repo": "Qwen/Qwen2.5-VL-3B-Instruct",
            "robot": "robotwin",
            "norm_stats": "assets/norm_stats/robotwin_50.json",
            "use_length": 25,
            "param_bytes": 16789932052,
        },
    },
    # The upstream VLA-V2 release keeps its HF checkpoint three directories below the
    # repository root and does not publish an InstinctFlash declaration. Keep the original
    # bytes in place; `_declared_view` exposes the nested config at the package root while the
    # adapter consumes `checkpoint_subdir` to load the six shards from their real location.
    "robbyant/lingbot-vla-v2-6b-robotwin": {
        "instinctflash_schema": 1,
        "execution": {
            "model_id": "robbyant/lingbot-vla-v2-6b-robotwin",
            "backbone": "lingbot_vla_v2",
            "servable": True,
            "guidance": {"action": "none"},
            "nfe": {"prefix": 1, "action": 10},
            "base_weights": "robbyant/lingbot-vla-v2-6b-robotwin",
            "checkpoint_subdir": "checkpoints/global_step_50000/hf_ckpt",
            "tokenizer_repo": "Qwen/Qwen3-VL-4B-Instruct",
            "robot": "robotwin",
            "param_bytes": 25503630044,
        },
    },
}


def lookup(model_id: str) -> "dict | None":
    """The declaration for a known upstream release, or None. Exact repo-id match only."""
    doc = KNOWN_DECLARATIONS.get(str(model_id))
    return copy.deepcopy(doc) if doc is not None else None
