#!/usr/bin/env python3
"""Write a real, publishable checkpoint package for the tiny model.

Produces exactly what a checkpoint author would push to the Hub:

    examples/checkpoint/tiny-wam-2v2a/
      instinctflash.json      the declaration -- execution + provenance
      config.json          the backbone's own config
      model.safetensors    REAL WEIGHTS, ~1 MB, loadable by safetensors
      README.md            model card

The weights are seeded, so re-running this reproduces the same file byte for byte and the end-to-end
script's output is stable.

    python examples/tiny_wam/build_checkpoint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from examples.tiny_wam import model as M  # noqa: E402

OUT = ROOT / "examples" / "checkpoint" / "tiny-wam-2v2a"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    net = M.build(seed=0)
    state = {k: v.contiguous() for k, v in net.state_dict().items()}
    param_bytes = sum(v.numel() * v.element_size() for v in state.values())
    save_file(state, str(OUT / "model.safetensors"),
              metadata={"format": "pt", "instinctflash_example": "tiny-wam"})

    (OUT / "config.json").write_text(json.dumps({
        "_class_name": "TinyWAM",
        "architectures": ["TinyWAM"],
        "hidden_size": M.HIDDEN,
        "num_layers": M.LAYERS,
        "num_attention_heads": M.HEADS,
        "obs_dim": M.OBS_DIM,
        "action_dim": M.ACTION_DIM,
        "action_horizon": M.ACTION_HORIZON,
        "_comment": "The backbone's own config. InstinctFlash does not read this file; the adapter and "
                    "the modelling code do.",
    }, indent=2) + "\n")

    (OUT / "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {
            "model_id": "example-org/tiny-wam-2v2a",
            # must name a REGISTERED adapter, or the checkpoint is not servable however well it
            # declares itself. That is the current, explicit contract.
            "backbone": "tiny-wam",
            "servable": True,
            "guidance": {"video": "cfg", "action": "positive_only"},
            "nfe": {"video": 2, "action": 2},
            "output_projection": {
                "kind": "per_interval_velocity_heads",
                "n_intervals": M.N_INTERVALS,
                "block": M.BLOCK,
                "velocity_convention": "sigma_descending",
                "foldable": True,
            },
            "param_bytes": param_bytes,
        },
        "provenance": {
            "training_method": "none -- weights are seeded random",
            "seed": 0,
            "note": "This block is FOR HUMANS. Delete it and the checkpoint still serves; "
                    "publishability() verifies exactly that.",
        },
    }, indent=2) + "\n")

    print(f"wrote {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:22} {f.stat().st_size:>9,} bytes")
    print(f"  param_bytes declared: {param_bytes:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
