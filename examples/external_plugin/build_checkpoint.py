#!/usr/bin/env python3
"""Write a real `some-org/my-world-model` package: weights + declaration, nothing else.

Run from this directory:  PYTHONPATH=. python build_checkpoint.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from gridworld_wm.model import GridworldAR

OUT = Path(__file__).parent / "my-world-model"
CFG = {"vocab": 64, "dim": 32, "action_dim": 4, "history": 8}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = GridworldAR(**CFG)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(OUT / "model.safetensors"))
    (OUT / "config.json").write_text(json.dumps(CFG, indent=2) + "\n")

    # The declaration. Execution facts only -- no model-specific knowledge, which is the point:
    # `history`, `vocab` and the GRU live in config.json and in the adapter, NOT here.
    (OUT / "instinctwm.json").write_text(json.dumps({
        "instinctwm_schema": 1,
        "execution": {
            "model_id": "some-org/my-world-model",
            "backbone": "gridworld_ar",
            "servable": True,
            "guidance": {},
            "nfe": {"act": 1},
        },
        "provenance": {"note": "trained by hand, in a hurry, on nothing"},
    }, indent=2) + "\n")
    (OUT / "README.md").write_text("# my-world-model\n\nA toy autoregressive world-action model.\n")
    total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    print(f"wrote {OUT}  ({total:,} bytes)")


if __name__ == "__main__":
    main()
