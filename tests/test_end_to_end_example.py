#!/usr/bin/env python3
"""Run the published end-to-end example as a test, so it cannot rot.

An example that is only ever run by hand drifts from the code it demonstrates. This executes
`examples/tiny_wam/run_end_to_end.py` in-process and fails if any of its nine steps fail.

Needs torch and safetensors, so it skips in the core venv like every other model-touching test.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import safetensors.torch  # noqa: F401
    import torch  # noqa: F401
except ImportError:                                    # pragma: no cover - core venv
    print("SKIP: needs torch")
    raise SystemExit(0)


def main() -> int:
    ckpt = ROOT / "examples" / "checkpoint" / "tiny-wam-2v2a"
    if not (ckpt / "model.safetensors").exists():
        from examples.tiny_wam import build_checkpoint
        build_checkpoint.main()
    from examples.tiny_wam import run_end_to_end
    return run_end_to_end.main()


if __name__ == "__main__":
    raise SystemExit(main())
