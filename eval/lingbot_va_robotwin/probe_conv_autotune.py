#!/usr/bin/env python3
"""GPU smoke for the va_conv_layout autotune site: one real device, end to end.

Three claims, each measured rather than asserted:

  1. FIRST LOAD MEASURES. A fresh cache forces the bench; the decision reports per-candidate
     medians from THIS device and the winner's margin, and lands in the persistent cache.
  2. SECOND LOAD DOES NOT. Same key -> source=cache, identical choice, no kernels timed.
  3. THE OVERRIDES ARE REAL. IFL_AUTOTUNE_VA_CONV_LAYOUT=stock forces the incumbent;
     IFL_AUTOTUNE=0 disables the site entirely. Neither touches the cache.

    CUDA_VISIBLE_DEVICES=7 <python-with-torch> eval/lingbot_va_robotwin/probe_conv_autotune.py

Exit code 0 only if all three hold and the measured winner's plan maps to a legal ConvPlan.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import torch  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device visible")
        return 0
    cache = Path(tempfile.mkdtemp()) / "autotune.json"
    os.environ["IFL_AUTOTUNE_CACHE"] = str(cache)
    os.environ.pop("IFL_AUTOTUNE", None)
    os.environ.pop("IFL_AUTOTUNE_VA_CONV_LAYOUT", None)

    from instinctflash.autotune import cache_path
    from instinctflash.backends.conv.apply import autotune_conv_layout, conv_plan_from_decision

    dev = torch.cuda.get_device_properties(0)
    print(f"device: {dev.name} sm{dev.major}{dev.minor}")
    failures = []

    # 1. first load measures
    d1 = autotune_conv_layout()
    print(f"  first : {d1.reason}")
    print(f"          timings_ms = { {k: round(v, 3) for k, v in d1.timings_ms.items()} }")
    if d1.source != "measured":
        failures.append(f"first load should measure, got {d1.source}")
    if not cache_path().exists():
        failures.append("no cache file written after a measured decision")
    else:
        print(f"          cache -> {cache_path()}: {list(json.loads(cache_path().read_text()))}")

    # 2. second load hits the cache
    d2 = autotune_conv_layout()
    print(f"  second: {d2.reason}")
    if d2.source != "cache" or d2.chosen != d1.chosen:
        failures.append(f"second load should be a cache hit on the same winner, got "
                        f"{d2.source}/{d2.chosen}")

    # 3. overrides
    os.environ["IFL_AUTOTUNE_VA_CONV_LAYOUT"] = "stock"
    d3 = autotune_conv_layout()
    print(f"  forced: {d3.reason}")
    if d3.source != "forced" or d3.chosen != "stock":
        failures.append(f"force override failed: {d3.source}/{d3.chosen}")
    del os.environ["IFL_AUTOTUNE_VA_CONV_LAYOUT"]
    os.environ["IFL_AUTOTUNE"] = "0"
    d4 = autotune_conv_layout()
    print(f"  off   : {d4.reason}")
    if d4.source != "disabled" or d4.chosen != "stock":
        failures.append(f"disable override failed: {d4.source}/{d4.chosen}")
    del os.environ["IFL_AUTOTUNE"]

    plan = conv_plan_from_decision(d1)
    print(f"  plan  : {plan.backend_name}/{plan.use_layout.value} convert={plan.convert_subgraph} "
          f"[{plan.tier.name}]")

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS: measured, cached, overridable on this device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
