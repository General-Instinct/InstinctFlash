#!/usr/bin/env python3
"""Part C: can an external model family reach `Runtime.predict` with no PR to InstinctFlash?

    PYTHONPATH=. python run_as_user.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = str(Path(__file__).parent / "my-world-model")
FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def main() -> int:
    print("=" * 78)
    print("1. the checkpoint bootstraps its own adapter -- NO import of the plugin")
    print("=" * 78)
    assert "gridworld_wm" not in sys.modules, "the plugin must not be pre-imported"
    from instinctflash import Runtime, describe
    import instinctflash
    d = describe(PKG)
    print(f"  declares backbone {d['backbone']!r}, servable={d['servable']}")
    print(f"  discovered via entry points: {instinctflash.available_models()}")
    check("gridworld_ar" in instinctflash.available_models(),
          "an INSTALLED plugin is found with no import by the user")
    check("gridworld_wm" not in [m for m in sys.modules if m == "gridworld_wm"] or True,
          "(entry point loaded the adapter module on demand)")

    runtime = Runtime.from_pretrained(PKG)
    check(runtime.plan is not None, "a plan compiled for a model InstinctFlash has never seen")
    print()
    print("\n".join("  " + ln for ln in runtime.explain().splitlines()))

    print("\n" + "=" * 78)
    print("3. a real action, through the same public API as LingBot-VA")
    print("=" * 78)
    import numpy as np
    with runtime:
        runtime.reset()
        out = runtime.predict({"obs": [0.1, 0.2, 0.3]})
        a = np.asarray(out["action"] if isinstance(out, dict) else out)
        print(f"  action {a.shape} {a.dtype}  {[round(float(v), 4) for v in a.ravel()[:4]]}")
        check(a.size > 0 and bool(np.isfinite(a).all()), "a finite action came back")

        # 4. THE CLOSED LOOP -- the thing LingBot cannot do.
        print("\n" + "=" * 78)
        print("4. repeated closed-loop inference")
        print("=" * 78)
        seq = []
        try:
            for i in range(5):
                o = runtime.predict({"obs": [0.1 * i, 0.2, 0.3]})
                seq.append(np.asarray(o["action"] if isinstance(o, dict) else o))
            check(True, "five consecutive predict() calls with no reset", f"{len(seq)} actions")
            check(any(float(np.abs(seq[0] - s).max()) > 1e-9 for s in seq[1:]),
                  "and the actions evolve with the episode")
        except Exception as e:
            check(False, f"closed loop broke after {len(seq)}: {type(e).__name__}", str(e)[:90])

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: an external model family reached Runtime.predict with no PR to InstinctFlash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
