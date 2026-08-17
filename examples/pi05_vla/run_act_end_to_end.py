#!/usr/bin/env python3
"""A second model family, end to end through the public API: declaration -> plan -> real actions.

    /home/ubuntu/tools/pi05env/bin/python examples/pi05_vla/run_act_end_to_end.py

Nothing below names a planner, a pass, a device or a queue. The checkpoint is a DECLARATION over an
upstream LeRobot repo -- no weights are vendored -- which is the shape a third party adopting someone
else's checkpoint actually needs.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

UPSTREAM = "lerobot/act_aloha_sim_transfer_cube_human"
FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def main() -> int:
    import numpy as np
    import instinctwm
    from instinctwm import Runtime, describe

    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "act-declared"
        pkg.mkdir()
        (pkg / "config.json").write_text("{}")
        (pkg / "instinctwm.json").write_text(json.dumps({
            "instinctwm_schema": 1,
            "execution": {"model_id": "example-org/act-declared", "backbone": "act",
                          "servable": True, "guidance": {"action": "none"},
                          "nfe": {"action": 1}, "base_weights": UPSTREAM},
            "provenance": {"note": "declaration over an upstream LeRobot checkpoint"},
        }, indent=2))

        print("=" * 78)
        print("1. describe() -- a declaration with no vendored weights")
        print("=" * 78)
        d = describe(pkg)
        print(f"  backbone {d['backbone']}  servable {d['servable']}  nfe {d['nfe']}")
        check(d["backbone"] == "act", "declares the act backbone")

        print("\n" + "=" * 78)
        print("2. the backbone resolves, and a plan is compiled")
        print("=" * 78)
        import act_iwm  # noqa: F401  registers when not pip-installed
        instinctwm.register("act", act_iwm.ACTAdapter) if "act" not in \
            instinctwm.available_models() else None
        print(f"  registered: {instinctwm.available_models()}")
        runtime = Runtime.from_pretrained(pkg)
        check(runtime.plan is not None, "a plan was compiled")
        static, why = runtime.checkpoint and True, ""
        from instinctwm import load
        ok, why = load("act").spec().shapes_static_across_cycles()
        check(ok, "shapes are static across cycles (so capture would be viable here)", why[:52])

        print("\n" + "=" * 78)
        print("3. real actions, in a closed loop")
        print("=" * 78)
        obs = {"observation.images.top": np.zeros((1, 3, 480, 640), dtype=np.float32),
               "observation.state": np.zeros((1, 14), dtype=np.float32)}
        with runtime:
            with runtime.episode() as ep:
                first = None
                for i in range(1, 6):
                    out = ep.predict(obs)
                    a = np.asarray(out["action"])
                    if first is None:
                        first = a
                        print(f"  action {a.shape} {a.dtype}  "
                              f"{[round(float(v), 4) for v in a.ravel()[:5]]}")
                check(ep.steps == 5, "five consecutive control cycles", str(ep.steps))
                check(bool(np.isfinite(first).all()), "the action is finite")
                check(first.size > 0, "and non-empty")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: a LeRobot ACT policy served through InstinctWM from a declaration alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
