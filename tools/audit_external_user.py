#!/usr/bin/env python3
"""Part A measurement: cold-cache cost, first action, and closed-loop, on the real checkpoint.

Reports numbers rather than opinions: bytes fetched, seconds per stage, and whether repeated
predict() works. Run with the package's caches cleared to make the cold numbers meaningful.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HUB = Path.home() / ".cache" / "huggingface" / "hub"


def dirsize(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--cold", action="store_true", help="delete this package's caches first")
    ap.add_argument("--loops", type=int, default=4)
    a = ap.parse_args()

    slug = "models--" + a.package.replace("/", "--")
    caches = [HUB / slug, HUB / "models--robbyant--lingbot-va-posttrain-robotwin",
              Path.home() / ".cache" / "instinctflash"]
    if a.cold:
        for c in caches:
            if c.exists():
                shutil.rmtree(c)
        print("cold: caches removed")
    before = sum(dirsize(c) for c in caches)

    from instinctflash import Runtime, describe

    t0 = time.time()
    d = describe(a.package)
    t_describe = time.time() - t0
    print(f"\ndescribe()          {t_describe:6.2f}s   servable={d['servable']} "
          f"backbone={d['backbone']}")

    t0 = time.time()
    runtime = Runtime.from_pretrained(a.package)
    t_load = time.time() - t0
    fetched = sum(dirsize(c) for c in caches) - before
    print(f"from_pretrained()   {t_load:6.2f}s   fetched {fetched/1e9:5.2f} GB")

    obs, prompt = _observation()
    with runtime:
        t0 = time.time()
        runtime.reset(prompt=prompt)
        t_reset = time.time() - t0
        t0 = time.time()
        out = runtime.predict(obs)
        t_first = time.time() - t0
        print(f"reset()             {t_reset:6.2f}s")
        print(f"first predict()     {t_first:6.2f}s")

        print(f"\nclosed loop: {a.loops} consecutive predict() calls, no reset")
        ok = 1
        for i in range(2, a.loops + 1):
            try:
                runtime.predict(obs)
                ok = i
                print(f"  cycle {i}  OK")
            except Exception as e:                                # noqa: BLE001
                print(f"  cycle {i}  BROKE: {type(e).__name__}: {str(e)[:110]}")
                break
        print(f"\nVERDICT  cold {fetched/1e9:.2f} GB, first action in "
              f"{t_describe + t_load + t_reset + t_first:.1f}s, closed loop {ok}/{a.loops}")
        return 0 if ok == a.loops else 1


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _observation(n_frames: int = 8):
    """`n_frames` real recorded frames of the SAME task.

    HONESTY NOTE. Each recorded npz is the FIRST observation of a distinct episode, so a stack of
    them is not a trajectory. This exercises the control-cycle state machine -- does repeated
    predict() work through the public API -- and says nothing about whether a rollout succeeds. Task
    success lives in the RoboTwin evaluation, which needs the simulator.
    """
    import numpy as np
    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    task = ctx[0].name.split("__")[0]
    same = [p for p in ctx if p.name.startswith(task)][:max(n_frames, 1)]
    zs = [np.load(p, allow_pickle=True) for p in same]
    prompt = str(zs[0]["prompt"])
    frames = [{f"observation.images.{c}": z[c] for c in CAMERAS if c in z.files} for z in zs]
    return {"obs": frames, "prompt": prompt, "save_visualization": False}, prompt


if __name__ == "__main__":
    raise SystemExit(main())
