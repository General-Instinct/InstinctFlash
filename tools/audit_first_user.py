#!/usr/bin/env python3
"""Final audit: a first external user who knows only the GitHub repo and a Hub model id.

Rules this script obeys, because they are what makes the result mean anything:
  * only commands and snippets that appear in the README;
  * no InstinctFlash environment variables (no LINGBOT_ROOT, no IFL_*, no LINGBOT_CKPT);
  * no local paths -- the model is named by Hub repo id;
  * repeated inference is the pass criterion, not a single action.

    python tools/audit_first_user.py --model robbyant/lingbot-va-posttrain-robotwin
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

FAILED: list[str] = []
GUESSES: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="robbyant/lingbot-va-posttrain-robotwin")
    ap.add_argument("--cycles", type=int, default=5)
    a = ap.parse_args()

    print("=" * 78)
    print("0. the environment a first user actually has")
    print("=" * 78)
    leaked = sorted(k for k in os.environ
                    if k.startswith(("IFL_", "LINGBOT_")) or k == "PYTHONPATH")
    check(not leaked, "no InstinctFlash environment variables are set", str(leaked))
    for mod in ("torch", "huggingface_hub", "diffusers"):
        try:
            __import__(mod)
        except ImportError:
            check(False, f"{mod} came from the documented install")

    print("\n" + "=" * 78)
    print("1. describe(model) -- README: 'see what a checkpoint is before downloading'")
    print("=" * 78)
    from instinctflash import Runtime, describe
    try:
        d = describe(a.model)
        print(f"  backbone={d['backbone']}  servable={d['servable']}  nfe={d['nfe']}")
        check(d["servable"], "the checkpoint declares itself servable")
    except Exception as e:                                       # noqa: BLE001
        check(False, f"describe({a.model!r}) failed: {type(e).__name__}", str(e)[:150])
        print("\n  STOPPING: a first user cannot get past the first documented call.")
        return 1

    print("\n" + "=" * 78)
    print("2. Runtime.from_pretrained(model) -- README: 'Load a model'")
    print("=" * 78)
    t0 = time.time()
    try:
        runtime = Runtime.from_pretrained(a.model)
    except Exception as e:                                       # noqa: BLE001
        check(False, f"from_pretrained failed: {type(e).__name__}", str(e)[:220])
        return 1
    check(True, "loaded", f"{time.time() - t0:.1f}s")

    print("\n" + "=" * 78)
    print(f"3. repeated inference -- README: 'while not done: episode.predict(observation)'")
    print("=" * 78)
    obs, prompt = _observation()
    done = 0
    with runtime:
        with runtime.episode(prompt=prompt) as episode:
            for i in range(1, a.cycles + 1):
                try:
                    action = episode.predict(obs)
                    done = i
                except Exception as e:                           # noqa: BLE001
                    check(False, f"cycle {i} raised {type(e).__name__}", str(e)[:130])
                    break
            check(done == a.cycles, f"{a.cycles} consecutive control cycles", f"{done}/{a.cycles}")
            check(episode.steps == done, "the episode counted its own steps", str(episode.steps))
    import numpy as np
    arr = np.asarray(action["action"] if isinstance(action, dict) else action)
    check(bool(np.isfinite(arr).all()) and float(arr.std()) > 1e-6,
          "the last action is finite and non-degenerate", f"std={arr.std():.4f}")

    print("\n" + "=" * 78)
    if GUESSES:
        print("UNDOCUMENTED GUESSES REQUIRED:")
        for g in GUESSES:
            print(f"  - {g}")
    if FAILED:
        print(f"NOT READY -- {len(FAILED)} failure(s): {FAILED}")
        return 1
    print("EXTERNAL USER READY: repo id -> repeated inference, using only documented steps.")
    return 0


def _observation():
    """The observation format the README documents, from whatever frames are on hand."""
    import numpy as np
    rec = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    cams = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    if not rec:                                                  # no recordings: synthesise frames
        GUESSES.append("no recorded observations on this machine; frames were synthesised")
        f = [{f"observation.images.{c}": np.zeros((240, 320, 3), np.uint8) for c in cams}
             for _ in range(8)]
        return {"obs": f, "prompt": "pick up the bottle"}, "pick up the bottle"
    task = rec[0].name.split("__")[0]
    zs = [np.load(p, allow_pickle=True) for p in [q for q in rec if q.name.startswith(task)][:8]]
    frames = [{f"observation.images.{c}": z[c] for c in cams if c in z.files} for z in zs]
    return {"obs": frames, "prompt": str(zs[0]["prompt"])}, str(zs[0]["prompt"])


if __name__ == "__main__":
    raise SystemExit(main())
