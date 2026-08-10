#!/usr/bin/env python3
"""The real LingBot-VA workflow, end to end, through the public API only.

    describe(repo)                    metadata, no weights
    Runtime.from_pretrained(repo)     declaration -> adapter -> capabilities -> plan -> placement
    runtime.reset(prompt=...)         start an episode
    runtime.predict(obs)              one control step, on the real 10.2 GB checkpoint

The point is that nothing in this script mentions a planner, a pass, a tier, a port or a socket. The
model runs behind a websocket in a managed worker -- because the serving environment and the caller's
environment are genuinely incompatible on this box -- and the caller cannot tell.

WHY A WORKER HERE. Re-verified 2026-08-09: the client interpreter (torch 2.4.1 + sapien) cannot import
diffusers/transformers/safetensors, and the serving interpreter (torch 2.9.0 + diffusers) cannot
import sapien/mplib. The boundary is bidirectional and real, so it stays. `Runtime` hides it: run this
under the SERVING interpreter and placement resolves to in-process; run it under the client
interpreter and it resolves to a worker. Same three calls either way.

    PYTHONPATH=. $IWM_SERVER_PY tools/lingbot_end_to_end.py --package /home/ubuntu/hub/lingbot-va
    PYTHONPATH=. $IWM_SERVER_PY tools/lingbot_end_to_end.py --package ... --placement worker

STATUS, 2026-08-10. All four steps pass against a real Hugging Face repo id on an idle H100-80GB.
Needs a GPU with >= ~30 GB free: the resident set is transformer 10.18 + vae 0.51 + text_encoder
11.36 = 22.05 GB before activations, and it OOMs on a card with only ~23.4 GB free.

    CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=$IWM_FA_SHIM_DIR:. MASTER_ADDR=127.0.0.1 \
      MASTER_PORT=29854 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 HF_TOKEN=... $IWM_SERVER_PY \
      tools/lingbot_end_to_end.py --package GM717/lingbot-va \
      --base-weights /home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin

PROTOCOL: `predict()` IS NOT A WHOLE CONTROL CYCLE, AND THAT IS A KNOWN FACADE GAP. For wan_va a
cycle is two server calls -- the action prediction, then a KV-commit carrying observed frames plus
the executed action, which is what advances the ring. `Runtime.predict()` currently maps to the
first only, so the obvious user loop

    while True:
        action = runtime.predict(obs)          # <-- second iteration raises

dies on the second iteration with "Calculated padded input size per channel: (2 x 32 x 40). Kernel
size: (3 x 1 x 1)" -- the ring never advanced, so the third temporal tap is missing. One call after
one reset is correct and is what this script exercises; a closed loop is not yet expressible through
the public API. Closing it means teaching the facade that a cycle can be multi-phase, which is a
deliberate design decision and not a bug fix, so it is recorded here rather than patched around.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


def step(n, title):
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="/home/ubuntu/hub/lingbot-va",
                    help="local package path, or a Hub repo id once published")
    ap.add_argument("--placement", default="auto", choices=["auto", "in_process", "worker"])
    ap.add_argument("--base-weights", default=None,
                    help="override execution.base_weights, e.g. to the local 23 GB training output "
                         "when the Hub repo is not reachable from this box")
    ap.add_argument("--timeout", type=float, default=1200.0)
    a = ap.parse_args()

    import instinctwm
    from instinctwm import Runtime, describe

    step(1, "describe() -- what it declares, WITHOUT downloading 10.2 GB")
    t0 = time.time()
    d = describe(a.package)
    dt = time.time() - t0
    for k in ("model_id", "backbone", "servable", "nfe", "guidance"):
        print(f"  {k:12} {d[k]}")
    print(f"  {'capabilities':12} {', '.join(d['capabilities'])}")
    print(f"  fetched in {dt:.2f}s")
    check(d["servable"], "declares servable")
    check(d["backbone"] == "wan_va", "declares a backbone", d["backbone"])
    check(not any(w in " ".join(d["capabilities"]).lower()
                  for w in ("recipe", "teacher", "dataset", "distill")),
          "no capability token mentions how it was trained")

    step(2, "the backbone resolves to a registered adapter")
    print(f"  registered: {instinctwm.available_models()}")
    check(d["backbone"] in instinctwm.available_models(), f"{d['backbone']!r} resolves")

    step(3, "Runtime.from_pretrained() -- one call, one handle")
    if a.base_weights:
        # only for running against a package whose base repo is not reachable from this box
        import os
        os.environ["LINGBOT_CKPT"] = a.base_weights
        print(f"  base_weights overridden -> {a.base_weights}")
    t0 = time.time()
    runtime = Runtime.from_pretrained(a.package, placement=a.placement)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print()
    print(runtime.explain())
    check(runtime.model_id == d["model_id"], "handle is for this checkpoint")
    check(runtime.plan is not None, "a plan was compiled from the declared capabilities")

    step(4, "one real inference")
    import numpy as np
    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        print("  no recorded observation available; cannot run a real step")
        FAILED.append("no observation")
        return 1
    z = np.load(ctx[0], allow_pickle=True)
    # The observation schema is the BACKBONE's contract, not InstinctWM's -- `predict` passes the
    # mapping through untouched. wan_va uses the LeRobot camera convention, and the recorded npz
    # stores the short names, so the prefix is applied here rather than assumed on either side.
    # An earlier version of this script fed the short names straight through and got
    # KeyError: 'observation.images.cam_high' from inside the model.
    CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    obs = [{f"observation.images.{c}": z[c] for c in CAMERAS if c in z.files}]
    if not obs[0]:
        print("  recorded context has no camera frames; cannot run a real step")
        FAILED.append("no camera frames")
        return 1
    prompt = str(z["prompt"]) if "prompt" in z.files else "put the bottle in the dustbin"

    try:
        with runtime:
            t0 = time.time()
            runtime.reset(prompt=prompt)
            print(f"  reset in {time.time() - t0:.1f}s   prompt={prompt!r}")
            t0 = time.time()
            out = runtime.predict({"obs": obs, "prompt": prompt, "save_visualization": False})
            dt = time.time() - t0
            # Input-dependence, via a fresh EPISODE rather than a second step. A wan_va control
            # cycle is two server calls -- the action prediction, then a KV-commit that feeds back
            # observed frames plus the executed action -- so calling predict() twice in a row skips
            # the commit, the ring never advances, and the next call dies on a missing third
            # temporal tap. Two clean episodes test the same thing without pretending one call is a
            # whole cycle. See the PROTOCOL note in the module docstring.
            arr2 = None
            if len(ctx) > 1:
                z2 = np.load(ctx[1], allow_pickle=True)
                obs2 = [{f"observation.images.{c}": z2[c] for c in CAMERAS if c in z2.files}]
                p2 = str(z2["prompt"]) if "prompt" in z2.files else prompt
                runtime.reset(prompt=p2)
                o2 = runtime.predict({"obs": obs2, "prompt": p2, "save_visualization": False})
                arr2 = np.asarray(o2["action"] if isinstance(o2, dict) and "action" in o2 else o2)
        act = out["action"] if isinstance(out, dict) and "action" in out else out
        arr = np.asarray(act)
        print(f"  predict in {dt * 1000:.0f} ms")
        print(f"  action {arr.shape} dtype={arr.dtype}")
        print(f"  first row {[round(float(v), 4) for v in arr.reshape(-1)[:6]]} ...")
        print(f"  range [{arr.min():.4f}, {arr.max():.4f}]  std {arr.std():.4f}  "
              f"unique {np.unique(arr).size}/{arr.size}")
        check(arr.size > 0, "an action came back")
        check(bool(np.isfinite(arr).all()), "and it is finite")

        # "an action came back" is NOT "the model ran". A constant tensor is finite, correctly
        # shaped, and completely meaningless -- and the first run printed six identical leading
        # values, which is exactly what a degenerate output looks like. Two gates close that:
        check(float(arr.std()) > 1e-6, "the action VARIES -- not a constant tensor",
              f"std={arr.std():.5f}")

        # ... and the output must actually depend on the input. Second observation, same episode:
        # if the model is genuinely conditioning on pixels these cannot be equal.
        if arr2 is not None:
            d = float(np.abs(arr - arr2).max())
            check(d > 1e-6, "a DIFFERENT episode gives a different action", f"max|Δ|={d:.5f}")
        else:
            print("  only one recorded context; input-dependence not checked")
    except Exception as e:                                     # noqa: BLE001
        print(f"  RAISED {type(e).__name__}: {e}")
        FAILED.append(f"inference raised: {type(e).__name__}")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: describe -> from_pretrained -> reset -> predict, on the real 10.2 GB package,")
    print("      through the public API only. Transport never appeared in it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
