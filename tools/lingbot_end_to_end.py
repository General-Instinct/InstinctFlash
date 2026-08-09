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

STATUS OF THE FINAL STEP, 2026-08-09. Steps 1-3 pass on the real 10.2 GB package and the real weights
LOAD from it -- the loader reports reading `.instinctwm_composed/transformer`. Step 4 is
NOT EVALUATED, not failing: every GPU on this box is held by an unrelated `lerobot-train` job
(~58 GB of 80 GB each, 2h47m elapsed and counting), and the model needs

    transformer 10.18 + vae 0.51 + text_encoder 11.36 = 22.05 GB of weights resident

against ~23.4 GB free, leaving 1.35 GB for activations, CUDA context and the KV pool. Two attempts
OOMed at 22.3 GB, one of them with expandable_segments, so this is capacity and not fragmentation.
Killing someone else's training run to claim a green tick would be the wrong trade.

Re-run it on a GPU with >= ~30 GB free; an idle H100-80GB is ample:

    CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=$IWM_FA_SHIM_DIR:. MASTER_ADDR=127.0.0.1 \
      MASTER_PORT=29854 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 $IWM_SERVER_PY \
      tools/lingbot_end_to_end.py --package /home/ubuntu/hub/lingbot-va \
      --base-weights /home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin
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
    import json as _json
    cfg_keys = _json.loads((Path(a.package) / "instinctwm.json").read_text())
    del cfg_keys
    # the observation as the served path expects it: one frame per camera, plus the prompt
    cams = [k for k in z.files if k.endswith(("cam_high", "cam_left_wrist", "cam_right_wrist"))]
    short = {c.split(".")[-1]: c for c in cams}
    obs = [{c: z[s] for s, c in short.items()}] if short else None
    prompt = str(z["prompt"]) if "prompt" in z.files else "put the bottle in the dustbin"

    try:
        with runtime:
            t0 = time.time()
            runtime.reset(prompt=prompt)
            print(f"  reset in {time.time() - t0:.1f}s   prompt={prompt!r}")
            t0 = time.time()
            out = runtime.predict({"obs": obs, "prompt": prompt, "save_visualization": False})
            dt = time.time() - t0
        act = out["action"] if isinstance(out, dict) and "action" in out else out
        arr = np.asarray(act)
        print(f"  predict in {dt * 1000:.0f} ms")
        print(f"  action {arr.shape} dtype={arr.dtype}")
        print(f"  first row {[round(float(v), 4) for v in arr.reshape(-1)[:6]]} ...")
        check(arr.size > 0, "an action came back")
        check(bool(np.isfinite(arr).all()), "and it is finite")
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
