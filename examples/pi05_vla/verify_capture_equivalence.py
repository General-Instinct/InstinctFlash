#!/usr/bin/env python3
"""Paired check that graph capture changes pi05's speed and not its actions.

    HF_TOKEN=... python examples/pi05_vla/verify_capture_equivalence.py

THREE ARMS, each in its own process, on the same GPU, from the same declaration:

    baseline   Pi05Adapter.install neutered, so the model runs exactly as lerobot ships it
    control    byte-identical to baseline -- a second run of the same arm
    captured   the shipped path: constants hoisted, denoise step captured and replayed

WHY THE CONTROL ARM EXISTS. The first version of this script had two arms and reported
`max |d action| = 2.143e+00` against an action scale of 0.51 -- which reads as a catastrophically
broken capture and was in fact a broken HARNESS. pi05 is a flow-matching policy: `sample_actions`
draws the initial noise from the global RNG, so two processes that do not seed produce different
actions whatever else is true of them. The arms are seeded now, and baseline-vs-control proves this
comparison can register equality at all before baseline-vs-captured is allowed to mean anything. A
harness that cannot detect a null result cannot detect a real one either.

Both arms drive `Runtime.episode().predict()` over an identical fixed observation sequence and dump
every action. The comparison is then bytes, not eyeballs.

WHY SEPARATE PROCESSES. The hoist and the install both patch the CLASS (`type(model).embed_suffix`,
`type(model).denoise_step`), so two Runtimes in one interpreter would share them and the "baseline" arm
would silently be the captured arm. That is the shape of the two harness errors this project has
already made -- a reference arm that was not the reference -- so the arms are isolated by process.

WHY THE TIMING IS PAIRED TOO. The first chunk-cost number came off a different GPU on a different day.
Both arms run here, on the same device, back to back.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "lerobot/pi05_base"
PROMPT = "Put the exhaust fans back to the slots."
CALLS = 205                     # spans four chunk boundaries at chunk_size 50


def declare(into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    (into / "config.json").write_text("{}")
    (into / "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": "example-org/pi05", "backbone": "pi05", "servable": True,
                      "guidance": {"action": "none"}, "nfe": {"action": 10, "prefix": 1},
                      "base_weights": BASE}}))
    return into


ARM = r'''
import json, statistics, sys, time
from pathlib import Path
import numpy as np
sys.path[:0] = [%(root)r, %(here)r]
import instinctflash
from pi05_iwm.adapter import Pi05Adapter

ARM, PKG, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])

# SEED FIRST. pi05 samples its initial flow-matching noise from the global RNG, so without this the
# arms differ by the noise draw and every comparison below is meaningless.
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

if ARM in ("baseline", "control"):
    # Neuter the install so the model runs as lerobot ships it. Done BEFORE from_pretrained, and it
    # returns [] exactly like a machine where the plan declined -- not a different code path.
    Pi05Adapter.install = staticmethod(lambda policy, plan, *, device=None: [])

instinctflash.register("pi05", Pi05Adapter)
rt = instinctflash.Runtime.from_pretrained(PKG)
obs = rt.observation.example()
obs["prompt"] = %(prompt)r

acts, times = [], []
with rt, rt.episode(prompt=obs["prompt"]) as ep:
    for _ in range(%(calls)d):
        t0 = time.perf_counter()
        out = ep.predict(obs)
        times.append((time.perf_counter() - t0) * 1000)
        acts.append(np.asarray(out["action"] if isinstance(out, dict) else out, dtype=np.float64))

A = np.stack(acts)
np.save(OUT.with_suffix(".npy"), A)
slow = [(i, t) for i, t in enumerate(times) if t > 50]
fast = [t for t in times if t <= 50]
# skip the first TWO boundaries: the first carries warm-up, and for the captured arm the second
# carries the capture plus its startup-only bit-exact self-check (seconds, by design)
chunks = [t for _, t in slow[2:]] or [t for _, t in slow]
OUT.write_text(json.dumps({
    "arm": ARM, "chunk_ms": statistics.median(chunks), "dequeue_ms": statistics.median(fast),
    "boundaries": [i + 1 for i, _ in slow], "first_ms": times[0],
    "action_shape": list(A.shape)}))
print(f"  {ARM:9} chunk {statistics.median(chunks):6.1f} ms   boundaries at "
      f"{[i + 1 for i, _ in slow]}", flush=True)
'''


def main() -> int:
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        pkg = declare(Path(td) / "pi05")
        src = Path(td) / "arm.py"
        src.write_text(ARM % {"root": str(ROOT), "here": str(HERE), "prompt": PROMPT,
                              "calls": CALLS})
        results = {}
        print(f"{CALLS} predict() calls per arm, same GPU, separate processes, seeded:")
        for arm in ("baseline", "control", "captured"):
            out = Path(td) / arm
            t0 = time.time()
            p = subprocess.run([sys.executable, str(src), arm, str(pkg), str(out)],
                               env={**os.environ}, capture_output=True, text=True)
            if p.returncode != 0:
                print(f"  {arm}: FAILED\n{p.stdout[-1500:]}\n{p.stderr[-2500:]}")
                return 1
            print(p.stdout.rstrip() + f"   ({time.time() - t0:.0f}s)")
            results[arm] = (json.loads(out.read_text()), np.load(out.with_suffix(".npy")))

    (b, ba), (k, ka), (c, ca) = (results["baseline"], results["control"], results["captured"])
    d = float(np.abs(ba - ca).max())
    d_null = float(np.abs(ba - ka).max())
    speedup = b["chunk_ms"] / c["chunk_ms"]

    print(f"\ncontrol           max |d| baseline vs baseline = {d_null:.3e}   "
          f"{'harness can detect equality' if d_null == 0.0 else 'HARNESS IS NOT REPRODUCIBLE'}")

    print(f"\nchunk inference   {b['chunk_ms']:.1f} ms -> {c['chunk_ms']:.1f} ms   {speedup:.2f}x")
    print(f"amortised         {b['chunk_ms'] / 50:.1f} -> {c['chunk_ms'] / 50:.1f} ms/step "
          f"({1000 * 50 / b['chunk_ms']:.0f} -> {1000 * 50 / c['chunk_ms']:.0f} Hz equivalent)")
    print(f"dequeue           {b['dequeue_ms']:.2f} ms -> {c['dequeue_ms']:.2f} ms  (unchanged: "
          f"capture is on the chunk, not the queue)")
    print(f"actions           {ba.shape} vs {ca.shape}")
    print(f"max |d action|    {d:.3e}   {'BITEXACT' if d == 0.0 else 'NOT BIT-EXACT'}")

    # PER CHUNK, because "which chunk first differs" separates two very different diagnoses. If chunk
    # 1 matches and later chunks do not, replay is fine and something perturbed the RNG stream that
    # the NEXT chunk's initial noise is drawn from -- capture runs warm-up forwards that the baseline
    # never runs. If chunk 1 already differs, replay itself is wrong.
    for c0 in range(0, ba.shape[0], 50):
        seg = slice(c0, min(c0 + 50, ba.shape[0]))
        print(f"    calls {c0 + 1:3}-{min(c0 + 50, ba.shape[0]):3}   "
              f"max |d| {float(np.abs(ba[seg] - ca[seg]).max()):.3e}")

    ok = True
    if d_null != 0.0:
        print(f"  FAIL  the two identical arms disagree by {d_null:.3e}, so this harness cannot "
              f"establish equivalence and the number below means nothing. Fix the harness first.")
        return 1
    if b["boundaries"] != c["boundaries"]:
        print(f"  FAIL  chunk boundaries moved: {b['boundaries']} vs {c['boundaries']}")
        ok = False
    if d != 0.0:
        print("  FAIL  capture is declared BITEXACT and the actions differ. Either the tier is wrong "
              "or the capture is. Do not ship this until one of them changes.")
        ok = False
    if speedup < 1.05:
        print(f"  note  {speedup:.2f}x -- no win, which is NOT the shipped state anymore: the "
              f"static-KV capture is the pi05 default and should collect ~2.8x here. Either "
              f"IFL_PI05_NO_CAPTURE=1 is set, or the runtime self-check rejected the capture "
              f"(the captured arm's log says which, loudly). What this script asserts is still "
              f"the thing that matters: upstream's actions, byte for byte.")
    print("\n" + ("PASS: same actions, byte for byte, at " f"{speedup:.2f}x." if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
