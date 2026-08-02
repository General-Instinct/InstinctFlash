#!/usr/bin/env python3
"""Measure the batch-1 closed-loop cost model of a LingBot-VA policy server.

This is the latency counterpart to `check_prompt_parity.py`: that one says the server is
*correct*, this one says what it *costs*. Every InstinctWM optimization is scored against
the numbers this produces, so it replays the real episode message sequence rather than
timing an isolated forward pass:

    reset  ->  infer  ->  compute_kv_cache  ->  infer  ->  compute_kv_cache  ->  ...

That order matters. The server is stateful: `infer` only VAE-encodes the observation on
the very first cycle (`wan_va_server.py:445-447`, `if frame_st_id == 0`), and thereafter
runs purely off the KV cache, while `compute_kv_cache` folds the real observed frames back
in. Timing `infer` in isolation would measure a cycle that never occurs in an episode.

Attention runs over the *valid KV population*, which grows every cycle until the pool
saturates, so per-cycle latency is NOT flat. `--cycles` should be large enough to show the
ramp; the report prints per-cycle timings, not just an average.

Measurement hygiene (learned from a previous sweep that produced a flat curve because a
dead server silently served every point):
  * the server must be otherwise idle -- this refuses to run if other clients are attached
    is not detectable, so CHECK YOURSELF that no eval is running;
  * cycle 0 is reported separately and excluded from steady-state stats, because it pays
    T5, the VAE encode and cold kernels;
  * p99 is reported alongside the mean, because a control loop is judged at its tail.

Usage:
    python probe_latency.py --port 29056 --cycles 12
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

# so `evaluation.robotwin...` resolves; env.sh exports ROBOTWIN_ROOT.
sys.path.insert(0, os.environ.get("ROBOTWIN_ROOT", "/home/ubuntu/RoboTwin"))

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

CAMS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def make_obs(rng, h=240, w=320):
    """One RoboTwin-shaped observation frame. Content is irrelevant to cost: the server
    resizes every camera to a fixed latent grid, so shape -- not pixels -- sets the work."""
    return {k: rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for k in CAMS}


def pct(xs, q):
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs), q))


def _one_run(cli, args) -> float:
    """One full probe run; returns the steady-state full-cycle mean in ms."""
    rng = np.random.default_rng(0)
    cli.infer(dict(reset=True, prompt=args.prompt, save_visualization=False))
    first_obs = make_obs(rng)
    cycle_ms = []
    for c in range(args.cycles):
        t = time.perf_counter()
        ret = cli.infer(dict(obs=first_obs, prompt=args.prompt, save_visualization=False))
        d_infer = (time.perf_counter() - t) * 1000
        action = ret["action"]
        nkf = args.keyframes // 2 if c == 0 else args.keyframes
        kfs = [make_obs(rng) for _ in range(nkf)]
        t = time.perf_counter()
        cli.infer(dict(obs=kfs, compute_kv_cache=True, imagine=False,
                       save_visualization=False, state=action))
        d_kv = (time.perf_counter() - t) * 1000
        cycle_ms.append(d_infer + d_kv)
    ss = cycle_ms[1:]                      # cycle 0 is cold: T5 + VAE encode + first kernels
    return statistics.mean(ss) if ss else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=29056)
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--prompt", default="Use the left arm to lift the plastic drink bottle head-up")
    ap.add_argument("--action-dim", type=int, default=16)
    ap.add_argument("--frame-chunk", type=int, default=2)
    ap.add_argument("--action-per-frame", type=int, default=16)
    ap.add_argument("--keyframes", type=int, default=8, help="real frames sent per compute_kv_cache")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Full probe runs. The FIRST is discarded: measured on this box, the first "
                         "run after a server starts is up to 37%% slower than steady state "
                         "(cuBLAS/cuDNN algorithm selection, allocator warm-up), and probe_latency "
                         "previously discarded only cycle 0, not the first run. Reporting a single "
                         "run silently mixes warm-up into the number -- it is how a 1.39x was first "
                         "recorded as 1.44x, and how the same config measured 2556 and 3503 ms.")
    args = ap.parse_args()

    cli = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"connected to ws://{args.host}:{args.port}")

    if args.repeats > 1:
        run_means = []
        for rep in range(args.repeats):
            m = _one_run(cli, args)
            tag = "DISCARDED (warm-up)" if rep == 0 else "kept"
            print(f"  run {rep}: full-cycle mean {m:8.1f} ms   [{tag}]")
            if rep > 0:
                run_means.append(m)
        lo, hi = min(run_means), max(run_means)
        spread = 100 * (hi - lo) / lo if lo else 0.0
        print(f"\nsteady-state mean over {len(run_means)} kept runs: "
              f"{sum(run_means)/len(run_means):8.1f} ms  (spread {spread:.1f}%)")
        if spread > 5.0:
            print("  WARNING: kept runs disagree by more than 5%. Do not quote this number; "
                  "the box is not in steady state.")
        return 0

    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    cli.infer(dict(reset=True, prompt=args.prompt, save_visualization=False))
    t_reset = time.perf_counter() - t0
    print(f"reset (includes T5 encode of the instruction): {t_reset*1000:8.1f} ms")

    first_obs = make_obs(rng)

    infer_ms, kv_ms, cycle_ms = [], [], []
    for c in range(args.cycles):
        t = time.perf_counter()
        ret = cli.infer(dict(obs=first_obs, prompt=args.prompt, save_visualization=False))
        d_infer = (time.perf_counter() - t) * 1000
        action = ret["action"]

        nkf = args.keyframes // 2 if c == 0 else args.keyframes
        kfs = [make_obs(rng) for _ in range(nkf)]
        t = time.perf_counter()
        cli.infer(dict(obs=kfs, compute_kv_cache=True, imagine=False,
                       save_visualization=False, state=action))
        d_kv = (time.perf_counter() - t) * 1000

        infer_ms.append(d_infer)
        kv_ms.append(d_kv)
        cycle_ms.append(d_infer + d_kv)
        print(f"  cycle {c:2d}  infer {d_infer:8.1f} ms   kv {d_kv:7.1f} ms   "
              f"total {d_infer+d_kv:8.1f} ms   action{tuple(action.shape)}")

    n_actions = args.frame_chunk * args.action_per_frame
    print()
    print("=" * 74)
    print(f"action chunk = {n_actions} control steps per cycle")
    print(f"denoise forwards per cycle = 26 video + 51 action = 77, each at batch 2 (CFG)")
    print()
    ss_i, ss_k, ss_c = infer_ms[1:], kv_ms[1:], cycle_ms[1:]
    if ss_c:
        print(f"{'':22s} {'mean':>10s} {'p50':>10s} {'p99':>10s} {'min':>10s} {'max':>10s}")
        for name, xs in (("infer (denoise)", ss_i), ("compute_kv_cache", ss_k), ("full cycle", ss_c)):
            print(f"{name:22s} {statistics.mean(xs):10.1f} {pct(xs,50):10.1f} "
                  f"{pct(xs,99):10.1f} {min(xs):10.1f} {max(xs):10.1f}")
        print()
        print(f"cycle 0 (cold: T5 + VAE encode + first kernels): "
              f"infer {infer_ms[0]:.1f} ms, kv {kv_ms[0]:.1f} ms")
        mean_cycle = statistics.mean(ss_c)
        print(f"steady-state per-control-step cost = {mean_cycle/n_actions:.1f} ms")
        print(f"implied sustainable control rate   = {1000.0*n_actions/mean_cycle:.1f} Hz")
        print()
        # The ramp is the signal that attention is over a growing KV population.
        if len(ss_c) >= 4:
            first_half = statistics.mean(ss_c[: len(ss_c) // 2])
            second_half = statistics.mean(ss_c[len(ss_c) // 2 :])
            print(f"ramp: first half {first_half:.1f} ms -> second half {second_half:.1f} ms "
                  f"({100*(second_half/first_half-1):+.1f}%)  "
                  f"[KV population grows each cycle until the pool saturates]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
