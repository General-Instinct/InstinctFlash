#!/usr/bin/env python3
"""The measurement the run is missing: per-head error at INITIALISATION, and target magnitudes.

WHY THIS IS NEEDED. The trainer reports raw per-head MSE, which is not interpretable on its own and
not comparable across sigma. Two facts are missing to read it:

  1. WHAT IT WAS AT INIT. Every head starts as a copy of `proj_out`, so it predicts the teacher's
     INSTANTANEOUS velocity while the target is the MEAN velocity over the interval. The gap grows
     with interval width, so init error is structurally uneven -- and without measuring it there is
     no way to tell a head that started bad and stayed bad from one that started good and got worse.
  2. HOW BIG THE TARGET IS. Velocity magnitude grows toward sigma=1, so an MSE of 1.7 near pure noise
     and 0.2 near clean data may be the same RELATIVE error, or may not. Reporting `err / E[|target|^2]`
     settles it.

An earlier baseline taken from the smoke run is NOT usable for this: it predates the fp32 ODE-state
fix, used a per-rank unseeded x0, and averaged one context. It is quoted nowhere.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
      -m torch.distributed.run --nproc_per_node 1 --master_port 29960 \\
      probe_pdd_baseline.py --contexts /home/ubuntu/iwm_results/pdd_ctx50 \\
      [--heads /home/ubuntu/iwm_results/pdd_heads_run1/final/heads.pt]

With --heads it evaluates a trained checkpoint under the identical protocol, so init and trained are
directly comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctwm.adapters.lingbot_velocity import LingBotChunk0Video  # noqa: E402
from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_fsdp_elision,
)
from instinct_pdd import advance, sample  # noqa: E402
from instinct_pdd.solvers import get_solver  # noqa: E402

from train_pdd_heads import load_context_files, read_context  # noqa: E402


@torch.no_grad()
def measure(adapter, student, teacher, grid, ctx, solver, seed):
    """Per-head squared error AND per-head target energy, on one context."""
    N, L = grid.n_intervals, grid.block
    est = get_solver(solver)
    err = torch.zeros(N)
    tgt = torch.zeros(N)
    g = torch.Generator(device=adapter.S.device).manual_seed(seed)
    x_n = torch.randn(adapter.state_shape(), generator=g,
                      dtype=adapter.STATE_DTYPE, device=adapter.S.device)
    for n in range(0, N, L):
        heads = student.heads(x_n, grid.cond(n), cond=ctx)
        for k in range(n, min(n + L, N)):
            x_k = advance(x_n, heads, grid, n, k)
            t = est(teacher, x_k, grid, k, ctx)
            err[k] = float(torch.nn.functional.mse_loss(heads[k], t))
            tgt[k] = float(t.float().pow(2).mean())
        x_n = advance(x_n, heads, grid, n, min(n + L, N))
    return err, tgt


@torch.no_grad()
def endpoint(adapter, student, teacher, grid, ctx, solver, seed):
    """Student at NFE=2 vs the teacher integrated over all N intervals, from identical noise.

    At INIT this is exactly the naive 2-step sampler: every head is a copy of proj_out evaluated at
    the block start, so the block step composes 128 identical vectors and
    sum_l h_l * v(x_n, t_n) = v(x_n, t_n) * (t_{n+L} - t_n) -- one Euler step per block.
    """
    est = get_solver(solver)
    g = torch.Generator(device=adapter.S.device).manual_seed(seed)
    x0 = torch.randn(adapter.state_shape(), generator=g,
                     dtype=adapter.STATE_DTYPE, device=adapter.S.device)
    x = x0
    for k in range(grid.n_intervals):
        x = x + est(teacher, x, grid, k, ctx) * grid.h(k)
    ref = x
    stu = sample(student, x0, grid, cond=ctx)
    return float((stu - ref).pow(2).mean().sqrt()), float(ref.pow(2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--heads", default=None, help="trained heads.pt; omit for the init baseline")
    ap.add_argument("--n-intervals", type=int, default=256)
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--n-contexts", type=int, default=3)
    ap.add_argument("--heldout-from-ep", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_pdd_baseline"
    os.makedirs(cfg.save_root, exist_ok=True)
    S.init_distributed(1, 0, 0)
    cfg.rank, cfg.local_rank, cfg.world_size = 0, 0, 1
    install_fsdp_elision(S)
    server = S.VA_Server(cfg)

    adapter = LingBotChunk0Video(server, guidance=5.0)
    grid = adapter.grid(a.n_intervals, a.n_intervals // a.nfe)
    student = adapter.student(n_heads=grid.n_intervals)
    label = "INIT (every head = a copy of proj_out)"
    if a.heads:
        student.load_state_dict(torch.load(a.heads, map_location=server.device))
        label = f"TRAINED ({a.heads})"

    _, held = load_context_files(a.contexts, a.heldout_from_ep)
    stride = max(1, len(held) // max(1, a.n_contexts))
    chosen = held[::stride][:a.n_contexts]

    E = torch.zeros(grid.n_intervals)
    T = torch.zeros(grid.n_intervals)
    EPS, SCS = [], []
    for i, f in enumerate(chosen):
        obs, prompt, task = read_context(f, cfg.obs_cam_keys)
        ctx = adapter.encode_context(obs, prompt=prompt, task=task)
        seed = 20260804 + 1000 * i
        e, t = measure(adapter, student, teacher_of(adapter), grid, ctx, "euler", seed)
        E += e
        T += t
        ep, sc = endpoint(adapter, student, teacher_of(adapter), grid, ctx, "euler", seed)
        EPS.append(ep)
        SCS.append(sc)
        print(f"  {task}: endpoint {ep:.4f} / scale {sc:.4f} = {100*ep/sc:.1f}%", flush=True)
    E /= len(chosen)
    T /= len(chosen)

    import statistics
    print(f"\n{'=' * 74}\n{label}   ({len(chosen)} held-out contexts)")
    print(f"  ENDPOINT RMSE {statistics.fmean(EPS):.4f}  scale {statistics.fmean(SCS):.4f}  "
          f"= {100*statistics.fmean(EPS)/statistics.fmean(SCS):.1f}% of scale")
    print(f"  {'heads':>10} {'sigma':>15} {'MSE':>10} {'E|target|^2':>12} {'rel err':>9} {'h':>9}")
    for b in range(8):
        lo, hi = b * 32, (b + 1) * 32
        mse = float(E[lo:hi].mean())
        te = float(T[lo:hi].mean())
        h = sum(grid.h(k) for k in range(lo, hi)) / 32
        print(f"  {f'{lo}-{hi-1}':>10} {f'{grid.cond(lo)/1000:.3f}-{grid.cond(hi-1)/1000:.3f}':>15} "
              f"{mse:>10.4f} {te:>12.4f} {100*(mse/max(te,1e-9))**0.5:>8.1f}% {h:>9.5f}")

    out = a.out or f"/tmp/pdd_baseline_{'trained' if a.heads else 'init'}.json"
    Path(out).write_text(json.dumps({
        "label": label, "per_head_error": E.tolist(), "per_head_target_energy": T.tolist(),
        "endpoint_rmse": sum(EPS)/len(EPS), "endpoint_scale": sum(SCS)/len(SCS),
        "widths": [grid.h(k) for k in range(grid.n_intervals)],
        "sigma": [grid.cond(k) / 1000.0 for k in range(grid.n_intervals)],
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


def teacher_of(adapter):
    return adapter.teacher()


if __name__ == "__main__":
    raise SystemExit(main())
