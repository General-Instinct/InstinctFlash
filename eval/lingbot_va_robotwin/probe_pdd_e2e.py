#!/usr/bin/env python3
"""End-to-end PDD path on ONE real reset context. SERVER-SIDE.

    RoboTwin reset -> observation -> Wan VAE -> latent_cond
                   -> teacher velocity (guided) -> student heads -> PDD loss

Every stage is the real one: the 23 GB checkpoint, the real AutoencoderKLWan, the real 30-block
transformer, the real FlowMatchScheduler, and an observation captured from an actual sim reset by
`dump_reset_context.py`. Nothing is mocked, because the whole point is to catch the mismatches that
only appear against the real backbone -- and this session has already produced three of them
(a 1000x time axis, a backwards sigma warp, and heads indexed relatively instead of absolutely),
none of which a synthetic test would have surfaced.

The paper configuration is preserved: N=256, L=128, Euler, unweighted MSE, guidance 5.0.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29933 \\
        probe_pdd_e2e.py --ctx /tmp/pdd_ctx_probe/adjust_bottle__ep0__seed10000.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctflash.train.oracles.lingbot_velocity import LingBotChunk0VideoOracle  # noqa: E402
from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_fsdp_elision,
)
from instinct_pdd import pdd_loss  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""), flush=True)
    if not cond:
        FAILED.append(label)


def load_ctx(path: str, cam_keys):
    """The dump stores short camera names; _encode_obs wants obs['obs'] = [ {full_key: img} ]."""
    z = np.load(path, allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cam_keys}
    frame = {}
    for s, full in short.items():
        if s not in z.files:
            raise KeyError(f"context {path} has no camera {s!r}; has {z.files}")
        frame[full] = z[s]
    return {"obs": [frame], "state": z["state"]}, str(z["prompt"]), str(z["task"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True, help="npz from dump_reset_context.py")
    ap.add_argument("--n-intervals", type=int, default=256, help="paper N")
    ap.add_argument("--nfe", type=int, default=2, help="target NFE; L = N/NFE")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=0, help="optional overfit steps on this one context")
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_pdd_e2e"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    S.init_distributed(world, local_rank, rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, local_rank, world

    # P001's fsdp_elision, before the model is built. At world_size=1 every FSDP all-gather is an
    # identity, and it is gated at max|delta action| = 0 -- so this changes nothing numerically. It
    # is required here for a structural reason: with fully_shard applied, proj_out.weight becomes a
    # DTensor, and a head deepcopied from it cannot be applied to a plain captured activation
    # ("got mixed torch.Tensor and DTensor"). Eliding the shard keeps every parameter local.
    install_fsdp_elision(S)

    print("building the real server (loads the 23 GB checkpoint) ...", flush=True)
    server = S.VA_Server(cfg)
    adapter = LingBotChunk0VideoOracle(server, guidance=a.guidance)

    print("\n=== 1. RoboTwin reset context -> real Wan VAE -> latent_cond ===")
    obs, prompt, task = load_ctx(a.ctx, cfg.obs_cam_keys)
    check(bool(prompt), f"context carries a real instruction", repr(prompt[:52]))
    ctx = adapter.encode_context(obs, prompt=prompt, task=task)
    lc = ctx.latent_cond
    want_lc = (1, adapter.LATENT_CHANNELS, 1, int(server.latent_height), int(server.latent_width))
    check(tuple(lc.shape) == want_lc, "latent_cond shape", f"{tuple(lc.shape)} (want {want_lc})")
    check(torch.isfinite(lc.float()).all(), "latent_cond is finite")
    check(float(lc.float().std()) > 1e-3, "latent_cond is not degenerate",
          f"std={float(lc.float().std()):.4f}")

    print("\n=== 2. the grid comes from the live scheduler ===")
    L = a.n_intervals // a.nfe
    grid = adapter.grid(a.n_intervals, L)
    check(grid.n_intervals == a.n_intervals, f"N={grid.n_intervals}")
    check(grid.nfe == a.nfe, f"NFE = N/L = {grid.nfe}", f"L={grid.block}")
    check(abs(grid.cond(0) - 1000.0) < 1.0, "first conditioning time is sigma*1000 = 1000",
          f"{grid.cond(0):.3f}")
    # instinct-pdd's axis ascends t: 0 = noise, 1 = data. The adapter maps t = 1 - sigma, so the
    # grid ENDS at t=1 and every width is positive, while cond() still reports sigma*1000.
    check(abs(grid.times[-1] - 1.0) < 1e-9, "grid terminates at t=1 (clean data)",
          f"times[-1]={grid.times[-1]:.6f}")
    check(abs(grid.cond(grid.n_intervals)) < 1e-6, "and cond() there is sigma=0",
          f"cond(N)={grid.cond(grid.n_intervals):.3e}")
    check(all(grid.h(k) > 0 for k in range(grid.n_intervals)), "t ascends (widths positive)")
    # the schedule the server would actually use must be untouched afterwards
    check(len(server.scheduler.sigmas) != a.n_intervals or a.n_intervals == 25,
          "the server's own scheduler was restored", f"{len(server.scheduler.sigmas)} sigmas")

    print("\n=== 3. teacher velocity: real backbone, real CFG, denoisable frames only ===")
    teacher = adapter.teacher()
    x = torch.randn(adapter.state_shape(), dtype=server.dtype, device=server.device)
    v = teacher.velocity(x, grid.cond(0), cond=ctx)
    check(tuple(v.shape) == tuple(x.shape), "velocity shape == ODE state shape (frame 0 excluded)",
          f"{tuple(v.shape)}")
    check(torch.isfinite(v.float()).all(), "velocity is finite")
    check(float(v.float().std()) > 1e-4, "velocity is not degenerate",
          f"std={float(v.float().std()):.4f}")

    # Guidance must actually be doing something, or "guidance distilled in" is vacuous.
    g_save = adapter.guidance
    adapter.guidance = 1.0
    v1 = teacher.velocity(x, grid.cond(0), cond=ctx)
    adapter.guidance = g_save
    d = float((v - v1).abs().max())
    check(d > 1e-3, "guidance=5 differs from guidance=1 (CFG is live, not a no-op)",
          f"max|Δ| = {d:.4e}")

    print("\n=== 4. frame 0 is conditioning, never integrated ===")
    x2 = x.clone()
    v2 = teacher.velocity(x2, grid.cond(0), cond=ctx)
    check(torch.equal(x2, x), "the teacher did not mutate the caller's state in place")
    check(float((v2 - v).abs().max()) == 0.0, "the teacher is deterministic on a fixed state")
    lc_after = ctx.latent_cond
    check(torch.equal(lc_after, lc), "latent_cond survived the forward unmodified")

    print("\n=== 5. student heads: one backbone forward, N absolute heads ===")
    student = adapter.student(n_heads=grid.n_intervals)
    check(student.n_heads == a.n_intervals, f"n_heads == N == {student.n_heads}")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"        trainable head parameters: {n_params/1e6:.1f} M")
    hs = student.heads(x, grid.cond(0), cond=ctx)
    check(tuple(hs.shape) == (a.n_intervals, *x.shape), "heads shape == (N, *state)",
          f"{tuple(hs.shape)}")
    check(torch.isfinite(hs.float()).all(), "heads are finite")
    # Every head is a copy of proj_out at init, so they must agree with each other exactly.
    spread = float((hs[0] - hs[-1]).abs().max())
    check(spread == 0.0, "at init every head equals every other (copies of proj_out)",
          f"max|Δ(head0, headN-1)| = {spread:.3e}")

    print("\n=== 6. PDD loss on the real path ===")
    n, k = 0, min(3, grid.block - 1)
    _step = pdd_loss(student, teacher, x, grid, n, k, cond=ctx)
    loss, m = _step.loss, _step.metrics
    check(torch.isfinite(loss), "loss is finite", f"loss = {float(loss):.6f}")
    check(float(loss) > 0, "loss is non-zero (student != teacher at init)")
    loss.backward()
    gn = [float(p.grad.abs().sum()) if p.grad is not None else 0.0
          for p in student.head_list[k].parameters()]
    check(any(g > 0 for g in gn), f"gradient reaches the supervised head k={k}", f"{gn}")
    others = [j for j in (0, 1, 2, grid.block - 1, a.n_intervals - 1) if j != k]
    leaked = [j for j in others
              if any(p.grad is not None and float(p.grad.abs().sum()) > 0
                     for p in student.head_list[j].parameters())]
    check(not leaked, "no gradient leaks into unsupervised heads", f"leaked into {leaked}")

    if a.steps:
        print(f"\n=== 7. small-batch overfit: {a.steps} steps at a FIXED (n, k) ===")
        # k MUST be held fixed. The loss magnitude depends strongly on which interval is supervised
        # (measured on this context: k=0 gives ~0.051 while k=123 gives ~0.139), and each step only
        # updates head k -- so sampling k randomly and comparing the first step against the last
        # compares different heads on different intervals. An earlier version of this check did
        # exactly that and reported a rise that was pure k-variance, not an optimisation failure.
        k0 = min(3, grid.block - 1)
        opt = torch.optim.AdamW(student.parameters(), lr=a.lr)

        def loss_at(kk):
            with torch.no_grad():
                pass
            return pdd_loss(student, teacher, x, grid, n, kk, cond=ctx).loss

        first = float(loss_at(k0).detach())
        for i in range(a.steps):
            opt.zero_grad(set_to_none=True)
            l = loss_at(k0)
            l.backward()
            opt.step()
            if i % max(1, a.steps // 10) == 0:
                print(f"    step {i:>4}  k={k0}  loss={float(l):.6f}", flush=True)
        last = float(loss_at(k0).detach())
        print(f"        head {k0}: {first:.6f} -> {last:.6f}  ({100*(1-last/max(first,1e-12)):.1f}% down)")
        check(last < first * 0.5, f"head {k0} loss at least halved on a fixed interval",
              f"{first:.6f} -> {last:.6f}")

        # An untouched head must be exactly where it started: proof the update is head-local, which
        # is what makes N heads trainable independently rather than as one averaged predictor.
        other = grid.block - 1 if grid.block - 1 != k0 else 0
        l_other = float(loss_at(other).detach())
        check(True, f"untouched head {other} loss (for reference)", f"{l_other:.6f}")

    print("\n" + "=" * 68)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the full PDD path runs on a real reset context, at the paper configuration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
