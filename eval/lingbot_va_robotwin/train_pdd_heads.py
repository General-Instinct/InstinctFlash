#!/usr/bin/env python3
"""Heads-only PDD on LingBot-VA chunk-0 video. SERVER-SIDE, one rank per GPU.

THE QUESTION THIS RUN ANSWERS, and it is deliberately narrow: can 256 repeated output heads absorb a
2-step PDD target while the LingBot trunk stays frozen? Not "what is the best achievable accuracy".
If heads-only works it is a strong result; if it plateaus we have a measured reason to unfreeze the
backbone rather than paying for full fine-tuning by default.

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
      -m torch.distributed.run --nproc_per_node 8 --master_port 29950 \\
      train_pdd_heads.py --contexts /home/ubuntu/iwm_results/pdd_ctx50 --steps 20000

WHAT IS TRAINABLE: 256 copies of `proj_out` (151.0 M parameters). Everything else -- 30 blocks, the
VAE, the text encoder -- is frozen, and the trunk runs under `no_grad`, so no activation graph is
built for it. That is what makes this cheap enough to be a first experiment.

HOW THE RANKS COOPERATE. Each rank draws its OWN (context, n, k), so eight different heads are
supervised per optimiser step and coverage advances eight times faster. Gradients are therefore
naturally disjoint, and only the union of touched heads is all-reduced -- 8 heads of 0.6 M parameters
instead of all 256, which is a 19 MB exchange rather than 604 MB. Each head's summed gradient is
divided by the number of ranks that touched it, so a head's effective learning rate does not depend
on how many ranks happened to pick it that step.

HEAD COVERAGE IS A HARD GATE, not a metric. Each step supervises exactly ONE head out of 256, so a
run that touched 60 of them still shows a falling loss and still writes a checkpoint that samples
badly. `--min-updates-per-head` is enforced before anything is marked usable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from instinctflash.train.oracles.lingbot_velocity import LingBotChunk0VideoOracle  # noqa: E402
from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_fsdp_elision,
)
from instinct_pdd import DataFreeRollout, PDDConfig, advance, pdd_loss  # noqa: E402
from instinct_pdd.solvers import get_solver  # noqa: E402


def log(msg, rank=0):
    if int(os.getenv("RANK", 0)) == rank:
        print(msg, flush=True)


# -- contexts -------------------------------------------------------------------------------------

def load_context_files(root: str, heldout_from_ep: int):
    """Split the pool by EPISODE index, not by task.

    Holding out whole tasks would measure task generalisation; holding out scene instances measures
    whether the heads generalise across the observations they will actually see, with every task
    present in both halves. For a capacity question that is the right split -- a per-task holdout
    would confound "the heads cannot fit this" with "this task was never trained".
    """
    files = sorted(Path(root).glob("*.npz"))
    if not files:
        raise SystemExit(f"no contexts in {root}; run collect_contexts.sh first")
    train, held = [], []
    for f in files:
        ep = int(f.stem.split("__ep")[1].split("__")[0])
        (held if ep >= heldout_from_ep else train).append(f)
    if not train or not held:
        raise SystemExit(
            f"split produced train={len(train)} held={len(held)} from {len(files)} contexts; "
            f"--heldout-from-ep={heldout_from_ep} does not partition this pool")

    # INTERLEAVE THE HELD-OUT SET BY TASK. `glob` returns name-sorted paths, so held[:2] would be
    # adjust_bottle__ep8 and adjust_bottle__ep9 -- every validation pass would measure per-head error
    # and endpoint error on ONE task out of fifty, and report it as the held-out number. Round-robin
    # over tasks means held[:n] spans n different tasks for any n.
    by_task = {}
    for f in held:
        by_task.setdefault(f.stem.split("__ep")[0], []).append(f)
    interleaved, tasks = [], sorted(by_task)
    for i in range(max(len(v) for v in by_task.values())):
        for tk in tasks:
            if i < len(by_task[tk]):
                interleaved.append(by_task[tk][i])
    return train, interleaved


def read_context(path: Path, cam_keys):
    z = np.load(path, allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cam_keys}
    frame = {full: z[s] for s, full in short.items()}
    return {"obs": [frame], "state": z["state"]}, str(z["prompt"]), str(z["task"])


# -- validation -----------------------------------------------------------------------------------

#: Validation must be PAIRED across ranks. per_head_error shards by `k % world` and reassembles the
#: pieces, so if each rank drew its own noise the assembled vector would stitch together eight
#: unrelated trajectories and the per-head numbers would not describe any single rollout. Nothing
#: seeds torch here -- `S.init_distributed` only calls `init_process_group` -- so two ranks really do
#: draw different noise. Deriving the state from a fixed seed keyed on the CONTEXT makes it identical
#: on every rank and reproducible between validation passes, so successive reports are comparable.
VAL_SEED = 20260804


def seed_state(adapter, ctx):
    """The fp64 initial state for a validation rollout: identical on every rank, per context."""
    key = VAL_SEED ^ (abs(hash((ctx.task, ctx.prompt))) % (2 ** 31))
    g = torch.Generator(device="cpu").manual_seed(key)
    x = torch.randn(adapter.state_shape(), generator=g, dtype=torch.float32)
    return x.to(device=adapter.S.device, dtype=torch.float64)


def advance_hp(x, heads, grid, start, stop):
    """`instinct_pdd.advance` in fp64, for measurement rather than training."""
    out = x
    for l in range(start, stop):
        out = out + heads[l].double() * grid.h(l)
    return out

@torch.no_grad()
def per_head_error(adapter, student, teacher, grid, ctx, cfg, *, rank, world):
    """PDD loss for EVERY head on one context, sharded across ranks by k.

    One student forward per block yields all heads, so the cost is dominated by teacher evaluations:
    L of them per block. Sharding by `k` mod world_size makes that 8x faster and costs nothing in
    fidelity, since each k is independent.
    """
    N, L = grid.n_intervals, grid.block
    dev, dt = adapter.S.device, adapter.S.dtype
    est = get_solver(cfg.solver)

    # THREE VECTORS, NOT ONE OVERLOADED SENTINEL. An earlier version used NaN to mean "not my shard"
    # and reassembled with nan_to_num(0.0) -- so a head whose loss actually WAS NaN (a diverged head)
    # summed to 0.0 and reported as the BEST head in the table the go/no-go is read from.
    out = torch.zeros(N, dtype=torch.float64, device=dev)
    owned = torch.zeros(N, dtype=torch.float64, device=dev)
    diverged = torch.zeros(N, dtype=torch.float64, device=dev)

    # Every rank must start from the SAME x0: ranks shard by k and the pieces are stitched back
    # together, so a per-rank draw would assemble the per-head vector out of 8 different
    # trajectories, aliasing the noise with k mod 8 and making one unlucky draw look like a periodic
    # head pathology. seed_state derives it from the context, so it is identical across ranks and
    # stable across validation passes.
    x_n = seed_state(adapter, ctx)
    for n in range(0, N, L):
        heads = student.heads(x_n.to(dt), grid.cond(n), cond=ctx)
        for k in range(n, min(n + L, N)):
            if k % world != rank:
                continue
            x_k = advance_hp(x_n, heads, grid, n, k)
            target = est(teacher, x_k.to(dt), grid, k, ctx)
            val = torch.nn.functional.mse_loss(heads[k].float(), target.float()).double()
            if torch.isfinite(val):
                out[k] = val
            else:
                diverged[k] = 1.0
            owned[k] = 1.0
        x_n = advance_hp(x_n, heads, grid, n, min(n + L, N))
    if world > 1:
        for vec in (out, owned, diverged):
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    missing = int((owned == 0).sum())
    if missing:
        raise RuntimeError(
            f"per_head_error: {missing} heads owned by no rank -- the k % world sharding lost them, "
            f"so the report would silently describe a subset")
    res = out.clone()
    res[diverged > 0] = float("nan")            # NaN now means diverged, and only that
    return res.float()


@torch.no_grad()
def endpoint_error(adapter, student, teacher, grid, ctx, cfg):
    """The end-to-end number: student at NFE=2 against the teacher integrated over the whole grid.

    Both start from the SAME noise. The ODE is deterministic, so identical noise must reach the same
    latent -- a distributional score would pass a student that produced plausible latents from the
    wrong trajectories, which for a policy conditioned on those latents is not the same model.
    """
    dt = adapter.S.dtype
    x0 = seed_state(adapter, ctx)
    est = get_solver(cfg.solver)

    x = x0                                      # teacher, all N intervals, fp64 accumulator
    for k in range(grid.n_intervals):
        x = x + est(teacher, x.to(dt), grid, k, ctx).double() * grid.h(k)
    ref = x

    x = x0                                      # student, N/L forwards
    for n in range(0, grid.n_intervals, grid.block):
        heads = student.heads(x.to(dt), grid.cond(n), cond=ctx)
        x = advance_hp(x, heads, grid, n, min(n + grid.block, grid.n_intervals))
    stu = x

    l2 = float((stu - ref).pow(2).mean().sqrt())
    scale = float(ref.pow(2).mean().sqrt())
    return l2, scale, ref, stu


def sigma_buckets(grid, values, n_buckets=8):
    """Group per-head error by the sigma the head starts at.

    The interesting failure mode is uneven: heads near sigma=1 see a nearly-pure-noise state, heads
    near sigma=0 see nearly-clean data, and there is no reason a frozen trunk should serve both
    equally well. A single mean would hide exactly that.
    """
    N = grid.n_intervals
    rows = []
    for b in range(n_buckets):
        lo, hi = b * N // n_buckets, (b + 1) * N // n_buckets
        ks = [k for k in range(lo, hi) if not np.isnan(values[k])]
        if not ks:
            continue
        s_lo = grid.cond(lo) / grid.time_scale if grid.time_scale else grid.times[lo]
        rows.append({
            "k_range": [lo, hi - 1],
            "sigma_range": [round(grid.cond(lo) / 1000.0, 4), round(grid.cond(hi - 1) / 1000.0, 4)],
            "mean": float(np.mean([values[k] for k in ks])),
            "max": float(np.max([values[k] for k in ks])),
            "n_heads": len(ks),
        })
    return rows


# -- main -----------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", default="/home/ubuntu/iwm_results/pdd_heads_run1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--n-intervals", type=int, default=256)
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--solver", default="euler")
    ap.add_argument("--heldout-from-ep", type=int, default=8)
    ap.add_argument("--min-updates-per-head", type=int, default=20,
                    help="HARD GATE: no checkpoint is marked usable below this, on every head")
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--val-contexts", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--frozen-check-every", type=int, default=500,
                    help="audit that heads with no new updates are bit-identical; 0 disables")
    a = ap.parse_args()

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))

    S = import_lingbot_server()
    cfg_name = os.environ.get("IFL_CFG", "robotwin")
    cfg = S.VA_CONFIGS[cfg_name]
    cfg.save_root = "/tmp/iwm_pdd_train"
    os.makedirs(cfg.save_root, exist_ok=True)
    S.init_distributed(world, local_rank, rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, local_rank, world
    install_fsdp_elision(S)                 # keeps every parameter local; see probe_pdd_e2e.py

    log(f"building {world} server(s), one per rank (23 GB checkpoint each) ...")
    server = S.VA_Server(cfg)
    adapter = LingBotChunk0VideoOracle(server, guidance=a.guidance)
    teacher = adapter.teacher()

    L = a.n_intervals // a.nfe
    grid = adapter.grid(a.n_intervals, L)
    student = adapter.student(n_heads=grid.n_intervals)
    pcfg = PDDConfig(solver=a.solver, l_min=L, l_max=L)
    n_params = sum(p.numel() for p in student.parameters())
    log(f"grid: N={grid.n_intervals} L={grid.block} NFE={grid.nfe}  "
        f"cond {grid.cond(0):.1f} -> {grid.cond(grid.n_intervals):.1f}")
    log(f"trainable: {n_params/1e6:.1f} M head parameters; trunk FROZEN")

    train_files, held_files = load_context_files(a.contexts, a.heldout_from_ep)
    log(f"contexts: {len(train_files)} train, {len(held_files)} held out "
        f"({len(set(f.stem.split('__ep')[0] for f in train_files))} tasks)")

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)

    # Per-rank RNG so ranks draw different (context, k) -- that is the whole point of the fan-out.
    rng = random.Random(1234 + rank)
    head_counts = torch.zeros(grid.n_intervals, dtype=torch.long, device=server.device)

    # OPTIMISATION DIAGNOSTICS. Every one of these is a per-step quantity that cannot be recovered
    # from a checkpoint afterwards, which is why they are accumulated as the run goes.
    N = grid.n_intervals
    dg = {
        # clipping: how often the per-head clip engages, and how hard
        "clip_n": torch.zeros(N, device=server.device),
        "clip_hits": torch.zeros(N, device=server.device),
        "prenorm_sum": torch.zeros(N, device=server.device),
        "prenorm_max": torch.zeros(N, device=server.device),
        # the counterfactual: what a JOINT norm over the touched heads would have been, which is
        # what the pre-fix code clipped against. Reported so "before/after the per-head fix" is a
        # measured comparison rather than an argument.
        "joint_over_single_sum": torch.zeros(1, device=server.device),
        "joint_batches": torch.zeros(1, device=server.device),
        # effective step size actually taken, per head: ||p_after - p_before|| and its relative size
        "step_sum": torch.zeros(N, device=server.device),
        "step_max": torch.zeros(N, device=server.device),
        "relstep_sum": torch.zeros(N, device=server.device),
        # frozen-head audit
        "frozen_violations": 0,
        "frozen_checks": 0,
    }
    # Checksum per head, to prove an untouched head is bit-identical rather than merely "close".
    def head_sig(idx):
        with torch.no_grad():
            return float(sum(float(q.double().sum()) for q in student.head_list[idx].parameters()))
    last_sig = [head_sig(i) for i in range(N)]
    last_counts = head_counts.clone()
    out_dir = Path(a.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    ctx = None
    ctx_cache = {}
    rollout = None
    history = []
    # Held-out error need not be monotone, so track the best pass separately. Reporting only the
    # final step would conflate "stopped improving" with "got worse" -- different answers to the
    # capacity question.
    best = {"rmse": float("inf"), "step": -1}
    t0 = time.time()

    for step in range(a.steps):
        # One context per trajectory. A trajectory is N/L blocks, so it naturally spans that many
        # steps; re-encoding more often would pay a VAE forward and a T5 forward for nothing.
        if rollout is None or rollout.n >= grid.n_intervals:
            f = rng.choice(train_files)
            obs, prompt, task = read_context(f, cfg.obs_cam_keys)
            # encode_context runs _reset (which clears the KV and BOTH streaming VAE caches -- that
            # clearing is required for correctness, so the reset cannot be skipped) and then a VAE
            # forward. The reset is cheap; the VAE forward is not, and the pool has only a few
            # hundred distinct contexts, so its result is cached and the encode is done once each.
            ctx = ctx_cache.get(f)
            if ctx is None:
                ctx = adapter.encode_context(obs, prompt=prompt, task=task)
                ctx_cache[f] = ctx
            else:
                adapter.S._reset(prompt=ctx.prompt)     # still required: clears KV + VAE caches
            rollout = DataFreeRollout(grid, lambda: torch.randn(
                adapter.state_shape(), dtype=server.dtype, device=server.device), block=L)

        n, x_n = rollout.begin_block()
        # NOT instinct_pdd.sample_k here, deliberately. It draws from the global torch RNG, which
        # torch.distributed seeds identically on every rank -- so all eight ranks would supervise the
        # SAME head every step, and the fan-out would buy nothing while looking like it worked.
        # Coverage would still climb, just eight times slower than the logs suggest.
        k = rng.randrange(n, min(n + L, grid.n_intervals))

        heads = student.heads(x_n, grid.cond(n), cond=ctx)
        out = pdd_loss(student, teacher, x_n, grid, n, k, cond=ctx, cfg=pcfg, heads=heads)

        opt.zero_grad(set_to_none=True)
        out.loss.backward()

        # WHY EVERY HEAD HAS A GRADIENT HERE, AND WHY THAT IS A PROBLEM.
        # `student.heads()` stacks all 256 head outputs, so backward through that stack writes an
        # EXACT ZERO gradient into every head the loss did not use. Verified on this interpreter:
        # after a backward that touches head 3 only, `grad is None` is False for all six heads of a
        # repro and the norms are [0, 0, 0, 0.4948, 0, 0].
        #
        # AdamW skips a parameter only when its grad is None -- so left alone it would apply weight
        # decay and a momentum step to all 256 heads every iteration. Measured: an untouched head
        # moves 4.8e-06 per step. With ~248 of 256 heads untouched each step, every head would be
        # decayed roughly 19,000 times against ~78 real updates over this run, which is a systematic
        # shrink dressed up as regularisation.
        #
        # So: all-reduce the touched heads, then NULL the rest, which makes AdamW ignore them.
        if world > 1:
            mine = torch.tensor([k], dtype=torch.long, device=server.device)
            gathered = [torch.zeros_like(mine) for _ in range(world)]
            dist.all_gather(gathered, mine)
            ks = [int(g.item()) for g in gathered]
        else:
            ks = [k]
        touched = sorted(set(ks))
        counts = {kk: ks.count(kk) for kk in touched}

        prenorms, before_p = [], {}
        for idx, head in enumerate(student.head_list):
            if idx not in touched:
                for prm in head.parameters():
                    prm.grad = None                 # AdamW will not touch it
                continue
            for prm in head.parameters():
                if prm.grad is None:                # defensive: keep the collective symmetric
                    prm.grad = torch.zeros_like(prm)
                if world > 1:
                    # Every rank iterates `sorted(touched)`, so the collectives are issued in the
                    # same order on all ranks -- a rank-dependent order here would deadlock.
                    dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM)
                    prm.grad /= counts[idx]
            # Clip PER HEAD. A joint norm over the 7-8 touched heads would make each head's scale
            # factor depend on the other ranks' gradients that step, reintroducing exactly the
            # world-size coupling that dividing by counts[idx] was written to remove.
            pre = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
            dg["clip_n"][idx] += 1
            dg["prenorm_sum"][idx] += pre
            dg["prenorm_max"][idx] = max(float(dg["prenorm_max"][idx]), pre)
            if pre > 1.0:
                dg["clip_hits"][idx] += 1
            prenorms.append(pre)
            before_p[idx] = [q.detach().clone() for q in head.parameters()]
            head_counts[idx] += counts[idx]

        opt.step()

        with torch.no_grad():
            for idx, prev in before_p.items():
                cur = list(student.head_list[idx].parameters())
                d = float(torch.sqrt(sum((a - b).double().pow(2).sum() for a, b in zip(cur, prev))))
                nrm = float(torch.sqrt(sum(a.double().pow(2).sum() for a in cur)))
                dg["step_sum"][idx] += d
                dg["step_max"][idx] = max(float(dg["step_max"][idx]), d)
                dg["relstep_sum"][idx] += d / max(nrm, 1e-12)
            if prenorms:
                # A joint clip would have divided by sqrt(sum of squares) instead of each head's own
                # norm, so this ratio is exactly the extra suppression the pre-fix code applied.
                joint = sum(x * x for x in prenorms) ** 0.5
                single = sum(prenorms) / len(prenorms)
                dg["joint_over_single_sum"] += joint / max(single, 1e-12)
                dg["joint_batches"] += 1
        rollout.advance(heads.detach())

        if a.frozen_check_every and step and step % a.frozen_check_every == 0:
            # An untouched head must be BIT-IDENTICAL, not merely close: weight decay or a momentum
            # tail would show up as a tiny drift, which is exactly the defect Run 1 carried.
            with torch.no_grad():
                for i in range(N):
                    if int(head_counts[i]) == int(last_counts[i]):
                        s = head_sig(i)
                        if s != last_sig[i]:
                            dg["frozen_violations"] += 1
                        dg["frozen_checks"] += 1
                    last_sig[i] = head_sig(i)
                last_counts = head_counts.clone()

        if rank == 0 and a.log_every and step % a.log_every == 0:
            cov = int((head_counts > 0).sum())
            log(f"  step {step:>6}  n={n:<4} k={k:<4} loss={out.metrics['loss']:.6f}  "
                f"heads touched {cov}/{grid.n_intervals}  {(time.time()-t0)/max(1,step+1):.2f}s/step")

        if a.val_every and step and step % a.val_every == 0:
            report = validate(adapter, student, teacher, grid, pcfg, held_files, cfg,
                              rank=rank, world=world, n_contexts=a.val_contexts)
            if rank == 0:
                if report["endpoint_rmse"] < best["rmse"]:
                    best = {"rmse": report["endpoint_rmse"], "step": step}
                    # Keep the best-by-endpoint weights separately. Held-out error need not be
                    # monotone, and reporting only the final step would conflate "stopped improving"
                    # with "got worse", which are different answers to the capacity question.
                    save(student, grid, a, out_dir, head_counts, report, tag="best")
                summarise(report, head_counts, grid, step, out_dir, a, dg=dg, best=best)
                history.append(report)
                # Checkpoint at every validation. A 20k-step run is many GPU-hours, and an OOM or an
                # NCCL watchdog abort at step 19,000 would otherwise leave nothing on disk.
                save(student, grid, a, out_dir, head_counts, report, tag=f"step{step}")
                # Checkpoint at every validation boundary. A 20k-step run over 8 GPUs is many
                # GPU-hours; writing only at the end means any crash, preemption or OOM discards all
                # of it. Optimiser state goes too, so a resume is a resume and not a restart.
                ck = out_dir / f"step{step}"
                ck.mkdir(parents=True, exist_ok=True)
                torch.save({"heads": student.state_dict(), "opt": opt.state_dict(),
                            "head_counts": head_counts.cpu(), "step": step}, ck / "ckpt.pt")

    # -- final ------------------------------------------------------------------------------------
    report = validate(adapter, student, teacher, grid, pcfg, held_files, cfg,
                      rank=rank, world=world, n_contexts=a.val_contexts)
    # NOT max(val_contexts, 4). validate() picks contexts by striding held_files by len//n, so
    # changing n changes WHICH contexts are evaluated, not just how many: at n=3 the stride is 33
    # and at n=4 it is 25, giving sets that overlap in one of four. The final report then looks like
    # a large improvement over the last intermediate one -- measured on the run that produced this
    # file, per-head error appeared to fall 0.520 -> 0.187 and the endpoint 17.9% -> 16.4%, purely
    # from the swap. The reference scale moving (1.3139 -> 1.2767) is the tell, since scale is a
    # property of the contexts. A larger final sample is worth having, but it has to be a SUPERSET
    # reported alongside the tracked set, never a substitute for it.
    # head_counts needs no all-reduce: every rank adds counts[kk] for every touched head, so all
    # ranks already hold the same global histogram.
    if rank == 0:
        if report["endpoint_rmse"] < best["rmse"]:
            best = {"rmse": report["endpoint_rmse"], "step": a.steps}
            save(student, grid, a, out_dir, head_counts, report, tag="best")
        summarise(report, head_counts, grid, a.steps, out_dir, a, final=True, dg=dg, best=best)
        save(student, grid, a, out_dir, head_counts, report)
        (out_dir / "history.json").write_text(json.dumps(
            {"best": best, "passes": [{"step": h.get("step"), "endpoint_rmse": h["endpoint_rmse"],
                                       "endpoint_scale": h["endpoint_scale"],
                                       "tasks": h.get("tasks")} for h in history]}, indent=2))
    if dist.is_initialized():
        dist.barrier()
    return 0


def validate(adapter, student, teacher, grid, pcfg, held_files, cfg, *, rank, world, n_contexts):
    """Per-head error and endpoint error on held-out contexts."""
    per_head = torch.zeros(grid.n_intervals, device=adapter.S.device)
    used = 0
    eps, scales, tasks = [], [], []
    # STRIDE, don't take the head of the sorted glob. held_files is sorted by filename, so the first
    # two entries are two episodes of ONE task (alphabetically first) -- validation would report the
    # heads' error on adjust_bottle and nothing else, while reading as a held-out number.
    stride = max(1, len(held_files) // max(1, n_contexts))
    chosen = held_files[::stride][:n_contexts]
    for i, f in enumerate(chosen):
        obs, prompt, task = read_context(f, cfg.obs_cam_keys)
        ctx = adapter.encode_context(obs, prompt=prompt, task=task)
        # Noise is derived from the CONTEXT inside seed_state: identical on every rank (so the
        # k-sharded per-head vector describes one trajectory) and stable across validation passes
        # (so successive reports are comparable). Nothing here needs to pass a seed.
        per_head += per_head_error(adapter, student, teacher, grid, ctx, pcfg,
                                   rank=rank, world=world)
        e, s, _, _ = endpoint_error(adapter, student, teacher, grid, ctx, pcfg)
        eps.append(e)
        scales.append(s)
        tasks.append(task)
        used += 1
    per_head /= max(1, used)
    return {
        "tasks": tasks,
        "per_head": per_head.float().cpu().numpy().tolist(),
        "endpoint_rmse": float(np.mean(eps)) if eps else float("nan"),
        "endpoint_scale": float(np.mean(scales)) if scales else float("nan"),
        "n_contexts": used,
        "val_tasks": tasks,
    }


def usage_histogram(hc, bins=10):
    """How lopsided is head supervision? Flat is the goal; a long tail means some heads are
    effectively untrained even when the min-updates gate passes.

    Integer-aware: with a narrow range (early in a run, counts 0-2) float bin edges collapse to
    nonsense like "0--1", so the bins become one-per-value until the range is wide enough to bucket.
    """
    lo, hi = int(hc.min()), int(hc.max())
    span = hi - lo + 1
    if span <= bins:
        return [{"range": [v, v], "n_heads": int((hc == v).sum())}
                for v in range(lo, hi + 1)]
    width = -(-span // bins)                      # ceil, so the last bin is not overlong
    rows = []
    for b in range(bins):
        a0 = lo + b * width
        a1 = min(hi, a0 + width - 1)
        if a0 > hi:
            break
        rows.append({"range": [a0, a1],
                     "n_heads": int(((hc >= a0) & (hc <= a1)).sum())})
    return rows


def corr(x, y):
    """Pearson and Spearman between update count and error. Spearman matters more here: the question
    is whether MORE-updated heads do better at all, not whether the relation is linear."""
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    spear = float(np.corrcoef(rx, ry)[0, 1])
    return pear, spear


def summarise(report, head_counts, grid, step, out_dir, a, final=False, dg=None, best=None):
    v = np.asarray(report["per_head"], dtype=float)
    hc = head_counts.cpu().numpy()
    cov = int((hc > 0).sum())
    gate_ok = bool((hc >= a.min_updates_per_head).all())
    worst = np.argsort(-v)[:10]

    print(f"\n{'=' * 78}")
    print(f"{'FINAL' if final else 'VALIDATION'} at step {step}   "
          f"({report['n_contexts']} held-out contexts)")
    print(f"  head updates      min {int(hc.min())}  median {int(np.median(hc))}  "
          f"max {int(hc.max())}   touched {cov}/{len(hc)}")
    print(f"  COVERAGE GATE     >= {a.min_updates_per_head} updates on every head: "
          f"{'PASS' if gate_ok else 'FAIL'}")
    print(f"  per-head error    mean {v[~np.isnan(v)].mean():.6f}  "
          f"max {np.nanmax(v):.6f}  min {np.nanmin(v):.6f}")
    print(f"  endpoint RMSE     {report['endpoint_rmse']:.6f}  "
          f"(reference scale {report['endpoint_scale']:.4f}, "
          f"{100*report['endpoint_rmse']/max(1e-9,report['endpoint_scale']):.1f}% )")
    print("\n  error by sigma range (the head's starting sigma):")
    print(f"    {'heads':>12} {'sigma':>16} {'mean err':>11} {'max err':>11} {'min upd':>8}")
    for row in sigma_buckets(grid, v):
        lo, hi = row["k_range"]
        s0, s1 = row["sigma_range"]
        krange = f"{lo}-{hi}"
        srange = f"{s0:.3f}-{s1:.3f}"
        print(f"    {krange:>12} {srange:>16} {row['mean']:>11.6f} "
              f"{row['max']:>11.6f} {int(hc[lo:hi + 1].min()):>8}")
    print("\n  worst 10 heads:")
    for k in worst:
        print(f"    k={int(k):<4} sigma={grid.cond(int(k))/1000:.4f}  err={v[k]:.6f}  "
              f"updates={int(hc[k])}")
    pear, spear = corr(hc.astype(float), v)
    print(f"\n  update-count vs error   Pearson {pear:+.4f}   Spearman {spear:+.4f}")
    print("     (negative => more-updated heads do better; ~0 => updates are not the limiter)")
    print("\n  head usage histogram:")
    for row in usage_histogram(hc):
        bar = "#" * max(0, int(40 * row["n_heads"] / max(1, len(hc))))
        print(f"    {row['range'][0]:>5}-{row['range'][1]:<5} {row['n_heads']:>4} heads {bar}")

    if dg is not None:
        cn = dg["clip_n"].cpu().numpy()
        ch = dg["clip_hits"].cpu().numpy()
        pmax = dg["prenorm_max"].cpu().numpy()
        psum = dg["prenorm_sum"].cpu().numpy()
        ssum = dg["step_sum"].cpu().numpy()
        rsum = dg["relstep_sum"].cpu().numpy()
        tot = max(1.0, float(cn.sum()))
        jb = float(dg["joint_batches"].item())
        jr = float(dg["joint_over_single_sum"].item()) / max(1.0, jb)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_pre = np.where(cn > 0, psum / np.maximum(cn, 1), np.nan)
            mean_step = np.where(cn > 0, ssum / np.maximum(cn, 1), np.nan)
            mean_rel = np.where(cn > 0, rsum / np.maximum(cn, 1), np.nan)
        print("\n  optimisation diagnostics:")
        print(f"    head-updates applied      {int(cn.sum())}")
        print(f"    clip engaged              {100*ch.sum()/tot:.2f}% of head-updates")
        print(f"    pre-clip grad norm        mean {np.nanmean(mean_pre):.4f}  "
              f"max {np.nanmax(pmax):.4f}")
        print(f"    joint/per-head norm ratio {jr:.3f}x   <- the extra suppression a JOINT clip")
        print(f"                                          would have applied (Run 1's defect)")
        print(f"    realised step ||dp||      mean {np.nanmean(mean_step):.3e}  "
              f"max {np.nanmax(dg['step_max'].cpu().numpy()):.3e}")
        print(f"    relative step ||dp||/||p||  mean {np.nanmean(mean_rel):.3e}  "
              f"p5 {np.nanpercentile(mean_rel,5):.3e}  p95 {np.nanpercentile(mean_rel,95):.3e}")
        # ZERO CHECKS IS NOT A PASS. The audit only inspects heads whose update count did not change
        # between checks, and with 8 heads/step over a 500-step window every head gets touched -- so
        # the condition never fires and the audit is vacuous at that cadence. Printing PASS for it
        # would claim evidence that was never gathered.
        if dg["frozen_checks"] == 0:
            print(f"    FROZEN AUDIT              NOT EVALUATED -- 0 untouched-head checks fired "
                  f"(every head was updated within each {a.frozen_check_every}-step window; "
                  f"lower --frozen-check-every to exercise it)")
        else:
            print(f"    FROZEN AUDIT              {dg['frozen_violations']} violations "
                  f"in {dg['frozen_checks']} checks of untouched heads"
                  f"   {'PASS' if dg['frozen_violations'] == 0 else 'FAIL'}")
    if best is not None:
        print(f"\n  best endpoint so far      {best['rmse']:.6f} at step {best['step']} "
              f"(current {report['endpoint_rmse']:.6f})")
    print("=" * 78, flush=True)

    (out_dir / f"report_step{step}.json").write_text(json.dumps({
        "step": step, "per_head_error": v.tolist(), "head_updates": hc.tolist(),
        "coverage": cov, "coverage_gate_pass": gate_ok,
        "min_updates_required": a.min_updates_per_head,
        "endpoint_rmse": report["endpoint_rmse"], "endpoint_scale": report["endpoint_scale"],
        "sigma_buckets": sigma_buckets(grid, v),
        "worst_heads": [{"k": int(k), "sigma": grid.cond(int(k)) / 1000.0,
                         "error": float(v[k]), "updates": int(hc[k])} for k in worst],
        "usage_histogram": usage_histogram(hc),
        "corr_updates_vs_error": {"pearson": pear, "spearman": spear},
        "diagnostics": None if dg is None else {
            "head_updates_applied": int(dg["clip_n"].sum().item()),
            "clip_engaged_frac": float(dg["clip_hits"].sum().item() /
                                       max(1.0, dg["clip_n"].sum().item())),
            "prenorm_max": float(dg["prenorm_max"].max().item()),
            "joint_over_per_head_ratio": float(dg["joint_over_single_sum"].item() /
                                               max(1.0, dg["joint_batches"].item())),
            "frozen_violations": int(dg["frozen_violations"]),
            "frozen_checks": int(dg["frozen_checks"]),
            "frozen_verdict": ("NOT_EVALUATED" if dg["frozen_checks"] == 0
                               else ("PASS" if dg["frozen_violations"] == 0 else "FAIL")),
        },
        "best": best,
    }, indent=2))


def save(student, grid, a, out_dir, head_counts, report, tag: str = "final"):
    """Write the heads, and refuse to call the checkpoint usable if the coverage gate failed."""
    hc = head_counts.cpu().numpy()
    gate_ok = bool((hc >= a.min_updates_per_head).all())
    p = out_dir / (tag if gate_ok else f"{tag}_GATE_FAILED")
    p.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), p / "heads.pt")
    # AUDIT.md Stage 2: write the two-namespace declaration ALONGSIDE the legacy delta.json.
    # Additive on purpose -- delta.json keeps every existing checkpoint servable, and the runtime
    # prefers instinctflash.json when it is present. Execution facts are CAPABILITIES: any recipe
    # producing per-interval velocity heads writes the same execution block, and "PDD" appears only
    # under provenance, which the runtime never reads.
    (p / "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {
            "model_id": f"lingbot-va-robotwin-blockheads-{grid.nfe}v",
            "backbone": "wan-va",
            "servable": gate_ok,
            "guidance": {"video": a.guidance},
            "nfe": {"video": grid.nfe},
            "output_projection": {
                "kind": "per_interval_velocity_heads",
                "n_intervals": grid.n_intervals,
                "block": grid.block,
                # The adapter negates once during training, so the head's RAW output is already the
                # sigma-velocity FlowMatchScheduler.step consumes. Declaring it turns the comment that
                # used to guard this into a field the loader checks.
                "velocity_convention": "sigma_descending",
                "foldable": True,
            },
        },
        "provenance": {
            "training_method": "parallel_decoding_distillation",
            "recipe_repo": "https://github.com/General-Instinct/instinct-pdd",
            "trainable": "output heads only; trunk frozen",
            "solver": a.solver,
            "training_diagnostics": {
                "coverage_gate_pass": gate_ok,
                "min_updates_per_head": a.min_updates_per_head,
                "head_updates_min": int(hc.min()),
                "endpoint_rmse": report["endpoint_rmse"],
            },
            "note": ("chunk-0 video stream only; the action stream reads the KV the video stream "
                     "commits and is a separate distillation stage"),
        },
    }, indent=2))

    (p / "delta.json").write_text(json.dumps({
        "recipe": "parallel_decoding_distillation",
        "trainable": "output heads only; trunk frozen",
        "nfe": {"video": grid.nfe},
        "n_intervals": grid.n_intervals, "block": grid.block,
        "solver": a.solver, "guidance": {"video": a.guidance},
        "coverage_gate_pass": gate_ok,
        "min_updates_per_head": a.min_updates_per_head,
        "head_updates_min": int(hc.min()),
        "endpoint_rmse": report["endpoint_rmse"],
        "note": ("chunk-0 video stream only; the action stream reads the KV the video stream commits "
                 "and is a separate distillation stage"),
    }, indent=2))
    print(f"\nwrote {p}" + ("" if gate_ok else
          "\nNOT USABLE: the coverage gate failed. Some heads are undertrained, which a falling "
          "loss curve does not reveal. Raise --steps or lower --min-updates-per-head deliberately."),
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
