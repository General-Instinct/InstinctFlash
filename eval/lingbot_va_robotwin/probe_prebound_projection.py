#!/usr/bin/env python3
"""Falsify (or don't) the Layer 6 top candidate BEFORE writing it as a pass.

The proposal: every `nn.Linear` re-derives `weight.t()` and the 3-D->2-D collapse on every call.
`aten::linear` internally computes `addmm(bias, input.reshape(-1,C), weight.t())`, so calling that
expression directly -- with the transposed view built ONCE at install -- removes five of the eight aten
events per Linear while invoking the identical kernel on identical operands.

A microbenchmark already showed 8 -> 3 events, bit-exact on four real shapes. That is region scale, and
region scale has lied four times in this project (RoPE 1.10x -> 0.3%; cast hoist 1.6% -> 0.66%; fused QKV
1.9% -> 0.2% SLOWER; graph persistence 1.72x -> 1.43x slower). So this probe answers the two questions
that decide whether the pass is worth writing, and nothing else:

  1. Does the CYCLE's aten event count actually fall by ~12,250 of 105,123?
     If not, the arithmetic is wrong and there is nothing to build.
  2. Does the cycle get FASTER, under ABBA ordering?
     If the count falls 11.7% and the clock does not move, the ~3.2 us/op model is refuted -- which is a
     larger result than the pass, and redirects Layer 6.

Plus the gate that must hold either way: max |delta action| = 0.

This is a monkeypatch measurement, NOT the pass. It is reversible in-process, so both ABBA arms run in
one server -- unlike graph capture, which rewrites a shared class and needs one arm per process.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29991 \\
        probe_prebound_projection.py [--cycles 8] [--arm-cycles 12]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


# --------------------------------------------------------------------------------------------------
# The transformation under test. Fail-closed: any condition we have not certified falls back to
# F.linear, so a wrong guess costs performance and never correctness.
# --------------------------------------------------------------------------------------------------
def prebind(mod: torch.nn.Linear):
    """Return a forward that calls addmm directly against a transposed view built once."""
    weight, bias = mod.weight, mod.bias
    wt = weight.t()                      # a view: stride swap, no copy, no storage
    ver = weight._version                # a later weight-mutating pass must not be silently ignored
    in_features = weight.shape[1]

    def forward(x):
        if weight._version != ver:       # FAIL CLOSED: a stale transposed view is wrong numerics
            return F.linear(x, weight, bias)
        if x.dim() == 2:
            return torch.addmm(bias, x, wt) if bias is not None else x.mm(wt)
        if not x.is_contiguous():        # FAIL CLOSED: view(-1, C) is only legal on a viewable layout
            return F.linear(x, weight, bias)
        out = x.view(-1, in_features)
        out = torch.addmm(bias, out, wt) if bias is not None else out.mm(wt)
        return out.view(*x.shape[:-1], out.shape[-1])

    return forward


def install(root: torch.nn.Module) -> list[torch.nn.Linear]:
    mods = [m for m in root.modules() if isinstance(m, torch.nn.Linear)]
    for m in mods:
        m._iwm_orig_forward = m.forward
        m.forward = prebind(m)
    return mods


def uninstall(mods) -> None:
    for m in mods:
        m.forward = m._iwm_orig_forward
        del m._iwm_orig_forward


def count_events(fn):
    """Profiler-visible aten events for one call of fn, warmed first."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    ka = p.key_averages()
    aten = {e.key: e.count for e in ka if e.key.startswith("aten::")}
    return sum(aten.values()), aten


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70, help="past ring saturation (~cycle 64)")
    ap.add_argument("--cycles", type=int, default=8, help="paired seeded cycles for the delta gate")
    ap.add_argument("--arm-cycles", type=int, default=12, help="cycles per ABBA arm")
    a = ap.parse_args()

    hot = [ln for ln in os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    ).read().strip().split("\n") if ln.strip() and int(ln.split(",")[1]) >= 15]
    if hot:
        print(f"LATENCY NOT EVALUATED: fleet busy ({'; '.join(x.strip() for x in hot)}%). "
              f"Counts and exactness are contention-insensitive and still run.")

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_prebound"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    print("building server at 2V/4A with the shipped stack (P003 ring KV + P007 conv layout) ...",
          flush=True)
    server = S.VA_Server(cfg)
    from instinctflash.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for _ in install_conditioning_prefill(S, type(server)):
        pass
    for _ in install_debug_dump_elision(S):
        pass
    from instinctflash.backends.conv.apply import install_conv_layout
    for _ in install_conv_layout(server):
        pass

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts; run collect_contexts.sh")
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = [{full: z[s] for s, full in short.items()}]
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)

    def cycle(rng, first=False):
        if first:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        act = server.infer(dict(obs=obs, prompt=prompt, save_visualization=False))["action"]
        kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
              for _ in range(4 if first else 8)]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        return np.asarray(act, dtype=np.float64).copy()

    def run(n, seed=0):
        """Both arms must see identical noise on EVERY cycle including the first.

        Seeding from i=1 leaves cycle 0 drawing from whatever global RNG state the previous arm left
        behind, which shows up as a nonzero delta that belongs to the harness rather than to the
        transformation. That is the failure this loop is written to avoid.
        """
        rng = np.random.default_rng(seed)
        acts = []
        for i in range(n):
            torch.manual_seed(1234 + i)
            acts.append(cycle(rng, first=(i == 0)))
        return acts

    def warm(n, seed=7):
        rng = np.random.default_rng(seed)
        cycle(rng, first=True)
        for _ in range(n):
            cycle(rng)
        return rng

    def timed_arm(n, rng):
        xs = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cycle(rng)
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(xs), xs

    print(f"warming {a.warm} cycles ...", flush=True)
    rng = warm(a.warm)

    n_lin = len([m for m in server.transformer.modules() if isinstance(m, torch.nn.Linear)])
    print(f"\n{'=' * 110}\n0. SCOPE\n{'=' * 110}")
    print(f"  nn.Linear modules in the transformer: {n_lin}")

    # ---- 1. host op count, the ranking term ------------------------------------------------------
    print(f"\n{'=' * 110}\n1. HOST OP COUNT PER CYCLE -- the only Layer 6 ranking term\n{'=' * 110}")
    base_n, base_ops = count_events(lambda: cycle(rng))
    mods = install(server.transformer)
    treat_n, treat_ops = count_events(lambda: cycle(rng))
    uninstall(mods)
    print(f"  baseline  {base_n:7d} aten events/cycle")
    print(f"  prebound  {treat_n:7d} aten events/cycle   ({base_n - treat_n:+d}, "
          f"{(base_n - treat_n) / max(base_n, 1):.1%})")
    print(f"  predicted 12,250 removed; at ~3.2 us/op that is "
          f"{(base_n - treat_n) * 3.2 / 1000:.1f} ms of a ~351 ms cycle")
    print(f"\n  {'operator':<34}{'base':>8}{'prebound':>10}{'delta':>8}")
    for op in ("aten::linear", "aten::t", "aten::reshape", "aten::as_strided", "aten::view",
               "aten::transpose", "aten::addmm", "aten::mm"):
        b, t = base_ops.get(op, 0), treat_ops.get(op, 0)
        if b or t:
            print(f"  {op:<34}{b:>8}{t:>10}{t - b:>+8}")
    check(base_n - treat_n > 8000, "the cycle's aten event count falls by more than 8,000",
          f"{base_n - treat_n} removed")

    # ---- 2. exactness ----------------------------------------------------------------------------
    print(f"\n{'=' * 110}\n2. BITEXACT gate: max |delta action| over {a.cycles} paired seeded "
          f"cycles\n{'=' * 110}")
    base_acts = run(a.cycles, seed=3)
    mods = install(server.transformer)
    treat_acts = run(a.cycles, seed=3)
    worst = max(float(np.abs(x - y).max()) for x, y in zip(base_acts, treat_acts))
    per = [float(np.abs(x - y).max()) for x, y in zip(base_acts, treat_acts)]
    uninstall(mods)
    print(f"  per-cycle max|delta|: {['%.3g' % v for v in per]}")
    check(worst == 0.0, f"max |delta action| = 0 over {a.cycles} cycles", f"worst {worst:.3e}")

    # ---- 3. the cycle gate, ABBA -----------------------------------------------------------------
    print(f"\n{'=' * 110}\n3. CYCLE GATE -- ABBA (base, treat, treat, base), "
          f"{a.arm_cycles} cycles/arm\n{'=' * 110}")
    rng = warm(20)
    arms = {}
    order = [("base", False), ("treat", True), ("treat", True), ("base", False)]
    for i, (name, on) in enumerate(order):
        mods = install(server.transformer) if on else None
        m, xs = timed_arm(a.arm_cycles, rng)
        if mods is not None:
            uninstall(mods)
        arms.setdefault(name, []).append(m)
        print(f"  arm {i + 1} {name:6s} median {m:7.1f} ms   min {min(xs):7.1f}  max {max(xs):7.1f}")
    b_mean = sum(arms["base"]) / 2
    t_mean = sum(arms["treat"]) / 2
    drift_b = abs(arms["base"][0] - arms["base"][1]) / b_mean
    drift_t = abs(arms["treat"][0] - arms["treat"][1]) / t_mean
    print(f"\n  base  {arms['base'][0]:.1f} / {arms['base'][1]:.1f} -> {b_mean:.1f} ms   "
          f"drift {drift_b:.1%}")
    print(f"  treat {arms['treat'][0]:.1f} / {arms['treat'][1]:.1f} -> {t_mean:.1f} ms   "
          f"drift {drift_t:.1%}")
    print(f"  speedup {b_mean / t_mean:.3f}x   ({b_mean - t_mean:+.1f} ms/cycle)")
    if hot or drift_b > 0.05:
        print(f"  LATENCY NOT EVALUATED: {'fleet busy' if hot else f'base drift {drift_b:.1%} > 5%'}")
    else:
        print(f"  predicted from the op count: {(base_n - treat_n) * 3.2 / 1000:.1f} ms; "
              f"measured {b_mean - t_mean:+.1f} ms  -> the ~3.2 us/op model is "
              f"{'CORROBORATED' if abs((b_mean - t_mean) - (base_n - treat_n) * 3.2 / 1000) < 20 else 'NOT corroborated'} "
              f"prospectively for the first time")

    print("\n" + "=" * 110)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the count falls and the actions are bit-identical. Read section 3 for whether it pays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
