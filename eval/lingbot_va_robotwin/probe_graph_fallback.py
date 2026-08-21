#!/usr/bin/env python3
"""P005 v1.0.1 regression gate: the eager FALLBACK must still advance the ring.

WHY THE STANDARD P005 GATE DOES NOT COVER THIS FIX. That gate runs 6 paired seeded cycles and checks
max|delta action| = 0. Capture succeeds on all of them, so the fallback path is never entered and the
gate would pass identically before and after the fix. Gating a bug fix with a test that cannot fail
on the bug is worse than not gating it, because it produces a signed-off feeling.

THE BUG. `install` sets `_iwm_defer_commit = True` permanently, so `WanAttention.forward` stops
committing inline and the ONLY thing that advances the ring is `_commit_all`. Both fallback returns
called `_stack` directly and skipped it. From the first capture failure onwards the ring froze: `count`
stopped growing, every later forward rewrote the same slots, and attention read a stale window. Nothing
raised. The actions stayed plausible and were wrong.

WHAT THIS PROBE DOES. Installs the pass, runs cycles normally to establish the ring's growth rate, then
forces `failed` (the pass instance IS the engine -- `engine = self`) and runs more cycles. The ring must
keep growing at the SAME rate. Before the fix it flatlines.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29981 probe_graph_fallback.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctflash.passes.lingbot.graph_capture import GraphBlockStack  # noqa: E402
from instinctflash.passes.lingbot.ring_kv import RingKVAddressing  # noqa: E402
from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_fsdp_elision,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""), flush=True)
    if not cond:
        FAILED.append(label)


class RingUnreadable(RuntimeError):
    """The ring counter could not be read. NOT EVALUATED -- never a pass, never a failure."""


def ring_count(server, cache_name) -> int:
    """Total occupied KV slots across all blocks -- the thing the bug froze.

    READ THROUGH `_iwm_ring_signature`, which is the accessor `ring_kv` actually installs. The first
    version of this probe guessed at `_iwm_count`, `_iwm_ring_count` and `kv_count`, none of which
    exist, so it summed nothing and reported `0 -> 0` for both the captured and the fallback path --
    which then FAILED the gate, as though the ring had frozen. It had not; the probe was blind.

    That is exactly the failure mode ring_kv.py:250-253 records about the original graph-capture
    integration: "The first integration keyed on an attribute that did not exist, captured 6 graphs
    where it needed many more, and replayed stale ones." Second occurrence of the same bug class, so
    this version refuses to return a number it could not read.
    """
    tot, seen = 0, 0
    for blk in server.transformer.blocks:
        a = blk.attn1
        sig = getattr(a, "_iwm_ring_signature", None)
        if sig is None:
            continue
        s = sig(cache_name)
        if s is None:
            continue
        tot += int(s[1])            # (start, count)
        seen += 1
    if seen == 0:
        raise RingUnreadable(
            f"no block exposed a readable _iwm_ring_signature({cache_name!r}). Either ring_kv is not "
            f"installed or its accessor was renamed. A gate that cannot read its own observable "
            f"reports NOT EVALUATED; it does not report 0.")
    return tot


def one_cycle(server, obs, prompt, first: bool):
    if first:
        server._reset(prompt=prompt)
    return server._infer(obs, frame_st_id=0 if first else 1)


def main() -> int:
    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_gate_fallback"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)

    print("building the real server ...", flush=True)
    server = S.VA_Server(cfg)

    # ring_kv MUST go first. graph_block_stack refuses to install without it (it needs _iwm_commit /
    # _iwm_ring_signature), and the first version of this probe skipped it -- so the gate exited 1 on a
    # harness error and reported nothing about the fix it exists to defend. A gate that cannot run is
    # indistinguishable from a gate that passes if you only read the exit code.
    RingKVAddressing().install(S, type(server))
    print("  installed ring_kv_addressing (prerequisite)", flush=True)

    gp = GraphBlockStack()
    for n in gp.install(S, type(server)):
        print(f"  installed {n}", flush=True)

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts; run collect_contexts.sh")
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])

    cache_name = server.cache_name
    print("\n=== 1. captured path: the ring advances ===")
    one_cycle(server, obs, prompt, first=True)
    try:
        c0 = ring_count(server, cache_name)
    except RingUnreadable as e:
        print(f"\nNOT EVALUATED: {e}")
        return 2
    for _ in range(3):
        one_cycle(server, obs, prompt, first=False)
    c1 = ring_count(server, cache_name)
    per_cycle = (c1 - c0) / 3.0
    check(c1 > c0, "ring grows while capture succeeds", f"{c0} -> {c1} ({per_cycle:.0f}/cycle)")
    check(not gp.failed, "capture did not fail on its own", f"failed={gp.failed!r}")

    print("\n=== 2. FORCE the fallback: the ring must keep advancing ===")
    # The pass instance is the engine (`engine = self`), so this is exactly the state a real capture
    # failure leaves behind -- an OOM mid-run is how it was originally found.
    gp.failed = "forced by probe_graph_fallback"
    c2 = ring_count(server, cache_name)
    for _ in range(3):
        one_cycle(server, obs, prompt, first=False)
    c3 = ring_count(server, cache_name)
    per_cycle_fb = (c3 - c2) / 3.0
    check(c3 > c2, "ring STILL grows on the eager fallback path", f"{c2} -> {c3}")
    check(per_cycle_fb > 0.9 * per_cycle,
          "and at the same rate as the captured path (this is the regression)",
          f"captured {per_cycle:.0f}/cycle vs fallback {per_cycle_fb:.0f}/cycle")
    if per_cycle_fb == 0:
        print("  ^ THIS IS THE BUG: the ring froze. Every later forward rewrites the same slots and "
              "attention reads a stale window, with nothing raised.", flush=True)

    print("\n" + "=" * 72)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: P005 v1.0.1 -- the eager fallback advances the ring at the captured-path rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
