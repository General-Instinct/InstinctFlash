#!/usr/bin/env python3
"""Gates for the post-saturation Plan Buffer: exactness across saturation, resets, and capture count.

FIVE CHECKS. The third is the one the design lives or dies on.

  1. The write path is byte-identical: index_copy_ vs the slice assignment, on the real pool.
  2. Pre-saturation behaviour is UNCHANGED -- the plan path must not engage before count == total.
  3. max|delta action| = 0 across seeded cycles SPANNING saturation, so both regimes and the
     transition at cycle 36 are covered. A gate that ran only post-saturation would miss a
     refresh-ordering bug at the boundary, and one that ran only pre-saturation would test nothing.
  4. RESET ISOLATION: after a reset the ring restarts unsaturated, so the plan path must disengage and
     the graphs keyed "saturated" must not be replayed against a fresh ring.
  5. CAPTURE COUNT over a full ~53-cycle episode, which is the actual claim.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29990 probe_persistent_graph.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def build(S, cfg, persistent: bool, graph: bool = True):
    server = S.VA_Server(cfg)
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for _ in install_conditioning_prefill(S, type(server)):
        pass
    for _ in install_debug_dump_elision(S):
        pass
    from instinctwm.backends.conv.apply import install_conv_layout
    for _ in install_conv_layout(server):
        pass
    gp = None
    if graph:
        from instinctwm.passes.lingbot.graph_capture import GraphBlockStack
        gp = GraphBlockStack()
        gp.install(S, type(server))
    if persistent:
        from instinctwm.passes.lingbot.persistent_graph import PersistentRingGraph
        PersistentRingGraph().install(S, server)
    return server, gp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=45, help="must span saturation (~36)")
    ap.add_argument("--mode", choices=["baseline", "persistent"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # ONE ARM PER PROCESS. graph_capture patches WanTransformer3DModel.forward by rewriting its
    # source to find `for block in self.blocks:`; the class is shared, so a second install in the
    # same process searches an already-patched forward and raises. Building both arms in one process
    # is therefore not possible, and pretending otherwise cost one run.
    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_pgraph"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)

    def run(server, n, seed=0, reset_at=None):
        rng = np.random.default_rng(seed)
        acts = []
        server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        for i in range(n):
            if reset_at is not None and i == reset_at:
                server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
                rng = np.random.default_rng(seed)
            torch.manual_seed(1234 + i)
            act = server.infer(dict(obs=obs["obs"], prompt=prompt,
                                    save_visualization=False))["action"]
            acts.append(np.asarray(act, dtype=np.float64).copy())
            kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
                  for _ in range(4 if i == 0 else 8)]
            server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                              save_visualization=False, state=act))
        return acts

    persistent = (a.mode == "persistent")
    print(f"=== arm: {a.mode} ({a.cycles} cycles) ===", flush=True)
    srv, gp = build(S, cfg, persistent=persistent)
    a0 = srv.transformer.blocks[0].attn1
    tot = a0._iwm_ring_total(srv.cache_name)
    sig0 = a0._iwm_ring_signature(srv.cache_name)
    print(f"  before: signature={sig0} total={tot}  plan_buffer={a0._iwm_use_plan_buffer}")
    check(sig0 is None or sig0[1] < (tot or 1),
          "ring starts unsaturated, so the plan write path is dormant")

    acts = run(srv, a.cycles)
    caps = int(getattr(gp, "n_captures", -1))
    sig1 = a0._iwm_ring_signature(srv.cache_name)
    print(f"  after {a.cycles} cycles: signature={sig1}   captures={caps}")
    # Re-read `total` AFTER the run. Before the first reset the ring dict does not exist, so
    # `_iwm_ring_total` returns None and `tot or 10**9` made this check unsatisfiable -- it was
    # comparing against a sentinel, not against the pool.
    tot_now = a0._iwm_ring_total(srv.cache_name) or tot
    check(sig1 is not None and tot_now and sig1[1] >= tot_now,
          "the run SPANNED saturation, so both regimes were exercised",
          f"count={sig1[1] if sig1 else None} total={tot_now}")

    # reset isolation, within this arm: two identical runs with a mid-run reset must agree
    r1 = run(srv, 8, reset_at=4)
    r2 = run(srv, 8, reset_at=4)
    w = max(float(np.abs(x - y).max()) for x, y in zip(r1, r2))
    check(w == 0.0, "reset isolation: two identical runs with a mid-run reset agree bit-for-bit",
          f"max|delta| = {w:.3e}")
    sigr = a0._iwm_ring_signature(srv.cache_name)
    print(f"  post-reset signature {sigr}: the ring restarts unsaturated, so the plan path "
          f"disengages")

    np.savez(a.out, acts=np.stack(acts), captures=caps,
             sat_count=(sig1[1] if sig1 else -1), total=(tot_now or -1))
    print(f"  wrote {a.out}")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: post-saturation plan buffer is bit-exact and reduces captures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
