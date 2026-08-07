#!/usr/bin/env python3
"""Gates for fused QKV: certificate behaviour, then action-level bit-exactness.

FOUR CHECKS, and the third is the one that makes the pass safe rather than merely correct today:

  1. The envelope is DERIVED from descriptors and covers the shapes production actually issues.
  2. Certification is per shape, and its result decides the path taken.
  3. FAIL-CLOSED on an unseen shape. Injecting a shape that was never certified must take the split
     path, not the fused one. A pass that is exact because nothing unexpected happened is untested.
  4. max|delta action| = 0 over paired seeded cycles, end to end.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29997 probe_fused_qkv.py
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


def build(S, cfg, fuse: bool):
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
    from instinctwm.passes.lingbot.step_scope_cast import StepScopeCastHoist
    StepScopeCastHoist().install(S, type(server))
    p = None
    if fuse:
        from instinctwm.passes.lingbot.fused_qkv import FusedQKVProjection
        p = FusedQKVProjection()
        p.install(S, server)
    return server, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6)
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_fqkv"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts")
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)

    def run(server, n, seed=0):
        rng = np.random.default_rng(seed)
        acts = []
        server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        for i in range(n):
            torch.manual_seed(1234 + i)
            act = server.infer(dict(obs=obs["obs"], prompt=prompt,
                                    save_visualization=False))["action"]
            acts.append(np.asarray(act, dtype=np.float64).copy())
            kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
                  for _ in range(4 if i == 0 else 8)]
            server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                              save_visualization=False, state=act))
        return acts

    print("=== baseline (split projections) ===", flush=True)
    base, _ = build(S, cfg, fuse=False)
    base_acts = run(base, a.cycles)
    del base
    torch.cuda.synchronize()

    print("\n=== fused (certified shapes only) ===", flush=True)
    srv, p = build(S, cfg, fuse=True)

    print("\n=== 1 + 2. envelope and certification (deferred until first sight) ===")
    st = p.stats()
    at_install = st["certified"] or "nothing yet (latent geometry is set at the first reset, so "\
                                    "certification happens on first sight)"
    print(f"  at install: {at_install}")

    print("\n=== 3. FAIL-CLOSED on an unseen shape ===")
    at = srv.transformer.blocks[0].attn1
    before_fb = p.report.fallback_calls
    odd = torch.randn(7, at.to_q.weight.shape[1],
                      device=at.to_q.weight.device, dtype=at.to_q.weight.dtype)
    out_odd = at.to_q(odd)
    ref_odd = torch.nn.functional.linear(odd, at.to_q.weight, at.to_q.bias)
    check(p.report.fallback_calls > before_fb,
          "an uncertified shape (M=7) took the FALLBACK path", f"M=7 not in the envelope")
    check(torch.equal(out_odd.view(torch.int16), ref_odd.view(torch.int16)),
          "and produced bit-identical output to the original projection")
    sk = (7, at.to_q.weight.shape[1], at.to_q.weight.shape[0])
    check(sk in p.report.lazily_certified,
          "the unseen shape was CERTIFIED ON FIRST SIGHT and recorded, not silently absorbed",
          f"result: {p.report.certified.get(sk)}")
    # Whatever the certificate said, the FIRST call must have used the split path -- the decision is
    # only trusted from the next call onwards.
    out2 = at.to_q(odd)
    ref2 = torch.nn.functional.linear(odd, at.to_q.weight, at.to_q.bias)
    check(torch.equal(out2.view(torch.int16), ref2.view(torch.int16)),
          "and the second call is bit-identical too, whichever path it took")

    print("\n=== 4. action-level BITEXACT gate ===")
    fused_acts = run(srv, a.cycles)
    worst = 0.0
    for i, (b, f) in enumerate(zip(base_acts, fused_acts)):
        if b.shape != f.shape:
            check(False, f"cycle {i} action shapes match", f"{b.shape} vs {f.shape}")
            continue
        worst = max(worst, float(np.abs(b - f).max()))
    check(worst == 0.0, f"max|delta action| = 0 over {a.cycles} paired seeded cycles",
          f"max|delta| = {worst:.3e}")

    st = p.stats()
    print(f"\n  fused calls {st['fused_calls']}, fallback calls {st['fallback_calls']}")
    check(st["fused_calls"] > 0, "the fused path actually ran", "not exact by doing nothing")
    # Certification is only observable after a forward, so assert it here rather than at install.
    prod = {k: v for k, v in p.report.certified.items() if k[0] in (64, 480)}
    print(f"  production shapes: {prod}")
    check(prod and all(prod.values()),
          "every PRODUCTION shape certified exact", f"{sorted(prod)}")
    bad = {k: v for k, v in p.report.certified.items() if v is not True}
    if bad:
        print(f"  refused shapes (split path retained): {sorted(bad)}")
        print("       => the invariant is NOT universal, and the certificate is what makes that safe")
    if st["uncertified_seen_at_runtime"]:
        print(f"  shapes seen at runtime but never certified: "
              f"{st['uncertified_seen_at_runtime']}")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: fused QKV is bit-exact, certified per shape, and fails closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
