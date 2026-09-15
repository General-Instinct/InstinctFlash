#!/usr/bin/env python3
"""GPU A/B for `action_terminal_forward_elision`: ON vs ELIDED through the ring wrap, ABBA, per-cycle.

THE CLAIM UNDER TEST. Skipping the action pred-commit forward (wan_va_server.py:542-546) and replaying
only its slot allocation is bit-exact PAST ring saturation (wrap at cycle 36 = 9792 / 272). The naive
skip was refuted on 2026-08-09 exactly there (0 through cycle ~37, then 0.0297..0.406), so a run that
stops before the wrap tests nothing; this one runs >= 45 cycles per arm and reports EVERY cycle.

PROTOCOL. One server process per (chain, operating point). `--deterministic-seed` serving
(install_deterministic_seed: torch seeded per chunk from frame_st_id, the same installer serve_variant
uses), identical observation streams (seeded rng), four episodes in ABBA order:

    ON  ->  ELIDED  ->  ELIDED  ->  ON

so latency is order-controlled and bit-exactness has a repeatability control built in: ON1 vs ON2 and
ELIDED1 vs ELIDED2 must be 0.000e+00 or the harness cannot decide anything and says NOT EVALUABLE.

VERDICT. max over cycles and arm pairs of max|delta action| == 0.0 -> BITEXACT (a bit-exact pass needs
no closed-loop certificate; this artifact is the evidence). Any nonzero -> the cycle and magnitude are
reported, the tier stays NUMERIC/SCREEN and the flag stays off by default.

CHAINS.  stock    = P001 substrate only (no-fsdp, no-empty-cache, no-debug-dump): the STOCK mask
                    allocator, so the pass runs its stock bookkeeping variant.
         shipped  = P001 + --conditioning-prefill --ring-kv --conv-layout: the ring bookkeeping variant.
POINTS.  2v4a_w5  = the certified default (10 forwards/cycle, batch 2); 2v2a_w1 = guidance-off (8, batch 1).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY -u -m torch.distributed.run \
        --nproc_per_node 1 --master_port 29977 probe_action_terminal_elision.py \
        --chain shipped --point 2v4a_w5 --cycles 48 --out /path/result.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_action_terminal_forward_elision, install_allocator_churn_elision,
    install_conditioning_prefill, install_debug_dump_elision, install_deterministic_seed,
    install_fsdp_elision, install_ring_kv_addressing,
)

POINTS = {"2v4a_w5": (2, 4, 5.0), "2v2a_w1": (2, 2, 1.0)}
CTX_DIR = os.environ.get("IFL_CTX_DIR", "/home/ubuntu/iwm_results/pdd_ctx50")


def _git_head(root):
    if os.environ.get("IFL_HEAD"):          # a synced tree without .git: the launcher states the head
        return os.environ["IFL_HEAD"]
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", choices=("stock", "shipped"), required=True)
    ap.add_argument("--point", choices=tuple(POINTS), default="2v4a_w5")
    ap.add_argument("--cycles", type=int, default=48, help=">= 45 so the wrap at ~36 is crossed")
    ap.add_argument("--seed", type=int, default=1234, help="--deterministic-seed value")
    ap.add_argument("--obs-seed", type=int, default=3)
    ap.add_argument("--out", default=None, help="JSON artifact path")
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_ate"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1

    nfe_v, nfe_a, w_video = POINTS[a.point]
    cfg.num_inference_steps, cfg.action_num_inference_steps = nfe_v, nfe_a
    if w_video != cfg.guidance_scale:
        from instinctflash.adapters.lingbot_va import apply_declared_guidance
        applied = apply_declared_guidance(cfg, {"video": w_video})
        assert applied.get("video") == w_video, applied

    # --- the chain, through the SAME installers serve_variant.py uses -------------------------------
    applied = []
    applied += install_fsdp_elision(S)
    applied += install_allocator_churn_elision(S)
    applied += install_debug_dump_elision(S)
    if a.chain == "shipped":
        applied += install_conditioning_prefill(S, S.VA_Server)
        applied += install_ring_kv_addressing(S, S.VA_Server)
    applied += install_action_terminal_forward_elision(S, S.VA_Server)
    applied += install_deterministic_seed(S, a.seed)

    print(f"chain={a.chain}  point={a.point}  (video {nfe_v} / action {nfe_a} steps, w_video={cfg.guidance_scale}, "
          f"video_exec_step={cfg.video_exec_step})", flush=True)
    print(f"applied: {applied}", flush=True)
    print("building server ...", flush=True)
    server = S.VA_Server(cfg)
    if a.chain == "shipped":
        from instinctflash.backends.conv.apply import install_conv_layout
        for line in install_conv_layout(server, prefer_bitexact=False, model_id="lingbot-va"):
            print(f"conv layout: {line}", flush=True)

    from instinctflash.passes.lingbot.action_terminal_elision import controller_of

    ctx = sorted(Path(CTX_DIR).glob("*.npz"))
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    first_obs = [{full: z[s] for s, full in short.items()}]
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)

    # transformer CALLS per cycle, counted at the instance (above the pass's class wrapper, so an elided
    # call is counted too); rows store calls minus elided = forwards actually executed
    tf = server.transformer
    n_fwd = {"n": 0}
    _cls_forward = type(tf).forward

    def counting_forward(*args, **kwargs):
        n_fwd["n"] += 1
        return _cls_forward(tf, *args, **kwargs)

    tf.forward = counting_forward

    def ring_state():
        attn = tf.blocks[0].attn1
        sig = getattr(attn, "_iwm_ring_signature", None)
        if sig is not None and sig(server.cache_name) is not None:
            start, count = sig(server.cache_name)
            return {"start": int(start), "count": int(count)}
        c = attn.attn_caches.get(server.cache_name)
        return {"live": int(c["mask"].sum().item())} if c else {}

    def episode(name, elide):
        ctl = controller_of(server)
        ctl.requested = elide
        before = dict(ctl.stats())
        rng = np.random.default_rng(a.obs_seed)
        rows = []
        server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        for i in range(a.cycles):
            n_fwd["n"] = 0
            e0 = ctl.n_elided
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            act = server.infer(dict(obs=first_obs, prompt=prompt, save_visualization=False))["action"]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
                  for _ in range(4 if i == 0 else 8)]
            server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                              save_visualization=False, state=act))
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            rows.append({
                "cycle": i, "infer_ms": (t1 - t0) * 1e3, "kv_ms": (t2 - t1) * 1e3,
                "cycle_ms": (t2 - t0) * 1e3, "forwards": n_fwd["n"] - (ctl.n_elided - e0),
                "elided": ctl.n_elided - e0, "ring": ring_state(),
                "action": np.asarray(act, dtype=np.float64).copy(),
            })
        after = dict(ctl.stats())
        print(f"  episode {name:8s} elide={elide!s:5s}  cycles={a.cycles}  "
              f"median cycle {statistics.median(r['cycle_ms'] for r in rows):.1f} ms  "
              f"median infer {statistics.median(r['infer_ms'] for r in rows):.1f} ms  "
              f"forwards executed/cycle {rows[-1]['forwards']}  elided {after['elided'] - before['elided']}  "
              f"materialized {after['materialized'] - before['materialized']}  "
              f"disabled={after['disabled_reason']!r}", flush=True)
        return rows, after

    # A FULL-LENGTH warm-up episode, discarded. Three cycles were not enough: the first 48-cycle episode
    # after build ran ~900 ms/cycle for cycles 3-41 and then settled at 187 (every later episode flat), so
    # whichever arm goes first would have absorbed a one-time transient into an 80% "drift".
    print(f"\nwarm-up: {a.cycles} cycles, elision off (discarded)", flush=True)
    ctl0 = controller_of(server)
    ctl0.requested = False
    episode("warm", False)

    print(f"\nABBA, {a.cycles} cycles per episode (wrap at ~36):", flush=True)
    eps = {}
    stats = {}
    for name, elide in (("ON1", False), ("EL1", True), ("EL2", True), ("ON2", False)):
        eps[name], stats[name] = episode(name, elide)

    def delta(x, y):
        return [float(np.abs(r1["action"] - r2["action"]).max()) for r1, r2 in zip(eps[x], eps[y])]

    rep_on, rep_el = delta("ON1", "ON2"), delta("EL1", "EL2")
    pairs = {p: delta(*p.split("-")) for p in ("ON1-EL1", "ON1-EL2", "ON2-EL1", "ON2-EL2")}
    worst = max(max(v) for v in pairs.values())
    evaluable = max(rep_on) == 0.0 and max(rep_el) == 0.0

    print(f"\n{'=' * 100}\nPER-CYCLE max|delta action| (ON1 vs EL1), with the ring position seen by the ON arm\n{'=' * 100}")
    for i in range(a.cycles):
        r = eps["ON1"][i]
        print(f"  cycle {i:2d}  {pairs['ON1-EL1'][i]:.3e}   ring {r['ring']}   "
              f"forwards ON {r['forwards']} / EL {eps['EL1'][i]['forwards']}   "
              f"elided {eps['EL1'][i]['elided']}")

    def med(name, key, lo=0, hi=None):
        return statistics.median(r[key] for r in eps[name][lo:hi])

    n = a.cycles
    lat = {}
    for key in ("cycle_ms", "infer_ms", "kv_ms"):
        lat[key] = {
            "ON": (med("ON1", key) + med("ON2", key)) / 2, "EL": (med("EL1", key) + med("EL2", key)) / 2,
            "ON_early": (med("ON1", key, 1, 13) + med("ON2", key, 1, 13)) / 2,
            "EL_early": (med("EL1", key, 1, 13) + med("EL2", key, 1, 13)) / 2,
            "ON_sat": (med("ON1", key, 36, n) + med("ON2", key, 36, n)) / 2,
            "EL_sat": (med("EL1", key, 36, n) + med("EL2", key, 36, n)) / 2,
            "drift_ON": abs(med("ON1", key) - med("ON2", key)) / med("ON1", key),
            "drift_EL": abs(med("EL1", key) - med("EL2", key)) / med("EL1", key),
        }

    print(f"\n{'=' * 100}\nLATENCY (ABBA means of per-episode medians; early = cycles 1-12, saturated = 36-{n - 1})\n{'=' * 100}")
    for key in ("cycle_ms", "infer_ms", "kv_ms"):
        L = lat[key]
        print(f"  {key:9s} ON {L['ON']:7.1f}  EL {L['EL']:7.1f}  saved {L['ON'] - L['EL']:+6.1f} ms "
              f"({L['ON'] / L['EL']:.3f}x)   early ON {L['ON_early']:.1f} EL {L['EL_early']:.1f}   "
              f"sat ON {L['ON_sat']:.1f} EL {L['EL_sat']:.1f}   drift ON {L['drift_ON']:.1%} EL {L['drift_EL']:.1%}")
    fwd_on, fwd_el = eps["ON1"][-1]["forwards"], eps["EL1"][-1]["forwards"]
    print(f"  forwards executed/cycle ON {fwd_on} -> EL {fwd_el}  (saved {fwd_on - fwd_el} of {fwd_on})")
    if lat["cycle_ms"]["drift_ON"] > 0.05 or lat["cycle_ms"]["drift_EL"] > 0.05:
        print("  LATENCY NOT EVALUATED: a same-arm repeat drifted > 5% (bit-exactness is unaffected)")

    print(f"\n{'=' * 100}\nVERDICT\n{'=' * 100}")
    print(f"  repeatability  ON1 vs ON2 max {max(rep_on):.3e}   EL1 vs EL2 max {max(rep_el):.3e}")
    for p, v in pairs.items():
        print(f"  {p}: max {max(v):.3e}" + (f"  first nonzero at cycle {v.index(next(x for x in v if x > 0))}"
                                          if max(v) > 0 else ""))
    materialized = stats["ON2"]["materialized"]
    disabled = stats["ON2"]["disabled_reason"]
    variant = stats["ON2"]["variant"]
    print(f"  elision variant {variant!r}   materialized {materialized}   disabled {disabled!r}")
    if not evaluable:
        tier, code = "NOT EVALUABLE (same-arm repeat is not bit-exact; the harness cannot decide)", 2
    elif materialized or disabled:
        tier, code = "NOT EVALUATED (the pass disabled itself on this message pattern)", 2
    elif worst == 0.0:
        tier, code = "BITEXACT through the wrap: max|delta action| = 0.000e+00 on every cycle and arm pair", 0
    else:
        first = min((v.index(next(x for x in v if x > 0)), p) for p, v in pairs.items() if max(v) > 0)
        tier, code = (f"NUMERIC: max|delta action| = {worst:.3e}; first nonzero at cycle {first[0]} "
                      f"({first[1]}); the flag stays off by default"), 1
    print(f"  => {tier}")

    if a.out:
        art = {
            "chain": a.chain, "point": a.point, "cycles": a.cycles, "seed": a.seed, "obs_seed": a.obs_seed,
            "applied": applied, "guidance_scale": float(cfg.guidance_scale),
            "video_exec_step": int(cfg.video_exec_step), "prompt": prompt, "context": ctx[0].name,
            "host": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "instinctflash_head": _git_head(IFL_ROOT),
            "verdict": tier, "exit_code": code, "worst": worst, "evaluable": evaluable,
            "repeat_on": rep_on, "repeat_el": rep_el, "pairs": pairs,
            "controller": stats, "latency": lat, "forwards_per_cycle": {"ON": fwd_on, "EL": fwd_el},
            "episodes": {k: [{kk: vv for kk, vv in r.items() if kk != "action"} for r in v]
                         for k, v in eps.items()},
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(art, indent=1))
        np.savez_compressed(str(Path(a.out).with_suffix(".actions.npz")),
                            **{k: np.stack([r["action"] for r in v]) for k, v in eps.items()})
        print(f"  artifact: {a.out} (+ .actions.npz)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
