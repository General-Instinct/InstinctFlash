#!/usr/bin/env python3
"""Does the timestep-modulation cast belong to a FAMILY of scope-lifting optimizations, or is it one bug?

P004 (`hoist_invariant_casts`) hoisted casts of WEIGHTS from per-forward to per-episode scope: 7,110
casts of a constant removed per cycle. Candidate 4 (LAYER5_NEXT.md) looks like the same shape applied
to ACTIVATIONS -- `temb.float()` at model.py:524 runs once per block but `temb` is the same tensor for
all 30 blocks. If that generalizes, Layer 5 contains a family and the abstraction should be built
before the instance. If it does not, Candidate 4 is a one-off and should just be fixed.

READING THE CODE CANNOT ANSWER THIS. A cast is redundant only if its INPUT VALUE is unchanged across
the repeats, and that is a runtime property. So this measures, per `_to_copy` callsite:

    calls per forward       how often it runs
    distinct inputs         how many different values it actually casts
    lifetime               the coarsest scope at which the value is constant

The classification that matters:

    distinct == calls              the value changes every call -> NOT hoistable, the work is real
    distinct == 1 per forward      one value re-cast N times per forward -> hoistable to STEP scope
    distinct == 1 per cycle        -> hoistable to CYCLE scope
    distinct == 1 per episode      -> hoistable to EPISODE scope (this is P004's case)

IDENTITY IS CHECKED TWO WAYS, because they answer different questions. A storage digest
(data_ptr, offset, shape, stride, dtype, version) proves the calls received the SAME TENSOR -- the
cheapest and most defensible evidence of redundancy. A value digest catches the case where a tensor is
rebuilt each block at a new address with equal contents: still redundant, but only provably so by
value. Both are reported, because a hoist justified by value equality alone needs a stronger argument
than one justified by object identity.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$IWM_FA_SHIM_DIR $IWM_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29993 \\
        probe_cast_lifetime.py [--warm 70]
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import traceback
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from instinctwm.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_conditioning_prefill, install_debug_dump_elision,
    install_fsdp_elision,
)

WATCH = ("_to_copy", "to", "copy_")


class Tracker(TorchDispatchMode):
    """Record, per callsite, the storage and value identity of every cast input, tagged by forward."""

    def __init__(self, value_digest: bool = True):
        super().__init__()
        self.value_digest = value_digest
        self.enabled = False
        self.fwd = 0                       # forward index within the cycle
        self.rows = collections.defaultdict(lambda: {
            "calls": 0, "per_fwd": collections.Counter(),
            "storage": collections.defaultdict(set), "value": collections.defaultdict(set),
            "shapes": collections.Counter(), "dtypes": collections.Counter()})

    def _site(self) -> str:
        for f in reversed(traceback.extract_stack()):
            fn = f.filename
            if "/torch/" in fn or "probe_cast_lifetime" in fn or "_python_dispatch" in fn:
                continue
            tag = ("iwm" if "/instinctwm/" in fn else
                   "lingbot" if "/wan_va/" in fn or "/lingbot" in fn else
                   "diffusers" if "diffusers" in fn else "app")
            return f"[{tag}] {Path(fn).name}:{f.lineno} {f.name}"
        return "[?] unattributed"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func).split(".")[-2] if "." in str(func) else str(func)
        if not self.enabled or name not in WATCH:
            return func(*args, **kwargs)
        t = args[0] if args and hasattr(args[0], "shape") else None
        if t is None:
            return func(*args, **kwargs)
        site = self._site()
        r = self.rows[site]
        r["calls"] += 1
        r["per_fwd"][self.fwd] += 1
        r["shapes"][tuple(t.shape)] += 1
        r["dtypes"][str(t.dtype).replace("torch.", "")] += 1
        try:
            # Storage identity: same tensor, unmodified. The strongest and cheapest evidence.
            r["storage"][self.fwd].add((t.data_ptr(), t.storage_offset(), tuple(t.shape),
                                        tuple(t.stride()), str(t.dtype),
                                        t._version if hasattr(t, "_version") else 0))
            if self.value_digest and t.numel():
                # Value identity: catches a tensor rebuilt at a new address with equal contents.
                # A small fixed slice keeps this affordable across thousands of calls.
                flat = t.detach().reshape(-1)[:64].double()
                r["value"][self.fwd].add((round(float(flat.sum()), 10),
                                          round(float(flat.abs().max()), 10), t.numel()))
        except Exception:
            pass
        return func(*args, **kwargs)


def classify(calls_per_fwd: float, distinct_storage: float, distinct_value: float,
             distinct_per_cycle: int, n_fwd: int) -> tuple[str, str]:
    """(lifetime, minimal legal scope). Classified on the VALUE digest, never on storage.

    THE STORAGE DIGEST IS NOT EVIDENCE OF REDUNDANCY, and the first version of this probe treated it
    as though it were. (data_ptr, offset, shape, stride, dtype) repeats whenever the CACHING ALLOCATOR
    hands the same address to a later block -- which it does aggressively, because each block frees its
    temporaries before the next allocates. So model.py:543 showed 3.5 distinct storages per forward
    against 30 distinct VALUES: the same address, thirty different tensors. Ranking on storage produced
    "4484 calls/cycle removable" when the true figure is 290.

    Storage identity is still worth reporting -- when it agrees with value identity it is stronger
    evidence, because the calls provably received the same object -- but it can only ever corroborate,
    never establish.
    """
    if calls_per_fwd <= 1.01:
        return "per-forward (already minimal)", "STEP"
    if distinct_value != distinct_value:                    # NaN: no value digest collected
        return "UNKNOWN (no value digest)", "NOT DETERMINED"
    if distinct_value <= 1.01:
        agree = distinct_storage <= 1.01
        if distinct_per_cycle <= 1:
            return ("per-cycle" + ("" if agree else ", by value only"), "CYCLE")
        return ("per-forward" + ("" if agree else ", by value only"), "STEP")
    if distinct_value >= calls_per_fwd * 0.95:
        return "per-call (value differs every call)", "LAYER (already minimal)"
    return (f"partially redundant ({distinct_value:.1f} distinct of {calls_per_fwd:.0f})",
            "STEP (partial)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=70)
    ap.add_argument("--cycles", type=int, default=2, help="cycles to track")
    ap.add_argument("--no-value-digest", action="store_true")
    a = ap.parse_args()

    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IWM_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_cast_life"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda *x, **k: None
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
    print("building server ...", flush=True)
    server = S.VA_Server(cfg)
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    for n in install_conditioning_prefill(S, type(server)):
        print(f"  installed {n}", flush=True)
    for n in install_debug_dump_elision(S):
        print(f"  installed {n}", flush=True)
    from instinctwm.backends.conv.apply import install_conv_layout
    for line in install_conv_layout(server):
        print(f"  {line}", flush=True)

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    if not ctx:
        raise SystemExit("no contexts; run collect_contexts.sh")
    z = np.load(ctx[0], allow_pickle=True)
    short = {k.split(".")[-1]: k for k in cfg.obs_cam_keys}
    obs = {"obs": [{full: z[s] for s, full in short.items()}], "state": z["state"]}
    prompt = str(z["prompt"])
    cams = list(cfg.obs_cam_keys)
    rng = np.random.default_rng(0)
    first = {"v": True}

    trk = Tracker(value_digest=not a.no_value_digest)

    # Count forwards so "per forward" is measured rather than assumed. The transformer's forward is
    # the boundary: 30 blocks run inside one, so a value constant across a forward is re-cast 30x.
    orig_fwd = server.transformer.forward

    def counting_fwd(*args, **kw):
        if trk.enabled:
            trk.fwd += 1
        return orig_fwd(*args, **kw)
    server.transformer.forward = counting_fwd

    def cycle():
        if first["v"]:
            server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        act = server.infer(dict(obs=obs["obs"], prompt=prompt,
                                save_visualization=False))["action"]
        kf = [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
              for _ in range(4 if first["v"] else 8)]
        server.infer(dict(obs=kf, compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=act))
        first["v"] = False
        return act

    print(f"warming {a.warm} cycles ...", flush=True)
    for _ in range(a.warm):
        cycle()

    print(f"tracking {a.cycles} cycles (instrumented, slow) ...", flush=True)
    with trk:
        trk.enabled = True
        for _ in range(a.cycles):
            cycle()
        trk.enabled = False
    n_fwd = trk.fwd

    rows = []
    for site, r in trk.rows.items():
        fwds = [f for f, c in r["per_fwd"].items() if c]
        if not fwds:
            continue
        calls_per_fwd = r["calls"] / max(len(fwds), 1)
        dsp = sum(len(r["storage"][f]) for f in fwds) / len(fwds)
        dvp = (sum(len(r["value"][f]) for f in fwds) / len(fwds)) if r["value"] else float("nan")
        all_storage = set().union(*(r["storage"][f] for f in fwds)) if fwds else set()
        lifetime, scope = classify(calls_per_fwd, dsp, dvp, len(all_storage), len(fwds))
        rows.append((r["calls"], site, calls_per_fwd, dsp, dvp, lifetime, scope,
                     r["shapes"].most_common(1)[0][0], len(r["shapes"])))
    rows.sort(reverse=True)

    print(f"\n{'=' * 124}")
    print(f"CAST LIFETIME BY CALLSITE   ({n_fwd} forwards over {a.cycles} cycles)")
    print(f"{'=' * 124}")
    print(f"{'calls':>7}{'/fwd':>6}{'distinct storage/fwd':>22}{'distinct value/fwd':>20}"
          f"  lifetime -> minimal scope")
    print("-" * 124)
    for calls, site, cpf, dsp, dvp, lifetime, scope, shp, nshapes in rows[:22]:
        dv = "n/a" if dvp != dvp else f"{dvp:.1f}"
        print(f"{calls:>7}{cpf:>6.0f}{dsp:>22.1f}{dv:>20}  {lifetime} -> {scope}")
        print(f"{'':>13}{site}   shape {shp}{'' if nshapes == 1 else f' (+{nshapes-1} more)'}")

    print(f"\n{'=' * 124}\nREDUNDANCY SUMMARY -- calls removable by hoisting to the minimal scope\n"
          f"{'=' * 124}")
    tot_removable = 0
    per_cycle = lambda n: n / a.cycles          # noqa: E731
    for calls, site, cpf, dsp, dvp, lifetime, scope, shp, nshapes in rows:
        if "already minimal" in scope or "already minimal" in lifetime:
            continue
        if dvp != dvp:                                  # no value digest -> cannot claim anything
            continue
        removable = max(0.0, calls - dvp * (calls / max(cpf, 1)))
        if removable < 1:
            continue
        tot_removable += removable
        print(f"  {per_cycle(removable):>8.0f} calls/cycle removable  ({lifetime})  {site}")
    print(f"  {'-' * 110}")
    print(f"  {per_cycle(tot_removable):>8.0f} calls/cycle removable in total, of "
          f"{per_cycle(sum(r[0] for r in rows)):.0f} tracked cast calls")
    print("\n  A family exists if this is spread over several INDEPENDENT callsites with the same")
    print("  lifetime pattern. If one site dominates, it is one bug and should be fixed as one.")
    print("\n  Note the storage/value columns. Where they DISAGREE (low storage, high value), the")
    print("  repetition is the caching allocator reusing an address, not a value being re-cast. Only")
    print("  the value column establishes redundancy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
