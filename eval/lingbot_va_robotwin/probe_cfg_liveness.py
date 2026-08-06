#!/usr/bin/env python3
"""CFG branch-1 liveness on the action stream: OUTPUT liveness vs SIDE-EFFECT liveness.

Config: guidance_scale=5 (video, uses both branches), action_guidance_scale=1 (action, takes
`[:1]`, so branch 1's OUTPUT is discarded). That alone does not make the branch dead: both batch
elements write the SHARED ring KV pool, and the video stream reads branch 1.

  A. OUTPUT liveness      corrupt branch 1 of the returned tensor, AFTER the forward, leaving all
                          state writes intact. Non-zero delta => the output is consumed.
  B. SIDE-EFFECT liveness suppress branch 1's writes into the shared KV pool during action
                          forwards, leaving the returned value untouched. Non-zero delta => the
                          computation is load-bearing through state.

Dead, and legally elidable, only if BOTH are bit-exact.

RESULT on LingBot-VA / RoboTwin (2026-08-02):

    OUTPUT liveness      max|d| = 5.640625   over 3 chunks
    SIDE-EFFECT liveness max|d| = 5.390625   over 3 chunks
    (chunk-to-chunk movement for scale: 1.031250)

Both axes LIVE. CFG elision on the action stream is NOT legal, and
`guidance = {"action": POSITIVE_ONLY}` must not be read as licence to remove the compute.

An honest caveat on the OUTPUT axis: the source says `action_noise_pred[:1]` under
`action_guidance_scale = 1`, which reads as though branch 1's return is discarded, yet corrupting
it moves the result. The measurement and the reading disagree and the reading is the one that is
unverified, so the conservative conclusion stands. Anyone revisiting elision must explain that
5.64 first.

    python probe_cfg_liveness.py
"""
import os, sys
from pathlib import Path

# Resolved from this file rather than written down, matching serve_variant.py and
# profile_cycle.py. The hardcoded /home/ubuntu/InstinctWM this replaced does not exist
# any more -- the tree moved to /home/ubuntu/Code/InstinctWM -- so the import of
# profile_cycle below could not have resolved. IWM_ROOT still wins when it is set.
_HERE = Path(__file__).resolve().parent
IWM_ROOT = os.environ.get("IWM_ROOT") or str(_HERE.parents[1])
for _p in (IWM_ROOT, str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np, torch
from profile_cycle import build_server, drive

CKPT, PROMPT = os.environ["LINGBOT_CKPT"], "Use the left arm to lift the plastic drink bottle head-up"
srv, S = build_server(CKPT, no_fsdp=True, prefill=True)
from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
RingKVAddressing().install(S, S.VA_Server)
import modules.model as M

cfg = srv.job_config
print(f"guidance_scale={cfg.guidance_scale}  action_guidance_scale={cfg.action_guidance_scale}")

MODE = {"m": "base", "in_action": False}
ACTS = []

# ---- capture the produced actions -----------------------------------------------------------
_orig_infer = type(srv)._infer
def infer(self, obs, frame_st_id=0):
    out = _orig_infer(self, obs, frame_st_id=frame_st_id)
    def flat(v):
        if isinstance(v, torch.Tensor):
            return v.detach().float().cpu().numpy().ravel()
        if isinstance(v, dict):
            return np.concatenate([flat(x) for x in v.values()]) if v else np.zeros(0)
        if isinstance(v, (list, tuple)):
            return np.concatenate([flat(x) for x in v]) if len(v) else np.zeros(0)
        return np.asarray(v, dtype=np.float64).ravel()
    a = out["action"] if isinstance(out, dict) and "action" in out else out
    ACTS.append(flat(a).astype(np.float64))
    return out
type(srv)._infer = infer

# ---- A: corrupt branch 1 of the RETURNED value, state writes already done ---------------------
Model = M.WanTransformer3DModel
_orig_fwd = Model.forward
def model_forward(self_m, input_dict, update_cache=0, cache_name="pos",
                  action_mode=False, train_mode=False):
    MODE["in_action"] = action_mode
    out = _orig_fwd(self_m, input_dict, update_cache, cache_name, action_mode, train_mode)
    MODE["in_action"] = False
    if MODE["m"] == "A" and action_mode and isinstance(out, torch.Tensor) and out.shape[0] > 1:
        out = out.clone()
        out[1] = out[1] + 1000.0
    return out
Model.forward = model_forward

# ---- B: suppress branch 1's writes into the shared KV pool during action forwards -------------
Attn = M.WanAttention
_orig_attn = Attn.forward
def attn_forward(self_a, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
    snap = None
    if MODE["m"] == "B" and MODE["in_action"]:
        c = (self_a.attn_caches or {}).get(cache_name)
        r = (c or {}).get("_ring")
        if c is not None and r is not None and c.get("k") is not None and c["k"].shape[0] > 1:
            n = k.shape[1]
            head = (r["start"] + r["count"]) % r["total"]
            sl = slice(head, head + n)
            snap = (sl, c["k"][1, sl].clone(), c["v"][1, sl].clone())
    out = _orig_attn(self_a, q, k, v, rotary_emb, update_cache, cache_name)
    if snap is not None:
        sl, k1, v1 = snap
        c = self_a.attn_caches[cache_name]
        c["k"][1, sl] = k1          # roll back branch 1's contribution to shared state
        c["v"][1, sl] = v1
    return out
Attn.forward = attn_forward


def run(mode, cycles=3):
    MODE["m"] = mode
    ACTS.clear()
    rng = np.random.default_rng(0)
    srv._reset(prompt=PROMPT)
    drive(srv, rng, cycles, PROMPT)
    return [a.copy() for a in ACTS]


base = run("base")
print(f"captured {len(base)} action chunks per run")
res = {}
for mode, label in (("A", "OUTPUT liveness (branch-1 return corrupted)"),
                    ("B", "SIDE-EFFECT liveness (branch-1 KV writes suppressed)")):
    got = run(mode)
    worst = max(np.abs(a - b).max() for a, b in zip(base, got))
    res[mode] = worst
    print(f"\n{label}\n  max|d| over {len(base)} chunks = {worst:.6e}   "
          f"{'no effect' if worst == 0 else 'AFFECTS THE OUTPUT'}")

print("\n" + "=" * 70)
if res["A"] == 0 and res["B"] == 0:
    print("VERDICT: branch 1 is DEAD on both axes -- elision is legal.")
elif res["A"] == 0:
    print("VERDICT: output unused, but the COMPUTATION IS LOAD-BEARING through shared state.\n"
          "         POSITIVE_ONLY is true about the output and cannot justify elision.")
else:
    print("VERDICT: branch 1's output IS consumed. The descriptor is wrong.")
