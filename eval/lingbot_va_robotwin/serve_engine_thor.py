#!/usr/bin/env python3
"""LingBot-VA on Thor — the ENGINE placement of the shipped chain (Stage-A hybrid, Stage 2).

Torch owns what it owned in every certified number: the umt5 prompt encode at reset, the Wan-VAE
streaming encoders at every kv message (the NUMERIC-certified ndhwc chain), the wire protocol
and action pre/post-processing. The flash_rt engine (``serving/flash_rt/frontends/torch/
wan_va_thor.py``) owns EVERY DiT forward of a control cycle and the KV slab. This script builds
the torch shipped chain EXACTLY as ``serve_variant.py`` does (same installers, same flags:
``--no-fsdp --no-empty-cache --no-debug-dump --conditioning-prefill --ring-kv --conv-layout
--degrade-nfe V,A [--guidance ...] [--deterministic-seed N]``), captures the constructed server,
and re-routes ``_infer`` / ``_compute_kv_cache`` to the engine.

THE OPERATING POINT IS DECLARED, NOT ASSUMED (h2_thor_realtime_design.md §5, engine_backend
9bf2337): the engine is built FOR the point the worker flags request; at every reset the point
the torch config RESOLVES to (``WanVaOperatingPoint.from_job_config``) is asserted against the
engine's baked point — a mismatch is an OperatingPointMismatch, never a silently different
computation. The declaration is printed at start-up and written to ``--ledger-out``.

Modes:
  serve  (default)  — websocket policy server, the same protocol as the torch chain; probe with
                      eval/lingbot_va_robotwin/probe_latency.py. Per-cycle ledger (VAE encode /
                      DiT / commit split) appended to --ledger-out as JSON lines.
  --parity NPZ      — in-process M2 harness: torch arm (the shipped chain) vs engine arm on the
                      SAME real-observation episode scripts (thor_va_engine/tools/
                      build_va_obs_set.py), matched noise, recorded VAE latents / prompt embeds
                      handed to the engine; per-cycle deltas, digests, perturbation rung, broken
                      controls, fp8 calibration on a declared subset. No websocket.

Launch like the arm runner (torchrun, single rank), e.g. on Thor:
  cd ~/lingbot-va && PYTHONPATH=$HOME/iwm_shims:$HOME/FlashRT-fork LINGBOT_CKPT=... IFL_ROOT=~/InstinctFlash_thor \
    ~/venv_va/bin/python -m torch.distributed.run --nproc_per_node 1 --master_port 29957 \
    ~/InstinctFlash_thor/eval/lingbot_va_robotwin/serve_engine_thor.py --config-name robotwin --port 29157 \
    --degrade-nfe 2,2 --guidance video=positive_only --engine-precision fp8 --act-scales scales.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)
FLASH_RT_ROOT = os.environ.get("FLASH_RT_ROOT") or os.path.expanduser("~/FlashRT-fork")
if os.path.isdir(FLASH_RT_ROOT) and FLASH_RT_ROOT not in sys.path:
    sys.path.insert(0, FLASH_RT_ROOT)
_SERVING = str(Path(IFL_ROOT) / "serving")
if os.path.isdir(_SERVING) and _SERVING not in sys.path:
    sys.path.append(_SERVING)          # after the fork tree: the fork's built .so wins on Thor

import numpy as np  # noqa: E402
import torch  # noqa: E402

CAMS = ["observation.images.cam_high", "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]
SHIPPED_FLAGS = ["--no-fsdp", "--no-empty-cache", "--no-debug-dump", "--conditioning-prefill",
                 "--ring-kv", "--conv-layout", "--action-terminal-elision"]   # P010 (d7e6103) is served


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                          ).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════
# 1. Build the torch shipped chain through serve_variant (captured, not served)
# ══════════════════════════════════════════════════════════════════

def build_torch_chain(args, extra_sv_args):
    """Run serve_variant.main() with the shipped flags and CAPTURE the constructed VA_Server
    (the websocket loop is intercepted). Returns (module S, model, port, orig_serve)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import serve_variant  # noqa: E402  (same directory)
    from instinctflash.runtime.lingbot_install import import_lingbot_server
    S = import_lingbot_server()
    captured = {}
    orig_serve = S.run_async_server_mode

    def _capture(model, local_rank, host, port):
        captured.update(model=model, host=host, port=port)

    S.run_async_server_mode = _capture
    flags = [f for f in SHIPPED_FLAGS if not (args.no_action_terminal_elision and f == "--action-terminal-elision")]
    sv_argv = ["serve_variant.py", "--config-name", args.config_name, "--port", str(args.port),
               *flags, "--degrade-nfe", args.degrade_nfe]
    if args.guidance:
        sv_argv += ["--guidance", args.guidance]
    if args.deterministic_seed is not None:
        sv_argv += ["--deterministic-seed", str(args.deterministic_seed)]
    if args.save_root:
        sv_argv += ["--save_root", args.save_root]
    sv_argv += list(extra_sv_args)
    old_argv = sys.argv
    sys.argv = sv_argv
    try:
        serve_variant.main()
    finally:
        sys.argv = old_argv
    if "model" not in captured:
        raise RuntimeError("serve_variant did not construct the server (see its output above)")
    return S, captured["model"], captured["host"], captured["port"], orig_serve


# ══════════════════════════════════════════════════════════════════
# 2. The engine placement (instance-level re-routing of the two DiT-bearing methods)
# ══════════════════════════════════════════════════════════════════

class EnginePlacement:
    """Re-route VA_Server._reset/_infer/_compute_kv_cache to the engine; keep the torch
    originals for the parity arm. Every method here mirrors the stock body line for line
    (wan_va_server.py:_infer :440-570, :_compute_kv_cache :572-604) with the DiT calls
    replaced by engine.infer_cycle / engine.commit_chunk."""

    def __init__(self, model, engine, seed, ledger_out=None, profile=False):
        from flash_rt.models.wan_va.operating_point import WanVaOperatingPoint
        self.model, self.engine, self.seed = model, engine, seed
        self.WanVaOperatingPoint = WanVaOperatingPoint
        self.ledger_out = ledger_out
        self.profile = profile
        self.ledger = []
        self.noise_override = None
        self.torch = {"_reset": model._reset, "_infer": model._infer,
                      "_compute_kv_cache": model._compute_kv_cache}
        self.enabled = False

    # -- switch ----------------------------------------------------------------
    def enable(self):
        m = self.model
        m._reset = self._reset
        m._infer = self._infer
        m._compute_kv_cache = self._compute_kv_cache
        self.enabled = True

    def disable(self):
        for k, v in self.torch.items():
            setattr(self.model, k, v)
        self.enabled = False

    # -- the three re-routed methods -------------------------------------------
    def _reset(self, prompt=None):
        self._flush_ledger()
        self.torch["_reset"](prompt=prompt)
        m = self.model
        served = self.WanVaOperatingPoint.from_job_config(m.job_config)
        # FAIL-CLOSED: the point the torch config resolves to must be the engine's baked point
        self.engine.assert_point(served, requester="this server's resolved job_config")
        if m.prompt_embeds is None:
            self.engine._prompt_set = False
            return
        if self.engine.B == 2:
            if m.negative_prompt_embeds is None:
                raise RuntimeError("cfg build but the torch chain produced no negative embedding")
            te = torch.cat([m.prompt_embeds, m.negative_prompt_embeds], dim=0)
        else:
            te = m.prompt_embeds
        self.engine.set_prompt(te.to(torch.bfloat16))

    def _draw_noise(self, frame_st_id):
        """Exactly the stock draws (server:449-462), seeded exactly like
        install_deterministic_seed when a seed is configured."""
        m = self.model
        if self.seed is not None:
            torch.manual_seed(self.seed + frame_st_id)
            torch.cuda.manual_seed_all(self.seed + frame_st_id)
        fcs = m.job_config.frame_chunk_size
        latents = torch.randn(1, 48, fcs, m.latent_height, m.latent_width,
                              device=m.device, dtype=m.dtype)
        actions = torch.randn(1, m.job_config.action_dim, fcs, m.action_per_frame, 1,
                              device=m.device, dtype=m.dtype)
        if self.noise_override is not None:
            latents, actions = self.noise_override
        return latents, actions

    @torch.no_grad()
    def _infer(self, obs, frame_st_id=0):
        m = self.model
        t0 = time.perf_counter()
        t_enc = 0.0
        if frame_st_id == 0:
            m.init_latent = m._encode_obs(obs)
            torch.cuda.synchronize()
            t_enc = (time.perf_counter() - t0) * 1000
        latents, actions = self._draw_noise(frame_st_id)
        t1 = time.perf_counter()
        a, lat = self.engine.infer_cycle(latents, actions,
                                         m.init_latent if frame_st_id == 0 else None)
        actions_np = m.postprocess_action(a)          # .cpu() inside = the sync point
        t2 = time.perf_counter()
        self.ledger.append({"ev": "infer", "frame_st_id": int(frame_st_id),
                            "encode_ms": t_enc, "dit_ms": (t2 - t1) * 1000,
                            "total_ms": (t2 - t0) * 1000,
                            **({"stage_ms": dict(self.engine.stage_ms)} if self.profile else {})})
        return actions_np, lat

    @torch.no_grad()
    def _compute_kv_cache(self, obs):
        m = self.model
        t0 = time.perf_counter()
        lat = m._encode_obs(obs)
        if m.frame_st_id == 0:
            lat = torch.cat([m.init_latent, lat], dim=2) if lat is not None else m.init_latent
        act = m.preprocess_action(obs["state"]).to(lat)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.engine.commit_chunk(lat, act)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.ledger.append({"ev": "kv", "frame_st_id": int(m.frame_st_id),
                            "encode_ms": (t1 - t0) * 1000, "dit_ms": (t2 - t1) * 1000,
                            "total_ms": (t2 - t0) * 1000, "F": int(lat.shape[2])})
        m.frame_st_id += lat.shape[2]

    def _flush_ledger(self):
        if self.ledger and self.ledger_out:
            with open(self.ledger_out, "a") as f:
                f.write(json.dumps({"episode_ledger": self.ledger,
                                    "declaration": self.engine.declaration()}) + "\n")
        self.ledger = []


# ══════════════════════════════════════════════════════════════════
# 3. Parity harness (M2): torch arm vs engine arm on real-observation episode scripts
# ══════════════════════════════════════════════════════════════════

def _decode(jpeg_bytes) -> np.ndarray:
    """JPEG bytes → RGB uint8 (H, W, 3); PIL (a diffusers dependency) — no cv2 on Thor."""
    import io
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(np.asarray(jpeg_bytes, dtype=np.uint8).tobytes())).convert("RGB"))


def _make_obs(frame3):
    """frame3: 3 JPEG byte arrays (cam_high, cam_left_wrist, cam_right_wrist) → the client's obs dict."""
    return {k: np.ascontiguousarray(_decode(frame3[i])) for i, k in enumerate(CAMS)}


def record_torch_episode(model, placement, ep, acts, prompt, cycles, perturb=None):
    """Run the TORCH shipped chain over one episode script, recording everything the engine
    arm needs: prompt embeds, init latent, per-cycle noise draws, commit inputs (VAE latents +
    preprocessed actions), and the normalized action output. ``perturb`` (optional) replaces
    each recorded noise draw by perturb(noise) — the ±1-ulp rung of the ladder."""
    placement.disable()
    rec = {"norm_actions": [], "noise": [], "commit": [], "digest": []}
    stash = {}
    orig_post = model.postprocess_action
    orig_prep = model._prepare_latent_input
    real_randn = torch.randn

    def post(a):
        stash["norm"] = a.detach().clone()
        return orig_post(a)

    def prep(latent_model_input, action_model_input, *a, **k):
        if latent_model_input is not None and action_model_input is not None:
            stash["commit"] = (latent_model_input.detach().clone(), action_model_input.detach().clone())
        return orig_prep(latent_model_input, action_model_input, *a, **k)

    draws = []

    def rec_randn(*a, **k):
        t = real_randn(*a, **k)
        if perturb is not None:
            t = perturb(t)
        draws.append(t.clone())
        return t

    model.postprocess_action = post
    model._prepare_latent_input = prep
    try:
        placement.torch["_reset"](prompt=prompt)
        rec["text_emb"] = (model.prompt_embeds.detach().clone(),
                           None if model.negative_prompt_embeds is None
                           else model.negative_prompt_embeds.detach().clone())
        for c in range(cycles):
            draws.clear()
            torch.randn = rec_randn
            try:
                obs0 = {"obs": [ep["obs0"]]}
                model._infer(obs0, frame_st_id=model.frame_st_id)   # class method (seeded)
            finally:
                torch.randn = real_randn
            assert len(draws) == 2, f"expected 2 noise draws, saw {len(draws)}"
            rec["noise"].append((draws[0].clone(), draws[1].clone()))
            rec["norm_actions"].append(stash["norm"].clone())
            rec["digest"].append(_digest(stash["norm"]))
            if c == 0:
                rec["init_latent"] = model.init_latent.detach().clone()
            # the REAL client protocol: 4 keyframes at cycle 0 (16 executed steps), 8 afterwards
            placement.torch["_compute_kv_cache"]({"obs": ep["kfs"][c], "state": acts[c],
                                                  "compute_kv_cache": True})
            rec["commit"].append(stash["commit"])
    finally:
        model.postprocess_action = orig_post
        model._prepare_latent_input = orig_prep
        torch.randn = real_randn
    return rec


def run_engine_episode(engine, rec, cycles, B):
    pos, neg = rec["text_emb"]
    te = pos if B == 1 else torch.cat([pos, neg], dim=0)
    engine.set_prompt(te.to(torch.bfloat16))
    outs, ms = [], []
    for c in range(cycles):
        nv, na = rec["noise"][c]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        a, _ = engine.infer_cycle(nv, na, rec["init_latent"] if c == 0 else None)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        engine.commit_chunk(*rec["commit"][c])
        torch.cuda.synchronize(); t2 = time.perf_counter()
        outs.append(a.clone())
        ms.append(((t1 - t0) * 1000, (t2 - t1) * 1000))
    return outs, ms


def compare(ref_list, out_list, used_mask):
    rows = []
    prev = None
    for c, (r, o) in enumerate(zip(ref_list, out_list)):
        r = r.float()[:, used_mask].reshape(-1)
        o = o.float()[:, used_mask].reshape(-1)
        d = (o - r).abs()
        mv = float((r - prev).abs().max()) if prev is not None else float("nan")
        cos = float(torch.nn.functional.cosine_similarity(r, o, dim=0))
        rows.append({"c": c, "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                     "p99_abs": float(d.quantile(0.99)), "cosine": cos, "movement": mv,
                     "ratio_max_over_movement": (float(d.max()) / mv if mv == mv and mv > 0 else None),
                     "ref_absmax": float(r.abs().max()), "identical": bool(torch.equal(r, o))})
        prev = r
    return rows


def _episode_obs(data, meta_e, e, cycles):
    """Decode one episode script: obs0 (init frame dict) + per-cycle keyframe dict lists."""
    obs0 = _make_obs(data[f"frame0_{e}"])
    jpegs = data[f"jpeg_{e}"]
    kfs, pos = [], 0
    for c in range(cycles):
        n = meta_e["kf_counts"][c]
        kfs.append([_make_obs(jpegs[pos + k]) for k in range(n)])
        pos += n
    return {"obs0": obs0, "kfs": kfs}


def run_parity(args, model, placement, engine):
    from flash_rt.frontends.torch.wan_va_thor import ROBOTWIN_USED_ACTION_CHANNELS
    data = np.load(args.parity, allow_pickle=True)
    meta = json.loads(Path(args.parity).with_suffix(".json").read_text())
    used = torch.zeros(30, dtype=torch.bool)
    used[ROBOTWIN_USED_ACTION_CHANNELS] = True
    eps = _parse_range(args.episodes, len(meta))
    calib = set(_parse_range(args.calibrate_episodes, len(meta))) if args.calibrate_episodes else set()
    out_dir = Path(args.parity_out); out_dir.mkdir(parents=True, exist_ok=True)
    report = {"point": engine.served_operating_point().served_point(),
              "declaration": engine.declaration(), "precision": engine.precision,
              "seed": args.deterministic_seed, "episodes": {}, "controls": {}, "ladder": {}}
    max_cycles = args.parity_cycles
    recs = {}
    # ── torch arm (records) ──
    for e in eps:
        m = meta[e]
        cycles = min(m["cycles"], max_cycles)
        t0 = time.perf_counter()
        ep = _episode_obs(data, m, e, cycles)
        recs[e] = record_torch_episode(model, placement, ep, data[f"actions_{e}"], m["prompt"], cycles)
        recs[e]["_ep"] = ep
        print(f"[parity] torch arm ep{e} {m['task']} {cycles} cycles in {time.perf_counter()-t0:.1f}s "
              f"digests {recs[e]['digest'][:3]}...", flush=True)
    # ── fp8 calibration on the declared subset (real activations, both regimes present) ──
    if engine.precision == "fp8" and calib and not args.act_scales:
        engine.begin_calibration()
        for e in sorted(calib):
            run_engine_episode(engine, recs[e], len(recs[e]["noise"]), engine.B)
        sc = engine.end_calibration()
        (out_dir / (f"act_scales_{engine.point.key()}" + ("_fp16-" + "-".join(engine.fp16_families) if engine.fp16_families else "") + ".json")).write_text(json.dumps(
            {"point": engine.point.key(), "calibration_episodes": sorted(calib), **sc}, indent=1))
        report["calibration"] = {"episodes": sorted(calib),
                                 "scale_min": min(min(r) for r in sc["act_scales"]),
                                 "scale_max": max(max(r) for r in sc["act_scales"])}
        print(f"[parity] fp8 calibrated on {sorted(calib)}: {report['calibration']}", flush=True)
    # ── engine arm ──
    for e in eps:
        rec = recs[e]; cycles = len(rec["noise"])
        outs, ms = run_engine_episode(engine, rec, cycles, engine.B)
        outs2, _ = run_engine_episode(engine, rec, cycles, engine.B)     # determinism replay
        rows = compare(rec["norm_actions"], outs, used)
        det = all(torch.equal(a, b) for a, b in zip(outs, outs2))
        report["episodes"][str(e)] = {
            "task": meta[e]["task"], "episode": meta[e]["episode"], "cycles": cycles,
            "held_out_of_calibration": e not in calib,
            "engine_replay_identical": det,
            "torch_digests": rec["digest"], "engine_digests": [_digest(a) for a in outs],
            "rows": rows, "engine_ms": ms,
            "max_abs_over_cycles": max(r["max_abs"] for r in rows),
            "median_ratio": float(np.nanmedian([r["ratio_max_over_movement"] or np.nan for r in rows])),
        }
        print(f"[parity] engine ep{e}: max|Δ| {report['episodes'][str(e)]['max_abs_over_cycles']:.4f} "
              f"cos min {min(r['cosine'] for r in rows):.5f} replay_identical={det}", flush=True)
    # ── ladder rung 2: the model's own ±1-ulp sensitivity (torch vs torch, episode eps[0]) ──
    if args.ladder:
        e = eps[0]; m = meta[e]; cycles = len(recs[e]["noise"])
        pert = record_torch_episode(model, placement, recs[e]["_ep"], data[f"actions_{e}"],
                                    m["prompt"], cycles,
                                    perturb=lambda t: (t.float() * (1.0 + 1.0 / 256)).to(t.dtype))
        rows = compare(recs[e]["norm_actions"], pert["norm_actions"], used)
        report["ladder"]["torch_vs_torch_noise_plus_1ulp"] = {"episode": e, "rows": rows,
            "max_abs_over_cycles": max(r["max_abs"] for r in rows)}
        # repeat identity of the torch arm itself (determinism of the reference)
        rep = record_torch_episode(model, placement, recs[e]["_ep"], data[f"actions_{e}"],
                                   m["prompt"], cycles)
        report["ladder"]["torch_repeat_identical"] = rep["digest"] == recs[e]["digest"]
        print(f"[parity] ladder: torch±1ulp max|Δ| {report['ladder']['torch_vs_torch_noise_plus_1ulp']['max_abs_over_cycles']:.4f}; "
              f"torch repeat identical={report['ladder']['torch_repeat_identical']}", flush=True)
    if args.ladder_all:
        # the model's own ±1-ulp sensitivity band on EVERY episode → per-cycle U for a like-for-like
        # comparison with the engine's per-cycle delta (the band varies 3x across cycles)
        per = {}
        for e in eps:
            m = meta[e]; cycles = len(recs[e]["noise"])
            pert = record_torch_episode(model, placement, recs[e]["_ep"], data[f"actions_{e}"],
                                        m["prompt"], cycles,
                                        perturb=lambda t: (t.float() * (1.0 + 1.0 / 256)).to(t.dtype))
            per[str(e)] = compare(recs[e]["norm_actions"], pert["norm_actions"], used)
            print(f"[parity] ladder ep{e}: U per cycle {[round(r['max_abs'], 4) for r in per[str(e)]]}", flush=True)
        report["ladder"]["per_episode_noise_plus_1ulp"] = per
    # ── broken controls that MUST fail (episode eps[0]) ──
    if args.controls:
        e = eps[0]; rec = recs[e]; cycles = min(len(rec["noise"]), 3)
        ctrl = {}
        engine.debug_force_t0 = True
        outs, _ = run_engine_episode(engine, rec, cycles, engine.B)
        engine.debug_force_t0 = False
        ctrl["wrong_t_tables(all_t0)"] = max(r["max_abs"] for r in compare(rec["norm_actions"][:cycles], outs, used))
        engine.debug_cross_len = 64
        outs, _ = run_engine_episode(engine, rec, cycles, engine.B)
        engine.debug_cross_len = None
        ctrl["dropped_text_pad_rows(64_of_512)"] = max(r["max_abs"] for r in compare(rec["norm_actions"][:cycles], outs, used))
        if engine.B == 2:
            pos, _ = rec["text_emb"]
            engine.set_prompt(torch.cat([pos, pos], 0).to(torch.bfloat16))
            outs = []
            for c in range(cycles):
                a, _ = engine.infer_cycle(*rec["noise"][c], rec["init_latent"] if c == 0 else None)
                outs.append(a.clone()); engine.commit_chunk(*rec["commit"][c])
            ctrl["negative_row_collapsed_to_positive"] = max(r["max_abs"] for r in compare(rec["norm_actions"][:cycles], outs, used))
        report["controls"] = ctrl
        print(f"[parity] controls: {ctrl}", flush=True)
    tag = f"{engine.point.key()}_{engine.precision}" + ("_fp16-" + "-".join(engine.fp16_families) if engine.fp16_families else "")
    (out_dir / f"parity_{tag}.json").write_text(json.dumps(report, indent=1))
    print("[parity] WROTE", out_dir / f"parity_{tag}.json", flush=True)


def _parse_range(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


# ══════════════════════════════════════════════════════════════════
# 4. main
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-name", default="robotwin")
    ap.add_argument("--port", type=int, default=29157)
    ap.add_argument("--save_root", default=None)
    ap.add_argument("--degrade-nfe", required=True, help="V,A — the declared schedule (2,4 or 2,2)")
    ap.add_argument("--guidance", default=None, help="e.g. video=positive_only")
    ap.add_argument("--deterministic-seed", type=int, default=None)
    ap.add_argument("--engine-precision", default="fp8", choices=["fp8", "fp16"])
    ap.add_argument("--act-scales", default=None, help="JSON from a calibration run (fp8)")
    ap.add_argument("--fp16-families", default="", help="comma list of GEMM families kept fp16 in the fp8 arm (ff2_w,o_w,...)")
    ap.add_argument("--no-action-terminal-elision", action="store_true",
                    help="A/B only: run BOTH the torch chain and the engine WITHOUT P010 (the stock 10/8 forwards)")
    ap.add_argument("--cuda-graph", action="store_true", help="lazy per-(kind,S,head,tail) graph cache")
    ap.add_argument("--num-sms", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--ledger-out", default=None)
    ap.add_argument("--parity", default=None, help="va_eval_obs.npz → in-process M2 harness")
    ap.add_argument("--parity-out", default=os.path.expanduser("~/thor_va_engine/parity"))
    ap.add_argument("--parity-cycles", type=int, default=8)
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--calibrate-episodes", default=None)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--ladder-all", action="store_true", help="±1ulp torch band on every episode")
    ap.add_argument("--torch-only", action="store_true",
                    help="serve the torch chain from this script (A/B of the wrapper itself)")
    args, extra = ap.parse_known_args()

    from flash_rt.models.wan_va.operating_point import WanVaOperatingPoint
    requested = WanVaOperatingPoint.from_worker_flags(args.degrade_nfe, args.guidance)
    print(f"[engine] requested operating point: {requested.served_point()}", flush=True)

    S, model, host, port, orig_serve = build_torch_chain(args, extra)
    served = WanVaOperatingPoint.from_job_config(model.job_config)
    if served != requested:
        raise SystemExit(f"the torch chain resolved {served.served_point()} but the flags requested "
                         f"{requested.served_point()} — refusing to build an engine for an ambiguous point")

    from flash_rt.frontends.torch.wan_va_thor import WanVaTorchFrontendThor
    ckpt = Path(model.job_config.wan22_pretrained_model_name_or_path) / "transformer"
    t0 = time.time()
    fams = tuple(x.strip() for x in args.fp16_families.split(",") if x.strip())
    engine = WanVaTorchFrontendThor(str(ckpt), point=served, precision=args.engine_precision,
                                    use_cuda_graph=args.cuda_graph, num_sms=args.num_sms,
                                    fp16_families=fams,
                                    action_terminal_elision=not args.no_action_terminal_elision)
    engine.profile = args.profile
    if args.act_scales:
        engine.load_act_scales(json.loads(Path(args.act_scales).read_text())["act_scales"])
        print(f"[engine] act scales loaded from {args.act_scales}", flush=True)
    decl = engine.declaration()
    print(f"[engine] built in {time.time()-t0:.1f}s: {json.dumps(decl)}", flush=True)
    placement = EnginePlacement(model, engine, args.deterministic_seed, args.ledger_out, args.profile)
    if args.ledger_out:
        with open(args.ledger_out, "a") as f:
            f.write(json.dumps({"start": time.time(), "declaration": decl, "args": vars(args)}) + "\n")

    if args.parity:
        run_parity(args, model, placement, engine)
        return 0
    if not args.torch_only:
        placement.enable()
        print(f"[engine] placement ENABLED — serving {served.served_point()} on port {port}", flush=True)
    else:
        print("[engine] --torch-only: serving the torch chain through this wrapper", flush=True)
    orig_serve(model, 0, host, port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
