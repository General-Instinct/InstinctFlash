"""T2-V2 repack: HF ckpt (+ calib scales) → engine-layout artifact dir.

Implements the repack table of
/home/ubuntu/iwm_distill/thor_t2v2/repack_calib_plan.md §1, mirroring
``frontends/torch/vla2_thor.py::_load_weights`` transform-for-transform
(single source of truth for key names/shapes is that loader; this
module re-states them so the repack runs WITHOUT a GPU frontend and on
EITHER box).

Artifact dir contents (torch.save, CPU tensors; fp8 = float8_e4m3fn):
    vit.pt      fp16 [K,N] weights + biases + LN vectors; patch-embed
                GEMM weight; pos_add fp16 constant (R1 table + conv
                bias fold) when the R1 table is available
    lm.pt       fp8 [K,N] qkv/o/gate/up/down (+RMS folds) + w_scales
                (180,) fp32 + q/k-norm fp16 + embed_tokens fp16
    expert.pt   fp8 [K,N] qkv(+bias fp16)/o/shared + w_scales (180,)
                + AdaRMS host fp32 params + final norm
    moe.pt      router fp32 (gate_w, e_bias) + routed experts fp8 in
                the NT ([out,in], native HF) orientation, BOTH scale
                modes materialized: per-expert (v0 loop) and shared
                per-(layer,matrix) quantized DIRECTLY from fp32 (v0.5
                batched; avoids double-rounding through the per-expert
                grid) + the R3 spread report
    heads.pt    fp16 heads (aout ×dt folded, action-time split) +
                host fp32 time-table halves + 16×2560 align constant
    act_scales.pt + calib provenance copied from --calib
    meta.json   shapes/conventions/sizes/provenance

Step R1 (pos-embed interpolation) is NOT re-implemented: use
``--r1-standalone`` to run the stock HF ``fast_pos_embed_interpolate``
on a minimally-constructed Qwen3VLVisionModel loaded with the ckpt's
pos_embed (cast bf16 first — deployed numerics), or pass an existing
table via --calib (m2_calibration.py dumps one from the fully-loaded
deployed model — the preferred source).

Usage (H100 dry-run == Thor local run; only paths change):
    python -m flash_rt.models.vla2.repack_v2 \
        --ckpt .../global_step_50000/hf_ckpt \
        --calib /home/ubuntu/iwm_distill/thor_t2v2/calib \
        --out   .../repack_v2_out [--device cuda]
    python -m flash_rt.models.vla2.repack_v2 --r1-standalone \
        --ckpt ... --out ...      # writes pos_embed_interp.pt only
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time

import torch

from flash_rt.frontends.torch.vla2_thor import (
    _ShardedCkpt, VIS_L, VIS_D, VIS_DEEPSTACK, VIS_S, LM_L, EXP_L, EXP_D,
    MOE_E, STEPS, NUM_TASK_TOKENS, PATCH_FLAT,
)

fp16 = torch.float16
fp8 = torch.float8_e4m3fn
P = "model.qwenvl_with_expert."


def _quantize_fp8(w32: torch.Tensor, scale: float | None = None):
    if scale is None:
        scale = max(w32.abs().max().item() / 448.0, 1e-12)
    return (w32 / scale).clamp(-448.0, 448.0).to(fp8), float(scale)


class Repacker:
    def __init__(self, ckpt_dir: pathlib.Path, device: str):
        self.sf = _ShardedCkpt(ckpt_dir)
        self.dev = device

    def g32(self, key):
        return self.sf.get(key).to(torch.float32)

    def b16(self, key):
        return self.sf.get(key).to(fp16)

    def kn16(self, key, fold=None):
        w = self.g32(key).to(self.dev)
        if fold is not None:
            w = w * fold.to(self.dev)[None, :]
        return w.t().contiguous().to(fp16).cpu()

    def kn8(self, key_or_t, fold=None):
        w = (self.g32(key_or_t) if isinstance(key_or_t, str)
             else key_or_t).to(self.dev)
        if fold is not None:
            w = w * fold.to(self.dev)[None, :]
        q, s = _quantize_fp8(w.t().contiguous())
        return q.cpu(), s

    # ── ViT (full fp16) ─────────────────────────────────────────
    def repack_vit(self, pos_table: torch.Tensor | None):
        vp = P + "qwenvl.model.visual."
        out = {k: [] for k in ("qkv_w", "qkv_b", "o_w", "o_b", "fc1_w",
                               "fc1_b", "fc2_w", "fc2_b", "n1_w", "n1_b",
                               "n2_w", "n2_b")}
        for l in range(VIS_L):
            bp = f"{vp}blocks.{l}."
            out["qkv_w"].append(self.kn16(bp + "attn.qkv.weight"))
            out["qkv_b"].append(self.b16(bp + "attn.qkv.bias"))
            out["o_w"].append(self.kn16(bp + "attn.proj.weight"))
            out["o_b"].append(self.b16(bp + "attn.proj.bias"))
            out["fc1_w"].append(self.kn16(bp + "mlp.linear_fc1.weight"))
            out["fc1_b"].append(self.b16(bp + "mlp.linear_fc1.bias"))
            out["fc2_w"].append(self.kn16(bp + "mlp.linear_fc2.weight"))
            out["fc2_b"].append(self.b16(bp + "mlp.linear_fc2.bias"))
            out["n1_w"].append(self.b16(bp + "norm1.weight"))
            out["n1_b"].append(self.b16(bp + "norm1.bias"))
            out["n2_w"].append(self.b16(bp + "norm2.weight"))
            out["n2_b"].append(self.b16(bp + "norm2.bias"))
        ds = {k: [] for k in ("n_w", "n_b", "fc1_w", "fc1_b",
                              "fc2_w", "fc2_b")}
        for i in range(len(VIS_DEEPSTACK)):
            dp = f"{vp}deepstack_merger_list.{i}."
            ds["n_w"].append(self.b16(dp + "norm.weight"))
            ds["n_b"].append(self.b16(dp + "norm.bias"))
            ds["fc1_w"].append(self.kn16(dp + "linear_fc1.weight"))
            ds["fc1_b"].append(self.b16(dp + "linear_fc1.bias"))
            ds["fc2_w"].append(self.kn16(dp + "linear_fc2.weight"))
            ds["fc2_b"].append(self.b16(dp + "linear_fc2.bias"))
        out["deepstack"] = ds
        out["merger"] = {
            "n_w": self.b16(vp + "merger.norm.weight"),
            "n_b": self.b16(vp + "merger.norm.bias"),
            "fc1_w": self.kn16(vp + "merger.linear_fc1.weight"),
            "fc1_b": self.b16(vp + "merger.linear_fc1.bias"),
            "fc2_w": self.kn16(vp + "merger.linear_fc2.weight"),
            "fc2_b": self.b16(vp + "merger.linear_fc2.bias"),
        }
        pe = self.g32(vp + "patch_embed.proj.weight").reshape(
            VIS_D, PATCH_FLAT)
        out["pe_w"] = pe.t().contiguous().to(fp16)
        pe_b32 = self.g32(vp + "patch_embed.proj.bias")
        out["pe_b32"] = pe_b32
        if pos_table is not None:
            assert tuple(pos_table.shape) == (VIS_S, VIS_D), pos_table.shape
            out["pos_add"] = (pos_table.float()
                              + pe_b32[None, :]).to(fp16)
        return out

    # ── LM (fp8 W, RMS folds) ───────────────────────────────────
    def repack_lm(self):
        lp = P + "qwenvl.model.language_model.layers."
        out = {k: [] for k in ("qkv_w", "o_w", "gate_w", "up_w",
                               "down_w", "qnorm_w", "knorm_w")}
        scales = []
        for l in range(LM_L):
            bp = f"{lp}{l}."
            n_in = self.g32(bp + "input_layernorm.weight")
            n_post = self.g32(bp + "post_attention_layernorm.weight")
            qkv = torch.cat([self.g32(bp + "self_attn.q_proj.weight"),
                             self.g32(bp + "self_attn.k_proj.weight"),
                             self.g32(bp + "self_attn.v_proj.weight")],
                            dim=0)
            q, s = self.kn8(qkv, fold=n_in)
            out["qkv_w"].append(q); scales.append(s)
            q, s = self.kn8(bp + "self_attn.o_proj.weight")
            out["o_w"].append(q); scales.append(s)
            q, s = self.kn8(bp + "mlp.gate_proj.weight", fold=n_post)
            out["gate_w"].append(q); scales.append(s)
            q, s = self.kn8(bp + "mlp.up_proj.weight", fold=n_post)
            out["up_w"].append(q); scales.append(s)
            q, s = self.kn8(bp + "mlp.down_proj.weight")
            out["down_w"].append(q); scales.append(s)
            out["qnorm_w"].append(self.b16(bp + "self_attn.q_norm.weight"))
            out["knorm_w"].append(self.b16(bp + "self_attn.k_norm.weight"))
        out["w_scales"] = torch.tensor(scales, dtype=torch.float32)
        out["embed_tokens"] = self.sf.get(
            P + "qwenvl.model.language_model.embed_tokens.weight").to(fp16)
        return out

    # ── expert attention + shared expert + AdaRMS hosts ────────
    def repack_expert(self):
        ep = P + "qwen_expert.model.layers."
        out = {k: [] for k in ("qkv_w", "qkv_b", "o_w", "sh_gate_w",
                               "sh_up_w", "sh_down_w")}
        scales, ada = [], []
        for l in range(EXP_L):
            bp = f"{ep}{l}."
            qkv = torch.cat([self.g32(bp + "self_attn.q_proj.weight"),
                             self.g32(bp + "self_attn.k_proj.weight"),
                             self.g32(bp + "self_attn.v_proj.weight")],
                            dim=0)
            q, s = self.kn8(qkv)
            out["qkv_w"].append(q); scales.append(s)
            out["qkv_b"].append(torch.cat([
                self.sf.get(bp + "self_attn.q_proj.bias"),
                self.sf.get(bp + "self_attn.k_proj.bias"),
                self.sf.get(bp + "self_attn.v_proj.bias")]).to(fp16))
            q, s = self.kn8(bp + "self_attn.o_proj.weight")
            out["o_w"].append(q); scales.append(s)
            q, s = self.kn8(bp + "mlp.shared_expert.gate_proj.weight")
            out["sh_gate_w"].append(q); scales.append(s)
            q, s = self.kn8(bp + "mlp.shared_expert.up_proj.weight")
            out["sh_up_w"].append(q); scales.append(s)
            q, s = self.kn8(bp + "mlp.shared_expert.down_proj.weight")
            out["sh_down_w"].append(q); scales.append(s)
            ada.append({k: self.g32(bp + n) for k, n in (
                ("in_w", "input_layernorm.weight"),
                ("in_gw", "input_layernorm.gamma.weight"),
                ("in_gb", "input_layernorm.gamma.bias"),
                ("in_bw", "input_layernorm.beta.weight"),
                ("in_bb", "input_layernorm.beta.bias"),
                ("po_w", "post_attention_layernorm.weight"),
                ("po_gw", "post_attention_layernorm.gamma.weight"),
                ("po_gb", "post_attention_layernorm.gamma.bias"),
                ("po_bw", "post_attention_layernorm.beta.weight"),
                ("po_bb", "post_attention_layernorm.beta.bias"))})
        out["w_scales"] = torch.tensor(scales, dtype=torch.float32)
        out["ada"] = ada
        out["final_norm_w"] = self.b16(P + "qwen_expert.model.norm.weight")
        return out

    # ── routed MoE (NT orientation, both scale modes from fp32) ─
    def repack_moe(self):
        ep = P + "qwen_expert.model.layers."
        out = {"num_layers": EXP_L, "orientation":
               "NT [out,in] per expert (fp8_gemm_batched_descale_nt_fp16"
               " / Vla2MoeEngine layout)",
               "gate_w": [], "e_bias": [],
               "gateup_fp8_per_expert": [], "down_fp8_per_expert": [],
               "gateup_fp8_shared": [], "down_fp8_shared": []}
        gu_pe, dn_pe, gu_sh, dn_sh = [], [], [], []
        spread = {}
        for l in range(EXP_L):
            bp = f"{ep}{l}.mlp."
            out["gate_w"].append(self.g32(bp + "gate.weight"))
            out["e_bias"].append(self.g32(bp + "e_score_correction_bias"))
            gu = torch.cat([self.g32(bp + "experts.gate_proj"),
                            self.g32(bp + "experts.up_proj")],
                           dim=1).to(self.dev)          # (E, 1024, 768)
            dn = self.g32(bp + "experts.down_proj").to(self.dev)
            # per-expert scales (v0 loop)
            q_pe = torch.empty_like(gu, dtype=fp8)
            d_pe = torch.empty_like(dn, dtype=fp8)
            gss, dss = [], []
            for e in range(MOE_E):
                q, s = _quantize_fp8(gu[e]); q_pe[e] = q; gss.append(s)
                q, s = _quantize_fp8(dn[e]); d_pe[e] = q; dss.append(s)
            # shared per-(layer,matrix) scale (v0.5 batched) — straight
            # from fp32, no double rounding
            q_sh, gs = _quantize_fp8(gu)
            d_sh, ds = _quantize_fp8(dn)
            out["gateup_fp8_per_expert"].append(q_pe.cpu())
            out["down_fp8_per_expert"].append(d_pe.cpu())
            out["gateup_fp8_shared"].append(q_sh.cpu())
            out["down_fp8_shared"].append(d_sh.cpu())
            gu_pe.append(gss); dn_pe.append(dss)
            gu_sh.append(gs); dn_sh.append(ds)
            spread[f"L{l:02d}.gateup"] = max(gss) / min(gss)
            spread[f"L{l:02d}.down"] = max(dss) / min(dss)
        out["gateup_scale_per_expert"] = torch.tensor(gu_pe)
        out["down_scale_per_expert"] = torch.tensor(dn_pe)
        out["gateup_scale_shared"] = torch.tensor(gu_sh)
        out["down_scale_shared"] = torch.tensor(dn_sh)
        out["r3_scale_spread"] = spread
        return out

    # ── heads + align constants ─────────────────────────────────
    def repack_heads(self):
        def t16(key):
            return self.g32(key).t().contiguous().to(fp16)

        atm_in_w = self.g32("model.action_time_mlp_in.weight")
        dt = -1.0 / STEPS
        out = {
            "state_w": t16("model.state_proj.weight"),
            "state_b": self.b16("model.state_proj.bias"),
            "ain_w": t16("model.action_in_proj.weight"),
            "ain_b": self.b16("model.action_in_proj.bias"),
            "atm_a_w": atm_in_w[:, :EXP_D].t().contiguous().to(fp16),
            "atm_t_w32": atm_in_w[:, EXP_D:],           # host fp32
            "atm_in_b32": self.g32("model.action_time_mlp_in.bias"),
            "atm_out_w": t16("model.action_time_mlp_out.weight"),
            "atm_out_b": self.b16("model.action_time_mlp_out.bias"),
            "aout_w_dt": (self.g32("model.action_out_proj.weight") * dt
                          ).t().contiguous().to(fp16),
            "aout_b_dt": (self.g32("model.action_out_proj.bias") * dt
                          ).to(fp16),
        }

        def _group(t):
            return t.view(NUM_TASK_TOKENS, -1, t.shape[-1]).mean(dim=1)

        cur = torch.cat([_group(self.g32("model.depth_align_embs")),
                         _group(self.g32("model.current_video_align_embs"))],
                        dim=-1)
        cur = torch.nn.functional.linear(
            cur, self.g32("model.current_shared_task_proj.weight"),
            self.g32("model.current_shared_task_proj.bias"))
        fut = torch.cat([_group(self.g32("model.future_depth_align_embs")),
                         _group(self.g32("model.future_video_align_embs"))],
                        dim=-1)
        fut = torch.nn.functional.linear(
            fut, self.g32("model.future_shared_task_proj.weight"),
            self.g32("model.future_shared_task_proj.bias"))
        out["align_emb"] = torch.cat([cur, fut], dim=0).to(fp16)
        return out


def r1_standalone(ckpt_dir: pathlib.Path, out_dir: pathlib.Path):
    """Run the STOCK HF fast_pos_embed_interpolate once (bf16 module —
    deployed numerics), store the (768, 1024) fp32 table. Needs
    transformers; does NOT need the lingbotvla repo or a GPU."""
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel)
    cfg = AutoConfig.from_pretrained(
        os.environ.get("QWEN3VL_PATH", "Qwen/Qwen3-VL-4B-Instruct"))
    vis = Qwen3VLVisionModel(cfg.vision_config)
    sf = _ShardedCkpt(ckpt_dir)
    pe = sf.get(P + "qwenvl.model.visual.pos_embed.weight")
    with torch.no_grad():
        vis.pos_embed.weight.copy_(pe)
    vis = vis.to(torch.bfloat16)          # match the deployed cast
    with torch.no_grad():
        pos = vis.fast_pos_embed_interpolate(
            torch.tensor([[1, 16, 16]] * 3))
    table = pos.reshape(-1, pos.shape[-1]).float()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(table, out_dir / "pos_embed_interp.pt")
    print(f"[R1] wrote {out_dir/'pos_embed_interp.pt'} "
          f"shape {tuple(table.shape)}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default=None,
                    help="dir with scales.pt/provenance.json/"
                         "pos_embed_interp.pt (m2_calibration.py output)")
    ap.add_argument("--device", default="cuda" if
                    torch.cuda.is_available() else "cpu")
    ap.add_argument("--r1-standalone", action="store_true",
                    help="only produce pos_embed_interp.pt (stock HF "
                         "interpolation, bf16 deployed numerics)")
    args = ap.parse_args()
    ckpt = pathlib.Path(args.ckpt)
    out_dir = pathlib.Path(args.out)

    if args.r1_standalone:
        r1_standalone(ckpt, out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pos_table = None
    calib_note = "NONE — pos_add not built (run R1 / pass --calib)"
    if args.calib:
        cal = pathlib.Path(args.calib)
        pt = cal / "pos_embed_interp.pt"
        if pt.exists():
            pos_table = torch.load(pt, map_location="cpu",
                                   weights_only=True)
            calib_note = str(cal)
        for f in ("scales.pt", "scales.json", "provenance.json"):
            if (cal / f).exists():
                shutil.copy2(cal / f, out_dir /
                             ("act_scales.pt" if f == "scales.pt" else
                              "calib_" + f))

    t0 = time.time()
    rp = Repacker(ckpt, args.device)
    sizes = {}
    for name, fn, kwargs in (
            ("vit", rp.repack_vit, {"pos_table": pos_table}),
            ("lm", rp.repack_lm, {}),
            ("expert", rp.repack_expert, {}),
            ("moe", rp.repack_moe, {}),
            ("heads", rp.repack_heads, {})):
        t1 = time.time()
        blob = fn(**kwargs)
        torch.save(blob, out_dir / f"{name}.pt")
        sizes[name] = (out_dir / f"{name}.pt").stat().st_size
        print(f"[{name}] repacked+saved in {time.time()-t1:.1f}s "
              f"({sizes[name]/1e9:.2f} GB)", flush=True)

    meta = {
        "date": "2026-08-25",
        "ckpt": str(ckpt),
        "device_used": args.device,
        "calib": calib_note,
        "quantization": "per-tensor symmetric E4M3 amax/448 "
                        "(pi05/vla4b recipe); LM qkv fold=input_ln, "
                        "gate/up fold=post_attention_ln (fp32 folds)",
        "layouts": {
            "backbone GEMM weights": "[K, N] (= hf.T.contiguous) fp8/fp16 "
                                     "— fp8_gemm_descale_fp16 convention",
            "routed MoE experts": "NT [out, in] per expert — "
                                  "fp8_gemm_batched_descale_nt_fp16 / "
                                  "Vla2MoeEngine (NN is cuBLASLt-"
                                  "unsupported on sm_90/cu12.8)",
            "moe scale modes": "BOTH materialized: per_expert (v0 loop) "
                               "and shared per-(layer,matrix) (v0.5 "
                               "batched), each quantized straight from "
                               "fp32",
        },
        "w_scale_order": {"lm": "l*5 + {qkv,o,gate,up,down}",
                          "expert": "l*5 + {qkv,o,sh_gate,sh_up,sh_down}"},
        "sizes_bytes": sizes,
        "total_gb": round(sum(sizes.values()) / 1e9, 2),
        "elapsed_s": round(time.time() - t0, 1),
        "pos_add_built": pos_table is not None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {meta['total_gb']} GB in {out_dir} "
          f"({meta['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
