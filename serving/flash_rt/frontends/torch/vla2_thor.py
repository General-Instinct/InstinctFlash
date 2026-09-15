"""FlashRT — Vla2TorchFrontendThor: LingBot-VLA-V2-6B on Thor SM110 (STAGE-1 SCAFFOLD).

Structure cloned from ``vla4b_thor.py`` (the T2-V1 frontend), with V2
shapes filled in from the real checkpoint
(``robbyant/lingbot-vla-v2-6b-robotwin`` @ global_step_50000, shapes
verified from the safetensors headers) and the real config
(``lingbotvla_cli.yaml`` + Qwen/Qwen3-VL-4B-Instruct config.json).

STATUS: Stage-1 scaffold. Implemented: weight loading + fp8/fp16
repack, align-token constant precompute, rope/step tables, buffer
allocation, KV slab + the Stage-A KV-handoff entry point
(``load_prefix_kv_from_torch``). NOT implemented: the routed-MoE
engine path (Stage-2 kernels; ``models/vla2/pipeline_thor.py``
``routed_moe_stub`` documents the contract; torch parity baseline in
``models/vla2/moe_ref.py``) and the ViT positional-embedding table
(must be produced ONCE by the repack harness running the stock
``fast_pos_embed_interpolate`` — correct-by-construction rather than a
re-implementation; see repack_calib_plan.md).

Key differences from vla4b (all source-verified, see
/home/ubuntu/iwm_distill/thor_t2v2/mapping_memo.md):
    * Qwen3-VL ViT: 24L/1024/16h×hd64, FULL attention per image (no
      windows/permutes), LayerNorm+bias, gelu-tanh MLP (no gate),
      deepstack taps [5,11,17] injected into LM layers 0-2.
    * LM: 36L/2560 GQA 32q/8kv×hd128, per-head q/k RMS norms, NO attn
      bias, interleaved 3D M-RoPE theta 5e6, CAUSAL prefill
      (vlm_causal=true).
    * Prefix carries 3×[start|64|end] image blocks + language (Qwen3
      chat template) + 16 trained align-query constants.
    * Expert: joint geometry 32q/8kv×hd128 (kv_row 1024 — 4× V1),
      qkv bias, AdaRMS tables identical to V1, token-MoE MLP
      (32e/top4/512 + shared 704 ungated) on all 36 layers.
    * chunk 50, state/action dim 55 (not 75), 10 Euler steps.
"""

import ctypes
import json
import logging
import math
import os
import pathlib
import time
from typing import Optional, Union

import numpy as np
import torch

try:  # import-clean on hosts without the built extension
    import flash_rt.flash_rt_kernels as fvk
except Exception:  # pragma: no cover - exercised on CPU-only hosts
    fvk = None

from flash_rt.models.vla2.pipeline_thor import (
    Vla2Fa2Attn,
    vit_forward,
    lm_prefill_forward,
    expert_forward,
    routed_moe_stub,
)
from flash_rt.models.vla2.rope_table import (
    build_vision_layout,
    build_joint_rope_tables,
)
from flash_rt.models.vla2.step_tables import build_step_tables

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8 = torch.float8_e4m3fn

# ── architecture constants (locked by ckpt shapes + configs) ──
VIS_L, VIS_D, VIS_H = 24, 1024, 4096
VIS_NH, VIS_HD = 16, 64
VIS_S = 768                  # 3 images × 256 patch tokens (256px grid)
VIS_SM = 192                 # merged tokens (3 × 64)
VIS_DEEPSTACK = (5, 11, 17)
LM_L, LM_D, LM_H = 36, 2560, 9728
LM_NH, LM_NKV, LM_HD = 32, 8, 128
LM_DQ, LM_DKV = 4096, 1024
LM_QKV_OUT = LM_DQ + 2 * LM_DKV          # 6144
EXP_L, EXP_D = 36, 768
MOE_E, MOE_TOPK, MOE_H, SHARED_H = 32, 4, 512, 704
ROUTED_SCALING = 4.0
CHUNK, ADIM, SDIM = 50, 55, 55
STEPS = 10
SUF = CHUNK + 1
KV_ROW = LM_DKV                          # 1024 (8 kv heads × 128)
PATCH_FLAT = 1536                        # 3 × 2 × 16 × 16
IMG_BLOCK = 66                           # start + 64 + end
NUM_TASK_TOKENS = 8                      # per align segment
ALIGN_TOKENS = 2 * NUM_TASK_TOKENS       # current-task 8 + future-task 8

# ══════════════════════════════════════════════════════════════════
# Stage-A KV-handoff contract (deliverable-B reference; keep in sync
# with mapping_memo.md §B)
# ══════════════════════════════════════════════════════════════════
KV_HANDOFF_CONTRACT = """
Engine prefix-KV slab (what pi05/vla4b-skeleton pipelines consume):
  K, V: torch fp16, shape [L=36, total_keys, kv_row=1024], contiguous;
        total_keys = Se + 51; layer stride = total_keys*1024 elements.
  Rows [0, Se)   : prefix — written once per prompt/frame.
  Rows [Se, Se+51): suffix — REWRITTEN by the engine every denoise step.
  K rows are POST q/k-norm, POST-M-RoPE (that is what the stock model
  caches: modeling_lingbot_vla_v2.py applies apply_mrope at :354 and
  handle_kv_cache at :355-362). V rows are the raw v_proj outputs.
  Head layout: (8 kv heads × 128) flattened row-major per token.

From the stock model (Stage-A source):
  past_key_values[l]["key_states"/"value_states"]: fp32 (the joint
  forward casts q/k/v to .float() at :347-349), shape (1, Se_pad, 8,
  128), where Se_pad includes right-padded language rows. Handoff =
  gather the pad-mask-valid rows (pads are fully masked in every
  attention there, and Qwen3-VL get_rope_index is mask-aware, so the
  dense positions are identical), reshape (Se, 1024), cast fp16, copy
  into rows [0, Se) of layer l.
  Precision deltas of the fp32→fp16 cast are gated against the model's
  intrinsic run-to-run envelope (1.4-2.9e-2, fused-MoE atomics).
Also handed off per prompt: Se, and the suffix M-RoPE tables (built
host-side from max(valid prefix position)+1 — see rope_table.py).
"""


def _quantize_fp8(w32: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric E4M3: amax/448 (pi05/vla4b recipe)."""
    amax = w32.abs().max().item()
    scale = max(amax / 448.0, 1e-12)
    w_fp8 = (w32 / scale).clamp(-448.0, 448.0).to(fp8)
    return w_fp8, float(scale)


class _ShardedCkpt:
    """Multi-shard safetensors reader (V2 ships 6 shards + index)."""

    def __init__(self, ckpt_dir: pathlib.Path):
        from safetensors import safe_open
        idx = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())
        self._key2file = idx["weight_map"]
        self._files = {
            fn: safe_open(str(ckpt_dir / fn), framework="pt", device="cpu")
            for fn in sorted(set(self._key2file.values()))
        }

    def get(self, key: str) -> torch.Tensor:
        return self._files[self._key2file[key]].get_tensor(key)


class Vla2TorchFrontendThor:
    """LingBot-VLA-V2-6B inference using only flash_rt kernels + FA2.

    Stage-1 scaffold: constructible on a GPU host (loads + repacks
    weights, builds tables/buffers); the denoise loop raises until the
    Stage-2 MoE kernels land (inject ``routed_moe_fn`` to override).
    """

    def __init__(self, checkpoint_dir: Union[str, pathlib.Path],
                 num_views: int = 3, use_cuda_graph: bool = True,
                 fa2_lib=None, routed_moe_fn=None,
                 pos_embed_table: Optional[torch.Tensor] = None):
        # Stage-A (M1, H100) runs on hosts WITHOUT the flash_rt_kernels
        # extension: only the torch-side paths are needed there (weight
        # load/repack, rope/style tables, KV slab + the
        # ``load_prefix_kv_from_torch`` handoff).  Engine compute
        # (_run_vis/_run_lm_expert/capture) still requires fvk.
        self.torch_ref_only = fvk is None
        if self.torch_ref_only and use_cuda_graph:
            raise RuntimeError(
                "flash_rt_kernels extension not available — only the "
                "Stage-A torch-ref path can run on this host; construct "
                "with use_cuda_graph=False")
        assert num_views == 3, "V2 RoboTwin deployment is 3-camera"
        self.num_views = num_views
        self.use_cuda_graph = bool(use_cuda_graph)
        self.latency_records = []
        self.graph_captured = False
        self._real_data_calibrated = False
        self._prompt_layout = None
        self._prompt_buffers_reused = False
        self.prompt_updates = 0
        self.prompt_buffer_reuses = 0
        self.prompt_graph_captures = 0
        self._routed_moe_fn = routed_moe_fn or routed_moe_stub

        self._ctx = fvk.FvkContext() if fvk is not None else None
        if fa2_lib is None and fvk is not None:
            from flash_rt import flash_rt_fa2 as fa2_lib
        self._fa2 = fa2_lib
        if self.torch_ref_only:
            logger.warning("flash_rt_kernels unavailable — Stage-A "
                           "torch-ref mode (repack/tables/KV handoff only, "
                           "no engine compute)")

        ckpt = pathlib.Path(checkpoint_dir)
        if not (ckpt / "model.safetensors.index.json").exists():
            raise FileNotFoundError(ckpt / "model.safetensors.index.json")
        self.lm_prefill_precision = "fp8"
        self._lm16 = None
        self._load_weights(ckpt)
        self._set_pos_embed_table(pos_embed_table)
        self._alloc_vit_buffers()
        self._build_vision_tables()
        logger.info("Vla2TorchFrontendThor initialised (scaffold)")

    # ────────────────────────────────────────────────────────────
    # Artifact constructor (repack_v2_out — no 25 GB ckpt re-read)
    # ────────────────────────────────────────────────────────────

    @classmethod
    def from_artifacts(cls, art_dir: Union[str, pathlib.Path],
                       num_views: int = 3, fa2_lib=None,
                       routed_moe_fn=None, use_cuda_graph: bool = False,
                       load_vit: bool = True, load_lm: bool = True):
        """Build the frontend from a ``repack_v2`` artifact dir.

        Loads ViT (fp16 + pos_add), LM (fp8 + folds + embed), expert
        attention/shared (fp8 + AdaRMS hosts) and heads. The routed
        MoE stays behind ``routed_moe_fn`` (``Vla2MoeEngine``); the
        fp16 LM-prefill parity arm is loaded separately via
        ``load_lm_fp16_stack`` (weights come from the fp32 ckpt).
        """
        from flash_rt.models.vla2.pipeline_thor import routed_moe_stub as _stub
        art = pathlib.Path(art_dir)
        fr = cls.__new__(cls)
        fr.torch_ref_only = fvk is None
        fr.num_views = num_views
        fr.use_cuda_graph = bool(use_cuda_graph)
        fr.latency_records = []
        fr.graph_captured = False
        fr._real_data_calibrated = False
        fr._prompt_layout = None
        fr._prompt_buffers_reused = False
        fr.prompt_updates = 0
        fr.prompt_buffer_reuses = 0
        fr.prompt_graph_captures = 0
        fr._routed_moe_fn = routed_moe_fn or _stub
        fr._ctx = fvk.FvkContext() if fvk is not None else None
        if fa2_lib is None and fvk is not None:
            try:
                from flash_rt import flash_rt_fa2 as fa2_lib
            except ImportError:
                fa2_lib = None  # torch-attn adapter must be installed
        fr._fa2 = fa2_lib
        fr.lm_prefill_precision = "fp8"
        fr._lm16 = None
        t0 = time.time()

        if load_vit:
            vit = torch.load(art / "vit.pt", map_location="cpu",
                             weights_only=False)
            fr._v = {k: [t.cuda() for t in vit[k]] for k in (
                "qkv_w", "qkv_b", "o_w", "o_b", "fc1_w", "fc1_b",
                "fc2_w", "fc2_b", "n1_w", "n1_b", "n2_w", "n2_b")}
            ds = vit["deepstack"]
            fr._ds = {k: [t.cuda() for t in ds[k]] for k in (
                "n_w", "n_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b")}
            mg = vit["merger"]
            fr._m_n_w = mg["n_w"].cuda(); fr._m_n_b = mg["n_b"].cuda()
            fr._m0_w = mg["fc1_w"].cuda(); fr._m0_b = mg["fc1_b"].cuda()
            fr._m2_w = mg["fc2_w"].cuda(); fr._m2_b = mg["fc2_b"].cuda()
            fr._pe_w = vit["pe_w"].cuda()
            fr._pe_b32 = vit["pe_b32"]
            fr._pos_embed_ckpt = None
            assert "pos_add" in vit, "vit.pt lacks pos_add (repack w/o R1)"
            fr._pos_add = vit["pos_add"].cuda()
            del vit
        else:
            fr._v = fr._ds = None
            fr._pe_w = fr._pe_b32 = fr._pos_embed_ckpt = None
            fr._pos_add = None

        lm = torch.load(art / "lm.pt", map_location="cpu",
                        weights_only=False)
        fr._embed_cpu = lm["embed_tokens"]
        if load_lm:
            fr._l = {k: [t.cuda() for t in lm[k]] for k in (
                "qkv_w", "o_w", "gate_w", "up_w", "down_w",
                "qnorm_w", "knorm_w")}
            fr._lm_w_scales = lm["w_scales"].float().cuda()
        else:
            fr._l = None
            fr._lm_w_scales = None
        del lm

        exp = torch.load(art / "expert.pt", map_location="cpu",
                         weights_only=False)
        fr._e = {k: [t.cuda() for t in exp[k]] for k in (
            "qkv_w", "qkv_b", "o_w", "sh_gate_w", "sh_up_w", "sh_down_w")}
        fr._exp_w_scales = exp["w_scales"].float().cuda()
        fr._ada = exp["ada"]
        fr._final_norm_w = exp["final_norm_w"].cuda()
        del exp

        heads = torch.load(art / "heads.pt", map_location="cpu",
                           weights_only=False)
        for k in ("ain_w", "ain_b", "atm_a_w", "atm_out_w", "atm_out_b",
                  "state_w", "state_b", "aout_w_dt", "aout_b_dt",
                  "align_emb"):
            setattr(fr, "_" + k, heads[k].cuda())
        fr._atm_t_w32 = heads["atm_t_w32"]
        fr._atm_in_b32 = heads["atm_in_b32"]
        fr._moe = None

        fr._alloc_vit_buffers()
        fr._build_vision_tables()
        logger.info("Vla2TorchFrontendThor from_artifacts in %.1fs",
                    time.time() - t0)
        return fr

    def load_lm_fp16_stack(self, ckpt_dir: Union[str, pathlib.Path]):
        """fp16 [K,N] UNFOLDED LM weights + explicit norm vectors from
        the fp32 ckpt — the fp16 LM-prefill parity arm (~7.3 GB)."""
        t0 = time.time()
        sf = _ShardedCkpt(pathlib.Path(ckpt_dir))
        lp = "model.qwenvl_with_expert.qwenvl.model.language_model.layers."
        W = {k: [] for k in ("qkv_w16", "o_w16", "gate_w16", "up_w16",
                             "down_w16", "n_in_w", "n_post_w")}
        for l in range(LM_L):
            bp = f"{lp}{l}."
            qkv = torch.cat([sf.get(bp + "self_attn.q_proj.weight"),
                             sf.get(bp + "self_attn.k_proj.weight"),
                             sf.get(bp + "self_attn.v_proj.weight")],
                            dim=0).to(torch.float32)
            W["qkv_w16"].append(qkv.t().contiguous().to(fp16).cuda())
            for key, name in (("o_w16", "self_attn.o_proj.weight"),
                              ("gate_w16", "mlp.gate_proj.weight"),
                              ("up_w16", "mlp.up_proj.weight"),
                              ("down_w16", "mlp.down_proj.weight")):
                w = sf.get(bp + name).to(torch.float32)
                W[key].append(w.t().contiguous().to(fp16).cuda())
            W["n_in_w"].append(
                sf.get(bp + "input_layernorm.weight").to(fp16).cuda())
            W["n_post_w"].append(
                sf.get(bp + "post_attention_layernorm.weight")
                .to(fp16).cuda())
        self._lm16 = W
        logger.info("fp16 LM stack loaded in %.1fs", time.time() - t0)

    def set_vision_rope_tables(self, cos: torch.Tensor, sin: torch.Tensor):
        """Override the builder tables with stock-dumped constants
        (the stock ViT computes cos/sin fully in bf16 — dumping the
        deployed tables is the R1-pos-table precedent)."""
        assert cos.shape == (VIS_S, VIS_HD), cos.shape
        self._vis_cos = cos.to(fp16).cuda().contiguous()
        self._vis_sin = sin.to(fp16).cuda().contiguous()

    # ────────────────────────────────────────────────────────────
    # Weight loading (fp32 ckpt → fp8/fp16 device tensors)
    # ────────────────────────────────────────────────────────────

    def _load_weights(self, ckpt_dir: pathlib.Path):
        t0 = time.time()
        sf = _ShardedCkpt(ckpt_dir)
        P = "model.qwenvl_with_expert."

        def g32(key):
            return sf.get(key).to(torch.float32)

        def b16(key):
            return sf.get(key).to(fp16).cuda()

        def to_dev_fp16_kn(w32_nk, fold=None):
            w = w32_nk.cuda()
            if fold is not None:
                w = w * fold.cuda()[None, :]
            return w.t().contiguous().to(fp16)

        def to_dev_fp8_kn(w32_nk, fold=None):
            w = w32_nk.cuda()
            if fold is not None:
                w = w * fold.cuda()[None, :]
            return _quantize_fp8(w.t().contiguous())

        w_scales_lm, w_scales_exp = [], []

        # ── ViT (FULL FP16 — T2-V1 recipe carried; LN kept explicit,
        # no norm folding in the scaffold) ──
        vp = P + "qwenvl.model.visual."
        self._v = {k: [] for k in (
            "qkv_w", "qkv_b", "o_w", "o_b", "fc1_w", "fc1_b", "fc2_w",
            "fc2_b", "n1_w", "n1_b", "n2_w", "n2_b")}
        for l in range(VIS_L):
            bp = f"{vp}blocks.{l}."
            self._v["qkv_w"].append(to_dev_fp16_kn(g32(bp + "attn.qkv.weight")))
            self._v["qkv_b"].append(b16(bp + "attn.qkv.bias"))
            self._v["o_w"].append(to_dev_fp16_kn(g32(bp + "attn.proj.weight")))
            self._v["o_b"].append(b16(bp + "attn.proj.bias"))
            self._v["fc1_w"].append(to_dev_fp16_kn(g32(bp + "mlp.linear_fc1.weight")))
            self._v["fc1_b"].append(b16(bp + "mlp.linear_fc1.bias"))
            self._v["fc2_w"].append(to_dev_fp16_kn(g32(bp + "mlp.linear_fc2.weight")))
            self._v["fc2_b"].append(b16(bp + "mlp.linear_fc2.bias"))
            self._v["n1_w"].append(b16(bp + "norm1.weight"))
            self._v["n1_b"].append(b16(bp + "norm1.bias"))
            self._v["n2_w"].append(b16(bp + "norm2.weight"))
            self._v["n2_b"].append(b16(bp + "norm2.bias"))
        # deepstack mergers (3) + final merger
        self._ds = {k: [] for k in ("n_w", "n_b", "fc1_w", "fc1_b",
                                    "fc2_w", "fc2_b")}
        for i in range(len(VIS_DEEPSTACK)):
            dp = f"{vp}deepstack_merger_list.{i}."
            self._ds["n_w"].append(b16(dp + "norm.weight"))
            self._ds["n_b"].append(b16(dp + "norm.bias"))
            self._ds["fc1_w"].append(to_dev_fp16_kn(g32(dp + "linear_fc1.weight")))
            self._ds["fc1_b"].append(b16(dp + "linear_fc1.bias"))
            self._ds["fc2_w"].append(to_dev_fp16_kn(g32(dp + "linear_fc2.weight")))
            self._ds["fc2_b"].append(b16(dp + "linear_fc2.bias"))
        self._m_n_w = b16(vp + "merger.norm.weight")
        self._m_n_b = b16(vp + "merger.norm.bias")
        self._m0_w = to_dev_fp16_kn(g32(vp + "merger.linear_fc1.weight"))
        self._m0_b = b16(vp + "merger.linear_fc1.bias")
        self._m2_w = to_dev_fp16_kn(g32(vp + "merger.linear_fc2.weight"))
        self._m2_b = b16(vp + "merger.linear_fc2.bias")
        # patch embed Conv3d [1024, 3, 2, 16, 16] → GEMM [1536, 1024];
        # its bias is folded into the pos-add constant (set later).
        pe = g32(vp + "patch_embed.proj.weight").reshape(VIS_D, PATCH_FLAT)
        self._pe_w = pe.t().contiguous().to(fp16).cuda()
        self._pe_b32 = g32(vp + "patch_embed.proj.bias")     # host fp32
        self._pos_embed_ckpt = g32(vp + "pos_embed.weight")  # host fp32 [2304, 1024]

        # ── LM (fp8 W, RMS folds like vla4b; NO attention biases) ──
        lp = P + "qwenvl.model.language_model.layers."
        self._l = {k: [] for k in ("qkv_w", "o_w", "gate_w", "up_w",
                                   "down_w", "qnorm_w", "knorm_w")}
        for l in range(LM_L):
            bp = f"{lp}{l}."
            n_in = g32(bp + "input_layernorm.weight")
            n_post = g32(bp + "post_attention_layernorm.weight")
            qkv = torch.cat([g32(bp + "self_attn.q_proj.weight"),
                             g32(bp + "self_attn.k_proj.weight"),
                             g32(bp + "self_attn.v_proj.weight")], dim=0)
            q, s = to_dev_fp8_kn(qkv, fold=n_in)
            self._l["qkv_w"].append(q); w_scales_lm.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "self_attn.o_proj.weight"))
            self._l["o_w"].append(q); w_scales_lm.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.gate_proj.weight"), fold=n_post)
            self._l["gate_w"].append(q); w_scales_lm.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.up_proj.weight"), fold=n_post)
            self._l["up_w"].append(q); w_scales_lm.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.down_proj.weight"))
            self._l["down_w"].append(q); w_scales_lm.append(s)
            self._l["qnorm_w"].append(b16(bp + "self_attn.q_norm.weight"))
            self._l["knorm_w"].append(b16(bp + "self_attn.k_norm.weight"))
        self._lm_w_scales = torch.tensor(w_scales_lm, dtype=torch.float32,
                                         device="cuda")

        # ── Expert (fp8 W attn + shared expert; MoE tensors repacked
        # separately with per-expert scales — consumed in Stage 2) ──
        ep = P + "qwen_expert.model.layers."
        self._e = {k: [] for k in ("qkv_w", "qkv_b", "o_w", "sh_gate_w",
                                   "sh_up_w", "sh_down_w")}
        self._moe = {k: [] for k in ("gate_w32", "e_bias32", "gateup_fp8",
                                     "down_fp8", "gateup_scales",
                                     "down_scales")}
        self._ada = []
        for l in range(EXP_L):
            bp = f"{ep}{l}."
            qkv = torch.cat([g32(bp + "self_attn.q_proj.weight"),
                             g32(bp + "self_attn.k_proj.weight"),
                             g32(bp + "self_attn.v_proj.weight")], dim=0)
            q, s = to_dev_fp8_kn(qkv)
            self._e["qkv_w"].append(q); w_scales_exp.append(s)
            self._e["qkv_b"].append(torch.cat([
                sf.get(bp + "self_attn.q_proj.bias"),
                sf.get(bp + "self_attn.k_proj.bias"),
                sf.get(bp + "self_attn.v_proj.bias")]).to(fp16).cuda())
            q, s = to_dev_fp8_kn(g32(bp + "self_attn.o_proj.weight"))
            self._e["o_w"].append(q); w_scales_exp.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.shared_expert.gate_proj.weight"))
            self._e["sh_gate_w"].append(q); w_scales_exp.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.shared_expert.up_proj.weight"))
            self._e["sh_up_w"].append(q); w_scales_exp.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.shared_expert.down_proj.weight"))
            self._e["sh_down_w"].append(q); w_scales_exp.append(s)
            # routed-MoE tensors: gate+up merged per expert → (32, 1024, 768);
            # per-EXPERT fp8 scales (finer than per-3D-tensor — the
            # batched-GEMM shared-scale question is a repack-time report)
            gu = torch.cat([g32(bp + "mlp.experts.gate_proj"),
                            g32(bp + "mlp.experts.up_proj")], dim=1)  # (32,1024,768)
            dn = g32(bp + "mlp.experts.down_proj")                    # (32,768,512)
            gu_q, gu_s, dn_q, dn_s = [], [], [], []
            for e in range(MOE_E):
                qe, se_ = _quantize_fp8(gu[e].t().contiguous().cuda())
                gu_q.append(qe); gu_s.append(se_)
                qe, se_ = _quantize_fp8(dn[e].t().contiguous().cuda())
                dn_q.append(qe); dn_s.append(se_)
            self._moe["gateup_fp8"].append(torch.stack(gu_q))
            self._moe["down_fp8"].append(torch.stack(dn_q))
            self._moe["gateup_scales"].append(gu_s)
            self._moe["down_scales"].append(dn_s)
            self._moe["gate_w32"].append(g32(bp + "mlp.gate.weight").cuda())
            self._moe["e_bias32"].append(g32(bp + "mlp.e_score_correction_bias").cuda())
            self._ada.append({
                "in_w": g32(bp + "input_layernorm.weight"),
                "in_gw": g32(bp + "input_layernorm.gamma.weight"),
                "in_gb": g32(bp + "input_layernorm.gamma.bias"),
                "in_bw": g32(bp + "input_layernorm.beta.weight"),
                "in_bb": g32(bp + "input_layernorm.beta.bias"),
                "po_w": g32(bp + "post_attention_layernorm.weight"),
                "po_gw": g32(bp + "post_attention_layernorm.gamma.weight"),
                "po_gb": g32(bp + "post_attention_layernorm.gamma.bias"),
                "po_bw": g32(bp + "post_attention_layernorm.beta.weight"),
                "po_bb": g32(bp + "post_attention_layernorm.beta.bias"),
            })
        self._exp_w_scales = torch.tensor(w_scales_exp, dtype=torch.float32,
                                          device="cuda")
        self._final_norm_w = b16(P + "qwen_expert.model.norm.weight")

        # ── heads (fp16 cuBLAS path; identical structure to vla4b,
        # dims 55) ──
        self._ain_w = g32("model.action_in_proj.weight").t().contiguous().to(fp16).cuda()
        self._ain_b = b16("model.action_in_proj.bias")
        atm_in_w = g32("model.action_time_mlp_in.weight")     # [768, 1536]
        self._atm_a_w = atm_in_w[:, :EXP_D].t().contiguous().to(fp16).cuda()
        self._atm_t_w32 = atm_in_w[:, EXP_D:]                 # host fp32
        self._atm_in_b32 = g32("model.action_time_mlp_in.bias")
        self._atm_out_w = g32("model.action_time_mlp_out.weight").t().contiguous().to(fp16).cuda()
        self._atm_out_b = b16("model.action_time_mlp_out.bias")
        self._state_w = g32("model.state_proj.weight").t().contiguous().to(fp16).cuda()
        self._state_b = b16("model.state_proj.bias")
        dt = -1.0 / STEPS
        self._aout_w_dt = (g32("model.action_out_proj.weight") * dt).t().contiguous().to(fp16).cuda()
        self._aout_b_dt = (g32("model.action_out_proj.bias") * dt).to(fp16).cuda()

        # ── align-query constants (16 × 2560, trained; MUST be in the
        # prefix — see mapping_memo.md correction #7) ──
        def _group(t):     # (256, 2560) → (8, 2560)
            return t.view(NUM_TASK_TOKENS, -1, t.shape[-1]).mean(dim=1)

        cur = torch.cat([_group(g32("model.depth_align_embs")),
                         _group(g32("model.current_video_align_embs"))], dim=-1)
        cur = torch.nn.functional.linear(
            cur, g32("model.current_shared_task_proj.weight"),
            g32("model.current_shared_task_proj.bias"))
        fut = torch.cat([_group(g32("model.future_depth_align_embs")),
                         _group(g32("model.future_video_align_embs"))], dim=-1)
        fut = torch.nn.functional.linear(
            fut, g32("model.future_shared_task_proj.weight"),
            g32("model.future_shared_task_proj.bias"))
        self._align_emb = torch.cat([cur, fut], dim=0).to(fp16).cuda()  # (16, 2560)

        # embed_tokens stays on CPU (fp16); gathered per prompt
        self._embed_cpu = sf.get(
            P + "qwenvl.model.language_model.embed_tokens.weight").to(fp16)

        logger.info("Weights loaded + quantized in %.1fs", time.time() - t0)

    def moe_scale_spread(self) -> dict:
        """Per-layer max/min per-expert fp8 scale ratio — the repack
        report that decides batched-GEMM shared scales vs per-expert
        loop scales (see repack_calib_plan.md)."""
        out = {}
        for l in range(EXP_L):
            for name in ("gateup_scales", "down_scales"):
                s = self._moe[name][l]
                out[f"L{l:02d}.{name}"] = max(s) / max(min(s), 1e-12)
        return out

    # ────────────────────────────────────────────────────────────
    # ViT constants
    # ────────────────────────────────────────────────────────────

    def _set_pos_embed_table(self, table: Optional[torch.Tensor]):
        """``table``: (VIS_S, VIS_D) fp32 — the stock model's
        ``fast_pos_embed_interpolate(grid_thw)`` output for the fixed
        3×(1,16,16) grid, produced once by the repack harness (we do
        NOT re-implement the bilinear interpolation). The patch-embed
        conv bias is folded in here."""
        if table is None:
            self._pos_add = None
            logger.warning("pos_embed table not provided — run the repack "
                           "harness step R1 (see repack_calib_plan.md) "
                           "before inference")
            return
        assert table.shape == (VIS_S, VIS_D), table.shape
        self._pos_add = (table.to(torch.float32)
                         + self._pe_b32[None, :]).to(fp16).cuda()

    def _alloc_vit_buffers(self):
        S, D, H = VIS_S, VIS_D, VIS_H
        D4 = 4 * D
        self._patches = torch.zeros(S, PATCH_FLAT, dtype=fp16, device="cuda")
        self._vx = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_xn = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_qkv_buf = torch.empty(S, 3 * D, dtype=fp16, device="cuda")
        self._vq = torch.empty(S, D, dtype=fp16, device="cuda")
        self._vk = torch.empty(S, D, dtype=fp16, device="cuda")
        self._vv = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_attn = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_fg = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_hid = torch.empty(S, H, dtype=fp16, device="cuda")
        self._mrg_in = torch.empty(VIS_SM, D4, dtype=fp16, device="cuda")
        self._m0_buf = torch.empty(VIS_SM, D4, dtype=fp16, device="cuda")
        self._vis_emb = torch.empty(VIS_SM, LM_D, dtype=fp16, device="cuda")
        self._ds_out = [torch.empty(VIS_SM, LM_D, dtype=fp16, device="cuda")
                        for _ in VIS_DEEPSTACK]
        self._v_lse = torch.zeros(self.num_views * VIS_NH * 256,
                                  dtype=torch.float32, device="cuda")
        self._calib_hid = None
        self._ones = torch.ones(LM_D, dtype=fp16, device="cuda")

    def _build_vision_tables(self):
        lay = build_vision_layout(num_images=self.num_views)
        self._vis_cos = lay.cos.to(fp16).cuda()
        self._vis_sin = lay.sin.to(fp16).cuda()

    # ────────────────────────────────────────────────────────────
    # set_prompt
    # ────────────────────────────────────────────────────────────

    _PROMPT_ZERO_BUFFERS = (
        "_l_fp8", "_l_lse", "_Kc", "_Vc", "_x_t", "_state_in",
        "_s_fp8", "_s_lse1", "_s_lse2", "_calib_hid",
    )
    _PROMPT_UNIT_BUFFERS = ("_lm_act_scales", "_exp_act_scales")

    def _reset_prompt_buffers(self):
        """Restore fresh initialized contents without changing graph pointers."""
        for name in self._PROMPT_ZERO_BUFFERS:
            getattr(self, name).zero_()
        for name in self._PROMPT_UNIT_BUFFERS:
            getattr(self, name).fill_(1.0)
        self._real_data_calibrated = False

    def set_prompt(self, prompt: Union[str, list, np.ndarray]):
        self._prompt_buffers_reused = False
        if isinstance(prompt, str):
            token_ids = self._tokenize(prompt)
        else:
            token_ids = np.asarray(prompt, dtype=np.int64)
        n_lang = len(token_ids)
        Se = self.num_views * IMG_BLOCK + n_lang + ALIGN_TOKENS
        layout = (self.num_views, n_lang, bool(self.use_cuda_graph))
        emb = self._embed_cpu[torch.from_numpy(token_ids)]
        if self._prompt_layout == layout:
            # M-RoPE, step tables and attention dimensions depend on the
            # layout, not token values. The staged engine owns a separate
            # graph, so graph_captured need not equal use_cuda_graph here.
            self._prompt_layout = None
            self._lang_emb.copy_(emb)
            self._reset_prompt_buffers()
            # The following calibration kernels use native stream zero.
            torch.cuda.synchronize()
            self._prompt_layout = layout
            self._prompt_buffers_reused = True
            self.prompt_updates += 1
            self.prompt_buffer_reuses += 1
            logger.info("set_prompt reused buffers (n_lang=%d, Se=%d)", n_lang, Se)
            return

        self._prompt_layout = None
        self.graph_captured = False
        self.Se = Se
        self.n_lang = n_lang
        self.total_keys = Se + SUF

        self._lang_emb = emb.cuda()
        # vision boundary token embeddings (constant)
        vs, ve = 151652, 151653          # vision_start / vision_end ids
        self._bound_emb = self._embed_cpu[torch.tensor([vs, ve])].cuda()

        self._alloc_prompt_buffers(Se)
        self._build_lm_tables(n_lang)
        self._build_style_tables()
        self._build_attn()
        self._real_data_calibrated = False
        if self.use_cuda_graph:
            self._capture_graphs()
        self.graph_captured = self.use_cuda_graph
        self._prompt_layout = layout
        self.prompt_updates += 1
        logger.info("set_prompt done (n_lang=%d, Se=%d)", n_lang, Se)

    def _tokenize(self, text: str) -> np.ndarray:
        """Replicates lingbotvla prepare_language for V2
        (transform.py :454-474): Qwen3 chat template
        (single user turn, tokenize=False, add_generation_prompt=False)
        then plain tokenization. The reference right-pads to
        tokenizer_max_length=72 with fully-masked pads → dense here.
        Prompt-parity gate in Stage 2 compares ids bit-exact."""
        from transformers import AutoTokenizer
        if not hasattr(self, "_tok"):
            self._tok = AutoTokenizer.from_pretrained(
                os.environ.get("QWEN3VL_PATH", "Qwen/Qwen3-VL-4B-Instruct"))
        templ = self._tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=False)
        ids = self._tok(templ, add_special_tokens=False, truncation=True,
                        max_length=72)["input_ids"]
        return np.asarray(ids, dtype=np.int64)

    def _alloc_prompt_buffers(self, Se):
        D, H = LM_D, LM_H
        self._enc_x = torch.empty(Se, D, dtype=fp16, device="cuda")
        self._l_fp8 = torch.zeros(Se * H, dtype=torch.uint8, device="cuda")
        self._l_qkv_buf = torch.empty(Se, LM_QKV_OUT, dtype=fp16, device="cuda")
        self._l_q = torch.empty(Se, LM_DQ, dtype=fp16, device="cuda")
        self._l_q2 = torch.empty(Se, LM_DQ, dtype=fp16, device="cuda")
        self._l_kt = torch.empty(Se, LM_DKV, dtype=fp16, device="cuda")
        self._l_attn = torch.empty(Se, LM_DQ, dtype=fp16, device="cuda")
        self._l_fg = torch.empty(Se, D, dtype=fp16, device="cuda")
        self._l_gate = torch.empty(Se, H, dtype=fp16, device="cuda")
        self._l_up = torch.empty(Se, H, dtype=fp16, device="cuda")
        self._l_xn = torch.empty(Se, D, dtype=fp16, device="cuda")
        # bf16 scratch for the FA2 causal prefill entry (bf16-hd128 is
        # the only causal instantiation — see Vla2Fa2Attn)
        bf16 = torch.bfloat16
        self._l_qb = torch.empty(Se, LM_DQ, dtype=bf16, device="cuda")
        self._l_kb = torch.empty(Se, LM_DKV, dtype=bf16, device="cuda")
        self._l_vb = torch.empty(Se, LM_DKV, dtype=bf16, device="cuda")
        self._l_ob = torch.empty(Se, LM_DQ, dtype=bf16, device="cuda")
        se_r = ((Se + 127) // 128) * 128
        self._l_lse = torch.zeros(LM_NH * se_r, dtype=torch.float32,
                                  device="cuda")
        self._Kc = torch.zeros(LM_L, self.total_keys, KV_ROW,
                               dtype=fp16, device="cuda")
        self._Vc = torch.zeros(LM_L, self.total_keys, KV_ROW,
                               dtype=fp16, device="cuda")
        self._lm_act_scales = torch.full((LM_L * 4,), 1.0,
                                         dtype=torch.float32, device="cuda")
        # expert buffers
        self._x_t = torch.zeros(CHUNK, ADIM, dtype=fp16, device="cuda")
        self._state_in = torch.zeros(1, SDIM, dtype=fp16, device="cuda")
        self._state_emb = torch.empty(1, EXP_D, dtype=fp16, device="cuda")
        self._a_emb = torch.empty(CHUNK, EXP_D, dtype=fp16, device="cuda")
        self._tmp50 = torch.empty(CHUNK, EXP_D, dtype=fp16, device="cuda")
        self._sx = torch.empty(SUF, EXP_D, dtype=fp16, device="cuda")
        # M2 fix (2026-08-25, Thor bring-up): the o-in quant point writes
        # SUF*LM_DQ (51*4096) fp8 bytes into this buffer (expert_forward
        # `_quant_point(attn_out, x_fp8, a_o, S * LM_DQ)`), so sizing it
        # SUF*max(EXP_D, SHARED_H) = 51*768 overflowed by ~170 KB into
        # neighbouring allocations. Size = the max over all four expert
        # quant sites (qkv-in 768, o-in 4096, ffn-in 768, shared-down 704).
        self._s_fp8 = torch.zeros(SUF * LM_DQ,
                                  dtype=torch.uint8, device="cuda")
        self._s_gatebuf = torch.empty(SUF, EXP_D, dtype=fp16, device="cuda")
        self._s_qkv = torch.empty(SUF, LM_QKV_OUT, dtype=fp16, device="cuda")
        self._s_q = torch.empty(SUF, LM_DQ, dtype=fp16, device="cuda")
        self._s_attn = torch.empty(SUF, LM_DQ, dtype=fp16, device="cuda")
        self._s_fg = torch.empty(SUF, EXP_D, dtype=fp16, device="cuda")
        self._s_shgate = torch.empty(SUF, SHARED_H, dtype=fp16, device="cuda")
        self._s_shup = torch.empty(SUF, SHARED_H, dtype=fp16, device="cuda")
        self._s_moe_out = torch.empty(SUF, EXP_D, dtype=fp16, device="cuda")
        self._s_xn = torch.empty(SUF, EXP_D, dtype=fp16, device="cuda")
        self._s_lse1 = torch.zeros(LM_NH * 128, dtype=torch.float32,
                                   device="cuda")
        self._s_lse2 = torch.zeros(LM_NH * 128, dtype=torch.float32,
                                   device="cuda")
        self._exp_act_scales = torch.full((EXP_L * 4,), 1.0,
                                          dtype=torch.float32, device="cuda")
        n_scratch = max(Se * H, VIS_S * VIS_H, SUF * SHARED_H)
        self._calib_hid = torch.zeros(n_scratch, dtype=fp16, device="cuda")

    def _build_lm_tables(self, n_lang):
        pre_cos, pre_sin, suf_cos, suf_sin = build_joint_rope_tables(
            n_lang, num_images=self.num_views)
        assert pre_cos.shape[0] == self.Se
        self._pre_cos = pre_cos.cuda()
        self._pre_sin = pre_sin.cuda()
        self._suf_cos = suf_cos.cuda()
        self._suf_sin = suf_sin.cuda()

    def _build_style_tables(self):
        sa, sf_, tc = build_step_tables(
            self._ada, self._atm_t_w32, self._atm_in_b32, steps=STEPS)
        self._style_attn = sa.cuda()
        self._style_ffn = sf_.cuda()
        self._t_contrib = tc.cuda()

    def _ensure_fmha(self) -> bool:
        """Load the strided cutlass FMHA lib (groot_n17 vit idiom) —
        the vendored FA2 lacks a hd64 instantiation."""
        if getattr(self, "_fmha_vit", None) is not None:
            return self._fmha_vit
        self._fmha_vit = False
        if fvk is not None:
            p = pathlib.Path(fvk.__file__).parent / "libfmha_fp16_strided.so"
            if p.exists():
                fvk.load_fmha_strided_library(str(p))
                self._fmha_vit = True
                logger.info("strided FMHA loaded for the ViT site")
        return self._fmha_vit

    def _build_attn(self):
        layer_stride = self.total_keys * KV_ROW
        kc, vc = self._Kc.data_ptr(), self._Vc.data_ptr()
        self._attn = Vla2Fa2Attn(
            self._fa2,
            vit={"q": self._vq.data_ptr(), "k": self._vk.data_ptr(),
                 "v": self._vv.data_ptr(), "o": self._v_attn.data_ptr(),
                 "lse": self._v_lse.data_ptr(), "nv": self.num_views,
                 "use_fmha": self._ensure_fmha()},
            lm={"q": self._l_q2.data_ptr(), "o": self._l_attn.data_ptr(),
                "lse": self._l_lse.data_ptr(), "Kc": kc, "Vc": vc,
                "layer_stride_elems": layer_stride,
                "qb": self._l_qb.data_ptr(), "kb": self._l_kb.data_ptr(),
                "vb": self._l_vb.data_ptr(), "ob": self._l_ob.data_ptr()},
            suffix={"q": self._s_q.data_ptr(), "o": self._s_attn.data_ptr(),
                    "lse1": self._s_lse1.data_ptr(),
                    "lse2": self._s_lse2.data_ptr(),
                    "Kc": kc, "Vc": vc,
                    "layer_stride_elems": layer_stride},
            fvk_mod=fvk,
        )

    # ────────────────────────────────────────────────────────────
    # Stage-A KV handoff (torch bf16 backbone prefill → engine loop)
    # ────────────────────────────────────────────────────────────

    def load_prefix_kv_from_torch(self, past_key_values: dict,
                                  prefix_pad_mask: torch.Tensor):
        """Fill KV slab rows [0, Se) from the stock model's cache.

        See ``KV_HANDOFF_CONTRACT``. ``past_key_values`` is the dict
        the stock ``fill_kv_cache=True`` pass produced;
        ``prefix_pad_mask`` is the (1, Se_pad) bool pad mask used in
        that pass.
        """
        valid = prefix_pad_mask.reshape(-1).bool()
        assert int(valid.sum()) == self.Se, (
            f"valid prefix rows {int(valid.sum())} != engine Se {self.Se} "
            "— prompt-parity gate must run first")
        for l in range(LM_L):
            k = past_key_values[l]["key_states"]
            v = past_key_values[l]["value_states"]
            k = k.reshape(k.shape[1], LM_NKV * LM_HD)[valid]
            v = v.reshape(v.shape[1], LM_NKV * LM_HD)[valid]
            self._Kc[l, :self.Se].copy_(k.to(fp16))
            self._Vc[l, :self.Se].copy_(v.to(fp16))
        torch.cuda.synchronize()

    # ────────────────────────────────────────────────────────────
    # Pipeline dict builders
    # ────────────────────────────────────────────────────────────

    def _vit_args(self):
        bufs = {
            "x": self._vx.data_ptr(), "xn": self._v_xn.data_ptr(),
            "qkv": self._v_qkv_buf.data_ptr(), "q": self._vq.data_ptr(),
            "k": self._vk.data_ptr(), "v": self._vv.data_ptr(),
            "attn_out": self._v_attn.data_ptr(), "fg": self._v_fg.data_ptr(),
            "hid": self._v_hid.data_ptr(),
            "mrg_in": self._mrg_in.data_ptr(), "m0": self._m0_buf.data_ptr(),
            "vis_emb": self._vis_emb.data_ptr(),
            "ds_out": [t.data_ptr() for t in self._ds_out],
        }
        weights = {
            "qkv_w": [w.data_ptr() for w in self._v["qkv_w"]],
            "qkv_b": [w.data_ptr() for w in self._v["qkv_b"]],
            "o_w": [w.data_ptr() for w in self._v["o_w"]],
            "o_b": [w.data_ptr() for w in self._v["o_b"]],
            "fc1_w": [w.data_ptr() for w in self._v["fc1_w"]],
            "fc1_b": [w.data_ptr() for w in self._v["fc1_b"]],
            "fc2_w": [w.data_ptr() for w in self._v["fc2_w"]],
            "fc2_b": [w.data_ptr() for w in self._v["fc2_b"]],
            "n1_w": [w.data_ptr() for w in self._v["n1_w"]],
            "n1_b": [w.data_ptr() for w in self._v["n1_b"]],
            "n2_w": [w.data_ptr() for w in self._v["n2_w"]],
            "n2_b": [w.data_ptr() for w in self._v["n2_b"]],
            "ds_n_w": [w.data_ptr() for w in self._ds["n_w"]],
            "ds_n_b": [w.data_ptr() for w in self._ds["n_b"]],
            "ds_fc1_w": [w.data_ptr() for w in self._ds["fc1_w"]],
            "ds_fc1_b": [w.data_ptr() for w in self._ds["fc1_b"]],
            "ds_fc2_w": [w.data_ptr() for w in self._ds["fc2_w"]],
            "ds_fc2_b": [w.data_ptr() for w in self._ds["fc2_b"]],
            "m_n_w": self._m_n_w.data_ptr(), "m_n_b": self._m_n_b.data_ptr(),
            "m0_w": self._m0_w.data_ptr(), "m0_b": self._m0_b.data_ptr(),
            "m2_w": self._m2_w.data_ptr(), "m2_b": self._m2_b.data_ptr(),
            "cos": self._vis_cos.data_ptr(), "sin": self._vis_sin.data_ptr(),
        }
        dims = {"S": VIS_S, "D": VIS_D, "H": VIS_H, "L": VIS_L,
                "S_m": VIS_SM, "D_enc": LM_D}
        return bufs, weights, dims

    def _ds_rows(self):
        """(row0, nrows) of the patch rows per image inside the prefix
        (visual_pos_masks covers ONLY the 64 patch rows, not the
        boundary tokens — modeling_..._v2.py :553-554)."""
        return [(i * IMG_BLOCK + 1, 64) for i in range(self.num_views)]

    def _lm_args(self):
        bufs = {
            "x": self._enc_x.data_ptr(), "x_fp8": self._l_fp8.data_ptr(),
            "qkv": self._l_qkv_buf.data_ptr(), "q": self._l_q.data_ptr(),
            "q2": self._l_q2.data_ptr(), "kt": self._l_kt.data_ptr(),
            "attn_out": self._l_attn.data_ptr(), "fg": self._l_fg.data_ptr(),
            "gate": self._l_gate.data_ptr(), "up": self._l_up.data_ptr(),
            "xn": self._l_xn.data_ptr(),
            "calib_hid": self._calib_hid.data_ptr(),
            "ones": self._ones.data_ptr(),
        }
        weights = {
            "qnorm_w": [w.data_ptr() for w in self._l["qnorm_w"]],
            "knorm_w": [w.data_ptr() for w in self._l["knorm_w"]],
            "Kc": self._Kc.data_ptr(), "Vc": self._Vc.data_ptr(),
            "cos": self._pre_cos.data_ptr(), "sin": self._pre_sin.data_ptr(),
            "ds_out": [t.data_ptr() for t in self._ds_out],
            "ds_rows": self._ds_rows(),
            "act_scales": self._lm_act_scales.data_ptr(),
        }
        if self.lm_prefill_precision == "fp16":
            assert self._lm16 is not None, "call load_lm_fp16_stack first"
            for k in ("qkv_w16", "o_w16", "gate_w16", "up_w16",
                      "down_w16", "n_in_w", "n_post_w"):
                weights[k] = [w.data_ptr() for w in self._lm16[k]]
        else:
            for src, dst in (("qkv_w", "qkv_w"), ("o_w", "o_w"),
                             ("gate_w", "gate_w"), ("up_w", "up_w"),
                             ("down_w", "down_w")):
                weights[dst] = [w.data_ptr() for w in self._l[src]]
            weights["w_scales"] = self._lm_w_scales.data_ptr()
        dims = {"Se": self.Se, "D": LM_D, "H": LM_H, "L": LM_L,
                "total_keys": self.total_keys, "kv_row": KV_ROW}
        return bufs, weights, dims

    def _expert_args(self):
        bufs = {
            "x_t": self._x_t.data_ptr(),
            "state_emb": self._state_emb.data_ptr(),
            "a_emb": self._a_emb.data_ptr(), "tmp": self._tmp50.data_ptr(),
            "x": self._sx.data_ptr(), "x_fp8": self._s_fp8.data_ptr(),
            "gatebuf": self._s_gatebuf.data_ptr(),
            "qkv": self._s_qkv.data_ptr(), "q": self._s_q.data_ptr(),
            "attn_out": self._s_attn.data_ptr(), "fg": self._s_fg.data_ptr(),
            "sh_gate": self._s_shgate.data_ptr(),
            "sh_up": self._s_shup.data_ptr(),
            "moe_out": self._s_moe_out.data_ptr(),
            "xn": self._s_xn.data_ptr(),
            "calib_hid": self._calib_hid.data_ptr(),
            "ones": self._ones.data_ptr(),
        }
        weights = {
            "qkv_w": [w.data_ptr() for w in self._e["qkv_w"]],
            "qkv_b": [w.data_ptr() for w in self._e["qkv_b"]],
            "o_w": [w.data_ptr() for w in self._e["o_w"]],
            "sh_gate_w": [w.data_ptr() for w in self._e["sh_gate_w"]],
            "sh_up_w": [w.data_ptr() for w in self._e["sh_up_w"]],
            "sh_down_w": [w.data_ptr() for w in self._e["sh_down_w"]],
            "style_attn": self._style_attn.data_ptr(),
            "style_ffn": self._style_ffn.data_ptr(),
            "Kc": self._Kc.data_ptr(), "Vc": self._Vc.data_ptr(),
            "cos": self._suf_cos.data_ptr(), "sin": self._suf_sin.data_ptr(),
            "final_norm_w": self._final_norm_w.data_ptr(),
            "ain_w": self._ain_w.data_ptr(), "ain_b": self._ain_b.data_ptr(),
            "atm_a_w": self._atm_a_w.data_ptr(),
            "t_contrib": self._t_contrib.data_ptr(),
            "atm_out_w": self._atm_out_w.data_ptr(),
            "atm_out_b": self._atm_out_b.data_ptr(),
            "aout_w_dt": self._aout_w_dt.data_ptr(),
            "aout_b_dt": self._aout_b_dt.data_ptr(),
            "act_scales": self._exp_act_scales.data_ptr(),
            "w_scales": self._exp_w_scales.data_ptr(),
        }
        dims = {"D": EXP_D, "L": EXP_L, "steps": STEPS,
                "Se": self.Se, "total_keys": self.total_keys,
                "kv_row": KV_ROW}
        return bufs, weights, dims

    # ────────────────────────────────────────────────────────────
    # Forward pieces (shared by eager + capture)
    # ────────────────────────────────────────────────────────────

    def _run_vis(self, stream_int):
        assert self._pos_add is not None, "pos_embed table missing (repack R1)"
        # patch embed GEMM [768, 1536] @ [1536, 1024], then + (pos+bias)
        fvk.gmm_fp16(self._ctx, self._patches.data_ptr(),
                     self._pe_w.data_ptr(), self._vx.data_ptr(),
                     VIS_S, VIS_D, PATCH_FLAT, 0.0, stream_int)
        # pos_add is a full [S, D] matrix — residual add, not a bias row
        fvk.residual_add_fp16(self._vx.data_ptr(), self._pos_add.data_ptr(),
                              VIS_S * VIS_D, stream_int)
        b, w, d = self._vit_args()
        vit_forward(fvk, self._ctx, b, w, d, stream_int, attn=self._attn)

    def _stage_enc_x(self):
        """Assemble the prefix residual stream:
        3 × [start | 64 vis rows | end] + lang + 16 align constants."""
        for i in range(self.num_views):
            r0 = i * IMG_BLOCK
            self._enc_x[r0:r0 + 1].copy_(self._bound_emb[0:1])
            self._enc_x[r0 + 1:r0 + 65].copy_(self._vis_emb[i * 64:(i + 1) * 64])
            self._enc_x[r0 + 65:r0 + 66].copy_(self._bound_emb[1:2])
        base = self.num_views * IMG_BLOCK
        self._enc_x[base:base + self.n_lang].copy_(self._lang_emb)
        self._enc_x[base + self.n_lang:self.Se].copy_(self._align_emb)

    def _run_lm_prefill_only(self, stream_int, calibrate=False,
                             stage=True):
        """Engine LM prefill: fill KV slab rows [0, Se) from the staged
        prefix (P1 gate entry; also the latency prefill segment)."""
        if stage:
            self._stage_enc_x()
        b, w, d = self._lm_args()
        lm_prefill_forward(fvk, b, w, d, stream_int, attn=self._attn,
                           calibrate=calibrate, ctx=self._ctx,
                           precision=self.lm_prefill_precision)

    def _run_lm_expert(self, stream_int, calibrate=False):
        self._run_lm_prefill_only(stream_int, calibrate=calibrate)
        self._run_expert_only(stream_int, calibrate=calibrate)

    def _run_expert_only(self, stream_int, calibrate=False, calibration_observer=None):
        """The Stage-A entry: assumes prefix KV rows are already in the
        slab (either from lm_prefill_forward or from
        load_prefix_kv_from_torch)."""
        fvk.gmm_fp16(self._ctx, self._state_in.data_ptr(),
                     self._state_w.data_ptr(), self._state_emb.data_ptr(),
                     1, EXP_D, SDIM, 0.0, stream_int)
        fvk.add_bias_fp16(self._state_emb.data_ptr(),
                          self._state_b.data_ptr(), 1, EXP_D, stream_int)
        b, w, d = self._expert_args()
        expert_forward(fvk, self._ctx, b, w, d, stream_int, attn=self._attn,
                       routed_moe_fn=self._routed_moe_fn,
                       calibrate=calibrate,
                       calibration_observer=calibration_observer,
                       ffn_xn_fp16=getattr(self, "_ffn_xn_fp16", False))

    # ────────────────────────────────────────────────────────────
    # Graph capture (vla4b two-graph discipline)
    # ────────────────────────────────────────────────────────────

    def _capture_graphs(self):
        torch.cuda.synchronize()
        for _ in range(2):
            self._run_vis(0)
            self._run_lm_expert(0)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        s_int = stream.cuda_stream
        self._g_vis = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            self._g_vis.capture_begin()
            self._run_vis(s_int)
            self._g_vis.capture_end()
        torch.cuda.synchronize()
        self._g_lm = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            self._g_lm.capture_begin()
            self._run_lm_expert(s_int)
            self._g_lm.capture_end()
        torch.cuda.synchronize()
        self.prompt_graph_captures += 1
        logger.info("CUDA graphs captured (Se=%d)", self.Se)

    # ────────────────────────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────────────────────────

    def stage_inputs(self, pixel_patches, state, noise):
        """pixel_patches: (3*256, 1536) HF Qwen3-VL pixel_values layout
        (pre-normalized; raw HF order — NO permute). state: (55,).
        noise: (50, 55)."""
        p = torch.as_tensor(pixel_patches).reshape(VIS_S, PATCH_FLAT)
        self._patches.copy_(p.to(fp16))
        self._state_in.copy_(torch.as_tensor(state).reshape(1, SDIM).to(fp16))
        self._noise_host = torch.as_tensor(noise).reshape(CHUNK, ADIM).to(fp16)
        self._x_t.copy_(self._noise_host)

    def infer_staged(self, pixel_patches, state, noise):
        """Deterministic-input inference (parity gates + serving glue)."""
        t0 = time.perf_counter()
        self.stage_inputs(pixel_patches, state, noise)
        if not self._real_data_calibrated:
            self._run_vis(0)
            self._run_lm_expert(0, calibrate=True)
            torch.cuda.synchronize()
            self._x_t.copy_(self._noise_host)
            self._real_data_calibrated = True
            logger.info("Real-data FP8 calibration done (no recapture)")
        if self.graph_captured:
            self._g_vis.replay()
            self._g_lm.replay()
        else:
            self._run_vis(0)
            self._run_lm_expert(0)
        torch.cuda.synchronize()
        actions = self._x_t.float().cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latency_records.append(latency_ms)
        return {"actions": actions, "latency_ms": latency_ms}

    def denoise_only(self, state, noise):
        """Stage-A inference: prefix KV must already be loaded via
        ``load_prefix_kv_from_torch``. Runs only the 10-step loop."""
        self._state_in.copy_(torch.as_tensor(state).reshape(1, SDIM).to(fp16))
        self._x_t.copy_(torch.as_tensor(noise).reshape(CHUNK, ADIM).to(fp16))
        self._run_expert_only(0)
        torch.cuda.synchronize()
        return self._x_t.float().cpu().numpy()

    def get_latency_stats(self):
        arr = np.asarray(self.latency_records[1:] or self.latency_records)
        return {"p50": float(np.percentile(arr, 50)),
                "p99": float(np.percentile(arr, 99)),
                "mean": float(arr.mean()), "n": int(arr.size)}
