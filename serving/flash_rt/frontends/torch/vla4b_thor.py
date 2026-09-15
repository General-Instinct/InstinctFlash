"""FlashRT — Vla4bTorchFrontendThor: LingBot-VLA-4B inference on Thor SM110.

Derived from ``pi05_thor.py`` (buffer-owning torch frontend + fvk
pointer pipeline + CUDA-graph capture), adapted to the Qwen2.5-VL-3B
backbone + 36-layer action expert of ``robbyant/lingbot-vla-4b``.

Usage:
    pipe = Vla4bTorchFrontendThor("/path/to/hf/snapshot")
    pipe.set_prompt(token_ids)            # or "<bos>task\\n" text
    out = pipe.infer_staged(pixel_patches, state, noise)
    actions = out["actions"]              # (50, 75) normalized

Key differences from the pi05 frontend (ground truth: the lingbotvla
torch implementation — see models/vla4b/pipeline_thor.py docstring):
    * three vision streams through ONE windowed ViT (12×64-token
      window-batched FA2, fullatt layers [7,15,23,31] at 3×256)
    * joint LM+expert attention in LM head geometry with a compact
      2-head KV slab (FA2 GQA-native)
    * AdaRMS step tables: 36L × 2 norms × 10 steps precomputed — all
      1440 in-loop modulation GEMMs eliminated
    * mixed precision recipe (measured, gate_p4_ablate_*.json):
      ViT fully FP16 (Qwen2.5-VL vision outliers break per-tensor act
      FP8: alone costs action_meandiff 0.041 vs the 0.006 fp16 floor);
      LM + expert FP8 W+A per-tensor (amax/448 E4M3), fp32 norm folds,
      device scale pointers → real-data recalibration never recaptures
      graphs. The legacy FP16 vision tower can overflow on native-processed
      real images. Runtime supplies native BF16 vision embeddings instead;
      infer_staged checks finiteness before language/expert execution. This
      mixed recipe requires its own matched timing and closed-loop evaluation.
"""

import ctypes
import logging
import math
import pathlib
import time
from typing import Optional, Union

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.vla4b.pipeline_thor import (
    Vla4bFa2Attn,
    vit_forward,
    lm_prefill_forward,
    expert_forward,
)
from flash_rt.models.vla4b.rope_table import (
    build_vision_layout,
    build_lm_rope_tables,
)

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8 = torch.float8_e4m3fn

# ── architecture constants (locked by the checkpoint) ──
VIS_L, VIS_D, VIS_H_RAW, VIS_H_PAD = 32, 1280, 3420, 3424
VIS_NH, VIS_HD = 16, 80
VIS_FULLATT = frozenset((7, 15, 23, 31))
VIS_S = 768                 # 3 images × 256 patches
VIS_SM, VIS_DM = 192, 5120  # merged tokens, merger hidden
LM_L, LM_D, LM_H = 36, 2048, 11008
LM_NH, LM_NKV, LM_HD = 16, 2, 128
EXP_L, EXP_D, EXP_H = 36, 768, 2752
CHUNK, ADIM, SDIM = 50, 75, 75
STEPS = 10
SUF = CHUNK + 1             # 51 suffix tokens (state + actions)
KV_ROW = LM_NKV * LM_HD     # 256
PATCH_FLAT = 1176           # 3 × 2 × 14 × 14


def _quantize_fp8(w32: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric E4M3: amax/448 (pi05 recipe, fp32 input)."""
    amax = w32.abs().max().item()
    scale = max(amax / 448.0, 1e-12)
    w_fp8 = (w32 / scale).clamp(-448.0, 448.0).to(fp8)
    return w_fp8, float(scale)


class Vla4bTorchFrontendThor:
    """LingBot-VLA-4B inference using only flash_rt kernels + FA2."""

    def __init__(self, checkpoint_dir: Union[str, pathlib.Path],
                 num_views: int = 3, use_cuda_graph: bool = True,
                 fa2_lib=None):
        assert num_views == 3, "VLA-4B RoboTwin deployment is 3-camera"
        self.num_views = num_views
        self.use_cuda_graph = bool(use_cuda_graph)
        self.latency_records = []
        self.graph_captured = False
        self._real_data_calibrated = False
        self._prompt_token_count = None
        self.prompt_updates = 0
        self.prompt_buffer_reuses = 0
        self.prompt_graph_captures = 0

        self._ctx = fvk.FvkContext()
        if fa2_lib is None:
            from flash_rt import flash_rt_fa2 as fa2_lib
        self._fa2 = fa2_lib

        ckpt = pathlib.Path(checkpoint_dir)
        st_path = ckpt / "model.safetensors"
        if not st_path.exists():
            raise FileNotFoundError(st_path)
        self._load_weights(st_path)
        self._alloc_vit_buffers()
        self._build_vision_tables()
        logger.info("Vla4bTorchFrontendThor initialised")

    # ────────────────────────────────────────────────────────────
    # Weight loading (fp32 folds → fp8/fp16)
    # ────────────────────────────────────────────────────────────

    def _load_weights(self, st_path):
        from safetensors import safe_open
        t0 = time.time()
        sf = safe_open(str(st_path), framework="pt", device="cpu")
        P = "model.qwenvl_with_expert."

        def g32(key):
            return sf.get_tensor(key).to(torch.float32)

        def to_dev_fp8_kn(w32_nk, fold: Optional[torch.Tensor] = None):
            """HF [N, K] fp32 → engine [K, N] fp8 on cuda + scale.
            fold: per-input-channel vector multiplied into columns (K)."""
            w = w32_nk.cuda()
            if fold is not None:
                w = w * fold.cuda()[None, :]
            w_kn = w.t().contiguous()
            q, s = _quantize_fp8(w_kn)
            return q, s

        def b16(key):
            return sf.get_tensor(key).to(fp16).cuda()

        w_scales_lm, w_scales_exp = [], []

        # ── ViT (FULL FP16 — T2 recipe: per-tensor act FP8 breaks on
        # Qwen2.5-VL vision outliers; see gate_p4_ablate_*.json) ──
        vp = P + "qwenvl.visual."

        def to_dev_fp16_kn(w32_nk, fold: Optional[torch.Tensor] = None):
            w = w32_nk.cuda()
            if fold is not None:
                w = w * fold.cuda()[None, :]
            return w.t().contiguous().to(fp16)

        self._v_qkv_w, self._v_qkv_b = [], []
        self._v_o_w, self._v_o_b = [], []
        self._v_gate_w, self._v_gate_b = [], []
        self._v_up_w, self._v_up_b = [], []
        self._v_down_w, self._v_down_b = [], []
        for l in range(VIS_L):
            bp = f"{vp}blocks.{l}."
            n1 = g32(bp + "norm1.weight")
            n2 = g32(bp + "norm2.weight")
            self._v_qkv_w.append(to_dev_fp16_kn(g32(bp + "attn.qkv.weight"),
                                                fold=n1))
            self._v_qkv_b.append(b16(bp + "attn.qkv.bias"))
            self._v_o_w.append(to_dev_fp16_kn(g32(bp + "attn.proj.weight")))
            self._v_o_b.append(b16(bp + "attn.proj.bias"))
            self._v_gate_w.append(to_dev_fp16_kn(
                g32(bp + "mlp.gate_proj.weight"), fold=n2))
            self._v_gate_b.append(b16(bp + "mlp.gate_proj.bias"))
            self._v_up_w.append(to_dev_fp16_kn(
                g32(bp + "mlp.up_proj.weight"), fold=n2))
            self._v_up_b.append(b16(bp + "mlp.up_proj.bias"))
            self._v_down_w.append(to_dev_fp16_kn(
                g32(bp + "mlp.down_proj.weight")))
            self._v_down_b.append(b16(bp + "mlp.down_proj.bias"))
        # merger: fold ln_q (tiled ×4) into mlp.0
        lnq = g32(vp + "merger.ln_q.weight")
        self._m0_w = to_dev_fp16_kn(g32(vp + "merger.mlp.0.weight"),
                                    fold=lnq.repeat(4))
        self._m0_b = b16(vp + "merger.mlp.0.bias")
        self._m2_w = to_dev_fp16_kn(g32(vp + "merger.mlp.2.weight"))
        self._m2_b = b16(vp + "merger.mlp.2.bias")
        # patch embed (fp16 GEMM): [1280,3,2,14,14] → [1176, 1280]
        pe = g32(vp + "patch_embed.proj.weight").reshape(VIS_D, PATCH_FLAT)
        self._pe_w = pe.t().contiguous().to(fp16).cuda()

        # ── LM ──
        lp = P + "qwenvl.model.layers."
        self._l_qkv_w, self._l_qkv_b = [], []
        self._l_o_w = []
        self._l_gate_w, self._l_up_w, self._l_down_w = [], [], []
        for l in range(LM_L):
            bp = f"{lp}{l}."
            n_in = g32(bp + "input_layernorm.weight")
            n_post = g32(bp + "post_attention_layernorm.weight")
            qw = g32(bp + "self_attn.q_proj.weight")
            kw = g32(bp + "self_attn.k_proj.weight")
            vw = g32(bp + "self_attn.v_proj.weight")
            qkv = torch.cat([qw, kw, vw], dim=0)
            q, s = to_dev_fp8_kn(qkv, fold=n_in)
            self._l_qkv_w.append(q); w_scales_lm.append(s)
            self._l_qkv_b.append(torch.cat([
                sf.get_tensor(bp + "self_attn.q_proj.bias"),
                sf.get_tensor(bp + "self_attn.k_proj.bias"),
                sf.get_tensor(bp + "self_attn.v_proj.bias")]).to(fp16).cuda())
            q, s = to_dev_fp8_kn(g32(bp + "self_attn.o_proj.weight"))
            self._l_o_w.append(q); w_scales_lm.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.gate_proj.weight"), fold=n_post)
            self._l_gate_w.append(q); w_scales_lm.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.up_proj.weight"), fold=n_post)
            self._l_up_w.append(q); w_scales_lm.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.down_proj.weight"))
            self._l_down_w.append(q); w_scales_lm.append(s)
        self._lm_w_scales = torch.tensor(w_scales_lm, dtype=torch.float32,
                                         device="cuda")

        # ── Expert ──
        ep = P + "qwen_expert.model.layers."
        self._e_qkv_w, self._e_qkv_b = [], []
        self._e_o_w = []
        self._e_gate_w, self._e_up_w, self._e_down_w = [], [], []
        # host fp32 adanorm params for style-table building
        self._ada = []
        for l in range(EXP_L):
            bp = f"{ep}{l}."
            qw = g32(bp + "self_attn.q_proj.weight")
            kw = g32(bp + "self_attn.k_proj.weight")
            vw = g32(bp + "self_attn.v_proj.weight")
            q, s = to_dev_fp8_kn(torch.cat([qw, kw, vw], dim=0))
            self._e_qkv_w.append(q); w_scales_exp.append(s)
            self._e_qkv_b.append(torch.cat([
                sf.get_tensor(bp + "self_attn.q_proj.bias"),
                sf.get_tensor(bp + "self_attn.k_proj.bias"),
                sf.get_tensor(bp + "self_attn.v_proj.bias")]).to(fp16).cuda())
            q, s = to_dev_fp8_kn(g32(bp + "self_attn.o_proj.weight"))
            self._e_o_w.append(q); w_scales_exp.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.gate_proj.weight"))
            self._e_gate_w.append(q); w_scales_exp.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.up_proj.weight"))
            self._e_up_w.append(q); w_scales_exp.append(s)
            q, s = to_dev_fp8_kn(g32(bp + "mlp.down_proj.weight"))
            self._e_down_w.append(q); w_scales_exp.append(s)
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

        # ── heads (fp16 cuBLAS path) ──
        self._ain_w = sf.get_tensor("model.action_in_proj.weight").to(
            torch.float32).t().contiguous().to(fp16).cuda()
        self._ain_b = b16("model.action_in_proj.bias")
        atm_in_w = sf.get_tensor("model.action_time_mlp_in.weight").to(
            torch.float32)                       # [768, 1536]
        self._atm_a_w = atm_in_w[:, :EXP_D].t().contiguous().to(fp16).cuda()
        self._atm_t_w32 = atm_in_w[:, EXP_D:]     # host fp32, for t_contrib
        self._atm_in_b32 = sf.get_tensor(
            "model.action_time_mlp_in.bias").to(torch.float32)
        self._atm_out_w = sf.get_tensor("model.action_time_mlp_out.weight").to(
            torch.float32).t().contiguous().to(fp16).cuda()
        self._atm_out_b = b16("model.action_time_mlp_out.bias")
        self._state_w = sf.get_tensor("model.state_proj.weight").to(
            torch.float32).t().contiguous().to(fp16).cuda()
        self._state_b = b16("model.state_proj.bias")
        dt = -1.0 / STEPS
        self._aout_w_dt = (sf.get_tensor("model.action_out_proj.weight").to(
            torch.float32) * dt).t().contiguous().to(fp16).cuda()
        self._aout_b_dt = (sf.get_tensor("model.action_out_proj.bias").to(
            torch.float32) * dt).to(fp16).cuda()

        # embed_tokens stays on CPU (fp16); gathered per prompt
        self._embed_cpu = sf.get_tensor(
            P + "qwenvl.model.embed_tokens.weight").to(fp16)

        logger.info("Weights loaded + quantized in %.1fs", time.time() - t0)

    # ────────────────────────────────────────────────────────────
    # Fixed buffers + vision tables
    # ────────────────────────────────────────────────────────────

    def _alloc_vit_buffers(self):
        S, D, H = VIS_S, VIS_D, VIS_H_RAW
        self._patches = torch.zeros(S, PATCH_FLAT, dtype=fp16, device="cuda")
        self._vx = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_xn = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_qkv_buf = torch.empty(S, 3 * D, dtype=fp16, device="cuda")
        self._vq = torch.empty(S, D, dtype=fp16, device="cuda")
        self._vk = torch.empty(S, D, dtype=fp16, device="cuda")
        self._vv = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_attn = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_fg = torch.empty(S, D, dtype=fp16, device="cuda")
        self._v_gate = torch.empty(S, H, dtype=fp16, device="cuda")
        self._v_up = torch.empty(S, H, dtype=fp16, device="cuda")
        self._v_hid = torch.empty(S, H, dtype=fp16, device="cuda")
        self._m0_buf = torch.empty(VIS_SM, VIS_DM, dtype=fp16, device="cuda")
        self._vis_emb = torch.empty(VIS_SM, LM_D, dtype=fp16, device="cuda")
        self._v_lse = torch.zeros(12 * VIS_NH * 256, dtype=torch.float32,
                                  device="cuda")
        self._calib_hid = None   # lazily sized at set_prompt (max Se*LM_H)
        self._ones = torch.ones(LM_D, dtype=fp16, device="cuda")

    def _build_vision_tables(self):
        lay = build_vision_layout(num_images=self.num_views)
        self._vis_cos = lay.cos.to(fp16).cuda()
        self._vis_sin = lay.sin.to(fp16).cuda()
        self._patch_perm_cpu = lay.patch_perm
        self._reverse_index = lay.reverse_index.cuda()

    # ────────────────────────────────────────────────────────────
    # set_prompt
    # ────────────────────────────────────────────────────────────

    _PROMPT_ZERO_BUFFERS = (
        "_l_fp8", "_l_lse", "_Kc", "_Vc", "_x_t", "_state_in",
        "_s_fp8", "_s_lse1", "_s_lse2", "_calib_hid",
    )
    _PROMPT_UNIT_BUFFERS = ("_lm_act_scales", "_exp_act_scales")

    def _reset_prompt_buffers(self):
        """Restore initialized contents while preserving every captured pointer."""
        for name in self._PROMPT_ZERO_BUFFERS:
            getattr(self, name).zero_()
        for name in self._PROMPT_UNIT_BUFFERS:
            getattr(self, name).fill_(1.0)
        self._real_data_calibrated = False

    def set_prompt(self, prompt: Union[str, list, np.ndarray]):
        if isinstance(prompt, str):
            token_ids = self._tokenize(prompt)
        else:
            token_ids = np.asarray(prompt, dtype=np.int64)
        n_lang = len(token_ids)
        Se = VIS_SM + n_lang

        # lang embeds: CPU gather → fp16 device (no sqrt scaling)
        emb = self._embed_cpu[torch.from_numpy(token_ids)]
        if (self._prompt_token_count == n_lang
                and self.graph_captured == self.use_cuda_graph):
            # Tables and captured kernel dimensions depend on token count,
            # not token values. Keep their storage and update the graph input.
            self._lang_emb.copy_(emb)
            self._reset_prompt_buffers()
            # Calibration uses native stream zero. The old capture path also
            # completed pending writes before returning from set_prompt.
            torch.cuda.synchronize()
            self.prompt_updates += 1
            self.prompt_buffer_reuses += 1
            logger.info("set_prompt reused buffers (n_lang=%d, Se=%d)", n_lang, Se)
            return

        self._prompt_token_count = None
        self.Se = Se
        self.total_keys = Se + SUF
        self._lang_emb = emb.cuda()

        self._alloc_prompt_buffers(Se)
        self._build_lm_tables(Se)
        self._build_style_tables()
        self._build_attn()
        self._real_data_calibrated = False
        if self.use_cuda_graph:
            self._capture_graphs()
        self.graph_captured = self.use_cuda_graph
        self._prompt_token_count = n_lang
        self.prompt_updates += 1
        logger.info("set_prompt done (n_lang=%d, Se=%d)", n_lang, Se)

    def _tokenize(self, text: str) -> np.ndarray:
        """Replicates lingbotvla prepare_language: '<bos>{t}\\n' via the
        Qwen2.5-VL tokenizer, unpadded (pads are fully masked in the
        reference → dense here)."""
        from transformers import AutoTokenizer
        if not hasattr(self, "_tok"):
            self._tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct")
        t = text if text.startswith("<bos>") else f"<bos>{text}"
        t = t if t.endswith("\n") else f"{t}\n"
        ids = self._tok(t, add_special_tokens=False)["input_ids"]
        return np.asarray(ids, dtype=np.int64)

    def _alloc_prompt_buffers(self, Se):
        D, H = LM_D, LM_H
        self._enc_x = torch.empty(Se, D, dtype=fp16, device="cuda")
        self._l_fp8 = torch.zeros(Se * H, dtype=torch.uint8, device="cuda")
        self._l_qkv_buf = torch.empty(Se, 2560, dtype=fp16, device="cuda")
        self._l_q = torch.empty(Se, 2048, dtype=fp16, device="cuda")
        self._l_attn = torch.empty(Se, 2048, dtype=fp16, device="cuda")
        self._l_fg = torch.empty(Se, D, dtype=fp16, device="cuda")
        self._l_gate = torch.empty(Se, H, dtype=fp16, device="cuda")
        self._l_up = torch.empty(Se, H, dtype=fp16, device="cuda")
        # FA2 lse is (batch, nheads, seqlen_q); pad seq axis to 128 mult
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
        self._s_fp8 = torch.zeros(SUF * EXP_H, dtype=torch.uint8,
                                  device="cuda")
        self._s_gatebuf = torch.empty(SUF, EXP_D, dtype=fp16, device="cuda")
        self._s_qkv = torch.empty(SUF, 2560, dtype=fp16, device="cuda")
        self._s_q = torch.empty(SUF, 2048, dtype=fp16, device="cuda")
        self._s_attn = torch.empty(SUF, 2048, dtype=fp16, device="cuda")
        self._s_fg = torch.empty(SUF, EXP_D, dtype=fp16, device="cuda")
        self._s_gate = torch.empty(SUF, EXP_H, dtype=fp16, device="cuda")
        self._s_up = torch.empty(SUF, EXP_H, dtype=fp16, device="cuda")
        self._s_xn = torch.empty(SUF, EXP_D, dtype=fp16, device="cuda")
        self._s_lse1 = torch.zeros(LM_NH * 128, dtype=torch.float32,
                                   device="cuda")
        self._s_lse2 = torch.zeros(LM_NH * 128, dtype=torch.float32,
                                   device="cuda")
        self._exp_act_scales = torch.full((EXP_L * 4,), 1.0,
                                          dtype=torch.float32, device="cuda")
        # calibration scratch (largest fp16 activation across towers)
        n_scratch = max(Se * H, VIS_S * VIS_H_PAD)
        self._calib_hid = torch.zeros(n_scratch, dtype=fp16, device="cuda")

    def _build_lm_tables(self, Se):
        pos = torch.arange(self.total_keys, dtype=torch.float32)
        cos, sin = build_lm_rope_tables(pos)
        self._pre_cos = cos[:Se].to(fp16).cuda()
        self._pre_sin = sin[:Se].to(fp16).cuda()
        self._suf_cos = cos[Se:Se + SUF].contiguous().to(fp16).cuda()
        self._suf_sin = sin[Se:Se + SUF].contiguous().to(fp16).cuda()

    def _build_style_tables(self):
        """AdaRMS step tables + t_contrib (models/vla4b/step_tables.py)."""
        from flash_rt.models.vla4b.step_tables import build_step_tables
        sa, sf_, tc = build_step_tables(
            self._ada, self._atm_t_w32, self._atm_in_b32, steps=STEPS)
        self._style_attn = sa.cuda()
        self._style_ffn = sf_.cuda()
        self._t_contrib = tc.cuda()

    def _build_attn(self):
        layer_stride = self.total_keys * KV_ROW
        kc, vc = self._Kc.data_ptr(), self._Vc.data_ptr()
        self._attn = Vla4bFa2Attn(
            self._fa2,
            vit={"q": self._vq.data_ptr(), "k": self._vk.data_ptr(),
                 "v": self._vv.data_ptr(), "o": self._v_attn.data_ptr(),
                 "lse": self._v_lse.data_ptr(), "nv": self.num_views},
            lm={"q": self._l_q.data_ptr(), "o": self._l_attn.data_ptr(),
                "lse": self._l_lse.data_ptr(), "Kc": kc, "Vc": vc,
                "layer_stride_elems": layer_stride},
            suffix={"q": self._s_q.data_ptr(), "o": self._s_attn.data_ptr(),
                    "lse1": self._s_lse1.data_ptr(),
                    "lse2": self._s_lse2.data_ptr(),
                    "Kc": kc, "Vc": vc,
                    "layer_stride_elems": layer_stride},
        )

    # ────────────────────────────────────────────────────────────
    # Pipeline dict builders
    # ────────────────────────────────────────────────────────────

    def _vit_args(self):
        bufs = {
            "x": self._vx.data_ptr(), "xn": self._v_xn.data_ptr(),
            "qkv": self._v_qkv_buf.data_ptr(), "q": self._vq.data_ptr(),
            "k": self._vk.data_ptr(), "v": self._vv.data_ptr(),
            "attn_out": self._v_attn.data_ptr(), "fg": self._v_fg.data_ptr(),
            "gate": self._v_gate.data_ptr(), "up": self._v_up.data_ptr(),
            "hid": self._v_hid.data_ptr(),
            "m0": self._m0_buf.data_ptr(), "vis_emb": self._vis_emb.data_ptr(),
            "ones": self._ones.data_ptr(),
        }
        weights = {
            "qkv_w": [w.data_ptr() for w in self._v_qkv_w],
            "qkv_b": [w.data_ptr() for w in self._v_qkv_b],
            "o_w": [w.data_ptr() for w in self._v_o_w],
            "o_b": [w.data_ptr() for w in self._v_o_b],
            "gate_w": [w.data_ptr() for w in self._v_gate_w],
            "gate_b": [w.data_ptr() for w in self._v_gate_b],
            "up_w": [w.data_ptr() for w in self._v_up_w],
            "up_b": [w.data_ptr() for w in self._v_up_b],
            "down_w": [w.data_ptr() for w in self._v_down_w],
            "down_b": [w.data_ptr() for w in self._v_down_b],
            "m0_w": self._m0_w.data_ptr(), "m0_b": self._m0_b.data_ptr(),
            "m2_w": self._m2_w.data_ptr(), "m2_b": self._m2_b.data_ptr(),
            "cos": self._vis_cos.data_ptr(), "sin": self._vis_sin.data_ptr(),
        }
        dims = {"S": VIS_S, "D": VIS_D, "H": VIS_H_RAW, "L": VIS_L,
                "fullatt": VIS_FULLATT, "S_m": VIS_SM, "D_m": VIS_DM,
                "D_enc": LM_D}
        return bufs, weights, dims

    def _lm_args(self):
        bufs = {
            "x": self._enc_x.data_ptr(), "x_fp8": self._l_fp8.data_ptr(),
            "qkv": self._l_qkv_buf.data_ptr(), "q": self._l_q.data_ptr(),
            "attn_out": self._l_attn.data_ptr(), "fg": self._l_fg.data_ptr(),
            "gate": self._l_gate.data_ptr(), "up": self._l_up.data_ptr(),
            "calib_hid": self._calib_hid.data_ptr(),
            "ones": self._ones.data_ptr(),
        }
        weights = {
            "qkv_w": [w.data_ptr() for w in self._l_qkv_w],
            "qkv_b": [w.data_ptr() for w in self._l_qkv_b],
            "o_w": [w.data_ptr() for w in self._l_o_w],
            "gate_w": [w.data_ptr() for w in self._l_gate_w],
            "up_w": [w.data_ptr() for w in self._l_up_w],
            "down_w": [w.data_ptr() for w in self._l_down_w],
            "Kc": self._Kc.data_ptr(), "Vc": self._Vc.data_ptr(),
            "cos": self._pre_cos.data_ptr(), "sin": self._pre_sin.data_ptr(),
            "act_scales": self._lm_act_scales.data_ptr(),
            "w_scales": self._lm_w_scales.data_ptr(),
        }
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
            "gate": self._s_gate.data_ptr(), "up": self._s_up.data_ptr(),
            "xn": self._s_xn.data_ptr(),
            "calib_hid": self._calib_hid.data_ptr(),
            "ones": self._ones.data_ptr(),
        }
        weights = {
            "qkv_w": [w.data_ptr() for w in self._e_qkv_w],
            "qkv_b": [w.data_ptr() for w in self._e_qkv_b],
            "o_w": [w.data_ptr() for w in self._e_o_w],
            "gate_w": [w.data_ptr() for w in self._e_gate_w],
            "up_w": [w.data_ptr() for w in self._e_up_w],
            "down_w": [w.data_ptr() for w in self._e_down_w],
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
        dims = {"D": EXP_D, "H": EXP_H, "L": EXP_L, "steps": STEPS,
                "Se": self.Se, "total_keys": self.total_keys,
                "kv_row": KV_ROW}
        return bufs, weights, dims

    # ────────────────────────────────────────────────────────────
    # Forward pieces (shared by eager + capture)
    # ────────────────────────────────────────────────────────────

    def _run_vis(self, stream_int):
        # patch embed: [768, 1176] @ [1176, 1280] (fp16 cuBLAS)
        fvk.gmm_fp16(self._ctx, self._patches.data_ptr(),
                     self._pe_w.data_ptr(), self._vx.data_ptr(),
                     VIS_S, VIS_D, PATCH_FLAT, 0.0, stream_int)
        b, w, d = self._vit_args()
        vit_forward(fvk, self._ctx, b, w, d, stream_int, attn=self._attn)

    def _run_lm_expert(self, stream_int, calibrate=False):
        # stage enc_x: vision rows (un-permute merger output) + lang rows
        torch.index_select(self._vis_emb, 0, self._reverse_index,
                           out=self._enc_x[:VIS_SM])
        self._enc_x[VIS_SM:self.Se].copy_(self._lang_emb)
        b, w, d = self._lm_args()
        lm_prefill_forward(fvk, b, w, d, stream_int, attn=self._attn,
                           calibrate=calibrate)
        # state embed (constant per infer): [1,75] @ [75,768] + bias
        fvk.gmm_fp16(self._ctx, self._state_in.data_ptr(),
                     self._state_w.data_ptr(), self._state_emb.data_ptr(),
                     1, EXP_D, SDIM, 0.0, stream_int)
        fvk.add_bias_fp16(self._state_emb.data_ptr(),
                          self._state_b.data_ptr(), 1, EXP_D, stream_int)
        b, w, d = self._expert_args()
        expert_forward(fvk, self._ctx, b, w, d, stream_int, attn=self._attn,
                       calibrate=calibrate)

    # ────────────────────────────────────────────────────────────
    # Graph capture
    # ────────────────────────────────────────────────────────────

    def _capture_graphs(self):
        torch.cuda.synchronize()
        for _ in range(2):  # warmup
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
        """pixel_patches: (3, 256, 1176) in the HF Qwen2.5-VL pixel_values
        layout (pre-normalized). state: (75,). noise: (50, 75)."""
        p = torch.as_tensor(pixel_patches).reshape(VIS_S, PATCH_FLAT)
        if p.is_cuda:
            self._patches.copy_(p[self._patch_perm_cpu.cuda()].to(fp16))
        else:
            self._patches.copy_(p[self._patch_perm_cpu].to(fp16))
        self._state_in.copy_(torch.as_tensor(state).reshape(1, SDIM).to(fp16))
        self._noise_host = torch.as_tensor(noise).reshape(CHUNK, ADIM).to(fp16)
        self._x_t.copy_(self._noise_host)

    def infer_staged(self, pixel_patches, state, noise, *, vision_embeddings=None):
        """Run with explicit noise and optional native BF16 vision output.

        ``vision_embeddings`` has shape (192, 2048), in original merged-token
        order. The caller computes it from the current cameras on every chunk;
        this method does not cache observations or silently reuse visual features.
        """
        t0 = time.perf_counter()
        self.stage_inputs(pixel_patches, state, noise)
        external_vision = vision_embeddings is not None
        if external_vision:
            # Native Qwen vision returns merged tokens in original image order;
            # _run_lm_expert expects the engine's window order before unpermuting.
            visual = torch.as_tensor(vision_embeddings, device=self._vis_emb.device)
            if tuple(visual.shape) != (VIS_SM, LM_D):
                raise ValueError("VLA4 vision embeddings must have shape (192, 2048)")
            self._vis_emb.copy_(visual[torch.argsort(self._reverse_index)].to(fp16))
        else:
            if self.graph_captured:
                self._g_vis.replay()
            else:
                self._run_vis(0)
        if not bool(torch.isfinite(self._vis_emb).all()):
            raise RuntimeError("VLA4 vision produced nonfinite features; use the native BF16 vision path")
        if not self._real_data_calibrated:
            self._run_lm_expert(0, calibrate=True)
            torch.cuda.synchronize()
            self._x_t.copy_(self._noise_host)  # calib consumed x_t
            self._real_data_calibrated = True
            logger.info("Real-data FP8 calibration done (no recapture)")
        if self.graph_captured:
            self._g_lm.replay()
        else:
            self._run_lm_expert(0)
        torch.cuda.synchronize()
        actions = self._x_t.float().cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latency_records.append(latency_ms)
        return {"actions": actions, "latency_ms": latency_ms}

    def get_latency_stats(self):
        arr = np.asarray(self.latency_records[1:] or self.latency_records)
        return {"p50": float(np.percentile(arr, 50)),
                "p99": float(np.percentile(arr, 99)),
                "mean": float(arr.mean()), "n": int(arr.size)}
