"""FlashRT — WanVaTorchFrontendThor: LingBot-VA on Thor SM110, built FOR a declared operating point.

Structure cloned from ``vla2_thor.py`` (the T2-V2 frontend), with wan_va shapes from the real
checkpoint (``/home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin/transformer``, 3 shards,
BF16, 841 tensors, 5.089B) and the real serving source (``/home/ubuntu/lingbot-va/wan_va``).
Mapping + gates: ``/home/ubuntu/iwm_distill/thor_va_engine/{mapping_memo,repack_calib_plan}.md``;
the operating-point requirement: ``iwm_distill/fewstep/h2_thor_realtime_design.md`` §5 and
``instinctflash/runtime/engine_backend.py`` (9bf2337).

STAGE 2 (2026-09-02). The engine owns every DiT forward of a control cycle and the KV slab;
torch owns T5 (episode scope), the VAE encoders (kv messages), the wire protocol and action
pre/post-processing — the Stage-A hybrid the mapping memo §B chose. The build takes a
``WanVaOperatingPoint`` and derives from it: the per-stream AdaLN step tables (t-grids), the
KV slab stream count (cfg_batch), whether a CFG combine exists, and the forward count per
cycle. ``assert_point`` refuses any other request (fail-closed; the engine tier honors the
operating point or declines).

Key facts this file encodes (all source-verified, memo §0):
    * 30 blocks x 3072 (24h x hd128), ffn 14336; B = cfg_batch streams (2 at w5: the two
      streams' K/V genuinely differ; 1 at w1); (V+1)+(A+1)+2 DiT forwards per cycle.
    * rope: f64 bases 44/42/42, complex-pair interleaved -> P-permutation repack makes
      ``rope_rotate_half_fp16`` exact; action tokens at fractional frames f + k/17, h=w=-1.
    * modulation: AdaLN step tables through the deployed dtype chain; video stream uses
      condition_embedder, action stream the _action copy; cycle-0 forwards are two-span
      (frame 0 at t=0).
    * text: 512 rows fixed, zero-pad rows LIVE; cross K post-norm_k, V raw; built in bf16 with
      the stock ops then cast to fp16 (exactly representable below 65504 — range-checked).
    * KV: linear episode slab [L, B, slab_rows, D] fp16 x (K, V); head/tail host ints
      (``models/wan_va/ring_ref.LinearSlabRef``, self-checked vs the stock allocator) — never
      inside a captured region.
    * the tiny tail (CFG combine, Euler step, patch permutes, cycle-0 pins, action channel
      mask) runs torch-side in bf16 with the stock expressions (exact dtype chain); the
      input embedding (patch_embedding_mlp / action_embedder) runs torch-side in bf16 exactly
      as stock, then casts to fp16 into the engine's residual buffer.
    * DETERMINISTIC: no atomics anywhere (FA2 single-launch, cuBLAS fixed shapes) — a matched
      (prompt, latents, noise) episode replays bit-for-bit; ``digest`` records it.
"""

import hashlib
import json
import logging
import os
import pathlib
import time
from typing import Optional, Union

import torch
import torch.nn.functional as F

if os.environ.get("WAN_VA_TORCH_REF_ONLY") == "1":
    fvk = None                       # CPU-only self-checks: never touch a GPU
else:
    try:  # import-clean on hosts without the built extension
        import flash_rt.flash_rt_kernels as fvk
    except Exception:  # pragma: no cover - exercised on CPU-only hosts
        fvk = None

from flash_rt.models.wan_va import (
    ACTION_DIM,
    ACTION_PER_FRAME,
    ACTION_TOKENS,
    DIT_D,
    DIT_FFN,
    DIT_L,
    DIT_NH,
    FRAME_CHUNK,
    LATENT_C,
    LATENT_H,
    LATENT_W,
    MAX_VIDEO_TOKENS,
    POOL_SLOTS,
    SLAB_ROWS,
    TEXT_DIM,
    TEXT_LEN,
    VIDEO_TOKENS,
)
from flash_rt.models.wan_va.geometry import WanVaGeometry
from flash_rt.models.wan_va.operating_point import (
    OperatingPointMismatch,
    POINT_2V4A_W5,
    WanVaOperatingPoint,
)
from flash_rt.models.wan_va.pipeline_thor import (
    ACT_SLOTS,
    W_SLOTS,
    WanFa2Attn,
    dit_forward,
    launches_per_forward,
    make_gate_fn,
)
from flash_rt.models.wan_va.ring_ref import LinearSlabRef
from flash_rt.models.wan_va.rope_table import (
    action_grid,
    build_cos_sin,
    permute_qk_rows,
    video_grid,
)
from flash_rt.models.wan_va.step_tables import (
    EmbedderWeights,
    build_stream_tables,
    flow_match_sigmas,
)
from flash_rt.models.wan_va.wan_ref import _rms

logger = logging.getLogger(__name__)

fp16 = torch.float16
bf16 = torch.bfloat16
fp8 = torch.float8_e4m3fn

#: va_robotwin_cfg.used_action_channel_ids — the 16 live channels of the 30-dim action state.
ROBOTWIN_USED_ACTION_CHANNELS = list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
POST_PATCH_H, POST_PATCH_W = LATENT_H // 2, LATENT_W // 2      # 12, 10
TOKENS_PER_FRAME = POST_PATCH_H * POST_PATCH_W                  # 120
PATCH_FLAT = LATENT_C * 4                                       # 192

# ══════════════════════════════════════════════════════════════════
# Stage-A conditioning handoff contract (keep in sync with mapping_memo.md §B).
# ══════════════════════════════════════════════════════════════════
CONDITIONING_HANDOFF_CONTRACT = """
Shapes below describe the legacy RoboTwin build. Explicit WanVaGeometry replaces
  F, latent H/W and actions/frame throughout. LIBERO uses F=4, H/W=8/16, actions/frame=4.
  Schedule shifts belong to WanVaOperatingPoint and are checked by assert_point.
Build: WanVaTorchFrontendThor(ckpt, point=WanVaOperatingPoint(...)). The build IS the
  operating point (per-stream grids -> step tables; guidance -> cfg_batch B, CFG combine;
  forwards/cycle). assert_point(requested) refuses any other point (OperatingPointMismatch).
set_prompt(text_emb): bf16 (B, 512, 4096) — row 0 positive (live T5 on prompt_clean(prompt),
  zero-padded to 512), row 1 (B=2 only) the negative "" embedding. ALL 512 rows live.
  Engine builds text_hidden + per-layer cross K/V once per episode (bf16 stock ops -> fp16).
reset_episode(): slab head=tail=0; pred marks cleared; frame_st_id=0; graphs survive
  (pointer-stable slabs/buffers; graph keys carry head/tail).
commit_chunk(video_latents bf16 (1, 48, F, 24, 20), action_state bf16 (1, 30, Fa, 16, 1)):
  clear_pred, then the 2 kv-commit forwards (t=0 rows, kind 'commit'); frame_st_id += F
  (F=3 at cycle 0 — init keyframe concat — else 2; Fa = 2).
infer_cycle(noise_v bf16 (1, 48, 2, 24, 20), noise_a bf16 (1, 30, 2, 16, 1),
  init_latent bf16 (1, 48, 1, 24, 20) at cycle 0 else None): V video forwards + 1 pred
  commit, then A action forwards + 1 pred commit, at the BUILD's grids; CFG combine on the
  video stream iff the build serves cfg@w>1; Euler state bf16 with the stock expression;
  returns (actions bf16 (1, 30, 2, 16, 1) normalized — the client de-normalizes ONCE —,
  latents bf16 (1, 48, 2, 24, 20)). Noise is handed off in LATENT layout.
Cycle-0 specials (engine-side): conditioning-frame pin + two-span t=0 modulation for
  frame 0 (both streams); action frame 0 zeroed; unused action channels zeroed every forward.
FAIL-LOUD: operating-point mismatch, slab overflow (> slab_rows committed rows), fp16 range
  (|x| >= 65504 in cross-KV / embeddings), message order (infer before set_prompt, frame_st_id
  drift), clear_pred on a non-suffix.
"""


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                          if t.dtype in (bf16, fp16) else
                          t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def _quantize_fp8(w32: torch.Tensor) -> tuple:
    """Per-tensor symmetric E4M3: amax/448 (pi05/vla4b/vla2 recipe)."""
    amax = w32.abs().max().item()
    scale = max(amax / 448.0, 1e-12)
    w_fp8 = (w32 / scale).clamp(-448.0, 448.0).to(fp8)
    return w_fp8, float(scale)


class _ShardedCkpt:
    """Multi-shard safetensors reader (diffusers index name)."""

    INDEX = "diffusion_pytorch_model.safetensors.index.json"

    def __init__(self, ckpt_dir: pathlib.Path):
        from safetensors import safe_open
        idx = json.loads((ckpt_dir / self.INDEX).read_text())
        self._key2file = idx["weight_map"]
        self._files = {
            fn: safe_open(str(ckpt_dir / fn), framework="pt", device="cpu")
            for fn in sorted(set(self._key2file.values()))
        }

    def get(self, key: str) -> torch.Tensor:
        return self._files[self._key2file[key]].get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self._key2file


# ── torch-side layout helpers (stock einops expressions, dependency-free) ──

def patchify(latents: torch.Tensor) -> torch.Tensor:
    """'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)', patch (1, 2, 2)."""
    B, C, Fr, H, W = latents.shape
    x = latents.reshape(B, C, Fr, 1, H // 2, 2, W // 2, 2)
    return x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, Fr * (H // 2) * (W // 2), C * 4)


def unpatchify(seq: torch.Tensor, Fr: int, latent_h: int = LATENT_H,
               latent_w: int = LATENT_W) -> torch.Tensor:
    """proj_out output (B, S, 192=(n c)) -> 'b l (n c) -> b (l n) c' -> data_seq_to_patch
    -> (B, 48, F, 24, 20). n = (p_t p_h p_w) outer, c inner (model:876-882 + utils:12-30)."""
    B = seq.shape[0]
    x = seq.reshape(B, Fr, latent_h // 2, latent_w // 2, 1, 2, 2, LATENT_C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(B, LATENT_C, Fr, latent_h, latent_w)


def actions_to_tokens(actions: torch.Tensor) -> torch.Tensor:
    """'b c f h w -> b (f h w) c'."""
    B, C, Fr, H, W = actions.shape
    return actions.permute(0, 2, 3, 4, 1).reshape(B, Fr * H * W, C)


def tokens_to_actions(seq: torch.Tensor, Fr: int) -> torch.Tensor:
    """'b (f n) c -> b c f n 1'."""
    B, S, C = seq.shape
    return seq.reshape(B, Fr, S // Fr, C).permute(0, 3, 1, 2).unsqueeze(-1)


class WanVaTorchFrontendThor:
    """LingBot-VA DiT inference using only flash_rt kernels + FA2, built for ONE point."""

    def __init__(self, checkpoint_dir: Union[str, pathlib.Path],
                 point: WanVaOperatingPoint = POINT_2V4A_W5,
                 precision: str = "fp8", device: str = "cuda",
                 use_cuda_graph: bool = False, fa2_lib=None, num_sms: int = 0,
                 slab_rows: int = SLAB_ROWS, pool_slots: int = POOL_SLOTS,
                 used_action_channels=None, fp16_families=(),
                 action_terminal_elision: bool = True,
                 geometry: WanVaGeometry = WanVaGeometry()):
        self.torch_ref_only = fvk is None
        if self.torch_ref_only and use_cuda_graph:
            raise RuntimeError(
                "flash_rt_kernels extension not available — only repack/tables/slab "
                "bookkeeping can run on this host")
        assert precision in ("fp8", "fp16")
        if not isinstance(point, WanVaOperatingPoint):
            raise TypeError("point must be a WanVaOperatingPoint (the build IS the point)")
        if not isinstance(geometry, WanVaGeometry):
            raise TypeError("geometry must be WanVaGeometry")
        self.geometry = geometry
        self.point = point
        self.B = point.cfg_batch
        self.precision = precision
        #: GEMM families kept fp16 inside the fp8 arm (memo §C pre-committed per-family fallback):
        #: any of qkv_w, o_w, cq_w, co_w, ff1_w, ff2_w
        self.fp16_families = tuple(sorted(set(fp16_families))) if precision == "fp8" else ()
        for f_ in self.fp16_families:
            assert f_ in ("qkv_w", "o_w", "cq_w", "co_w", "ff1_w", "ff2_w"), f_
        #: P010 (d7e6103, BITEXACT through the wrap, shipped_configuration): skip the action
        #: pred-commit forward, replay only its slab allocation. Default ON = the served chain.
        self.action_terminal_elision = bool(action_terminal_elision)
        self.device = device
        self.use_cuda_graph = bool(use_cuda_graph)
        self.num_sms = int(num_sms)
        self.slab_rows = int(slab_rows)
        self.pool_slots = int(pool_slots)
        self.calibrating = False
        self.profile = False
        self.stage_ms = {}
        self.forward_log = []
        self.latency_records = []
        self._graphs = {}
        self._graph_seen = {}
        self._warm_shapes = set()
        # M2 broken-control hooks (parity harness only; never set in serving)
        self.debug_force_t0 = False      # every span uses the t=0 table row
        self.debug_cross_len = None      # attend only the first N text rows
        self._ctx = fvk.FvkContext() if fvk is not None else None
        if fa2_lib is None and fvk is not None:
            from flash_rt import flash_rt_fa2 as fa2_lib
        self._fa2 = fa2_lib
        used = list(ROBOTWIN_USED_ACTION_CHANNELS if used_action_channels is None
                    else used_action_channels)
        mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        mask[used] = True
        self._action_unused = ~mask                      # host bool [30]

        ckpt = pathlib.Path(checkpoint_dir)
        if not (ckpt / _ShardedCkpt.INDEX).exists():
            raise FileNotFoundError(ckpt / _ShardedCkpt.INDEX)
        self._load_weights(ckpt)
        self._build_step_tables()
        self._alloc_slabs()
        self._alloc_buffers()
        self._build_attn()
        self.frame_st_id = 0
        self._prompt_set = False
        self._next_tok = 0
        self.slab = LinearSlabRef(capacity=self.pool_slots, slab_rows=self.slab_rows)
        logger.info("WanVaTorchFrontendThor built for %s (%s, B=%d, %d fwd/cycle, gate=%s)",
                    point.served_point(), precision, self.B, point.forwards_per_cycle,
                    getattr(self._gate_fn, "kind", "n/a"))

    # ────────────────────────────────────────────────────────────
    # Declaration surfaces (the 9bf2337 discipline)
    # ────────────────────────────────────────────────────────────

    def served_operating_point(self) -> WanVaOperatingPoint:
        return self.point

    def declaration(self) -> dict:
        d = self.point.declaration()
        d["geometry"] = dict(vars(self.geometry))
        d["pool_slots"] = self.pool_slots
        d["precision"] = self.precision
        d["fp16_families"] = list(self.fp16_families)
        d["action_terminal_elision"] = self.action_terminal_elision
        d["forwards_per_cycle_served"] = self.point.forwards_per_cycle_served(self.action_terminal_elision)
        d["slab_rows"] = self.slab_rows
        d["gate_kernel"] = getattr(self._gate_fn, "kind", "torch_ref_only")
        return d

    def assert_point(self, requested: WanVaOperatingPoint, *, requester="the request"):
        """Refuse to serve any point other than the build's (OperatingPointMismatch)."""
        self.point.assert_serves(requested, requester=requester)

    # ────────────────────────────────────────────────────────────
    # Weight loading (BF16 ckpt → fp8/fp16 device tensors + hosts)
    # ────────────────────────────────────────────────────────────

    def _load_weights(self, ckpt_dir: pathlib.Path):
        t0 = time.time()
        sf = _ShardedCkpt(ckpt_dir)
        dev = self.device if not self.torch_ref_only else "cpu"

        def g32(key):
            return sf.get(key).to(torch.float32)

        def d16(t):
            return t.to(fp16).to(dev)

        def dbf(key):
            return sf.get(key).to(bf16).to(dev)

        def to_kn_fp16(w32_nk):
            return w32_nk.t().contiguous().to(fp16).to(dev)

        def to_kn_fp8(w32_nk):
            return _quantize_fp8(w32_nk.t().contiguous().to(dev))

        assert sf.has("patch_embedding.weight"), \
            "expected the vestigial conv keys (loader sanity anchor)"
        # NOT loaded: patch_embedding.{weight,bias} — dead conv the architecture replaced
        # with patch_embedding_mlp (RESULTS.md §1).

        w_scales = []
        B = "blocks.{}."
        self._blk = {k: [] for k in (
            "qkv_w", "qkv_b", "nq_w", "nk_w", "o_w", "o_b",
            "cq_w", "cq_b", "cnq_w", "co_w", "co_b",
            "n2_w", "n2_b", "ff1_w", "ff1_b", "ff2_w", "ff2_b")}
        # cross to_k/to_v + norm_k: episode-scope bf16 device copies (set_prompt builds
        # the cross-KV with the STOCK bf16 ops, then casts to fp16 — memo §0.9 semantics)
        self._cross = {k: [] for k in ("ck_w", "ck_b", "cv_w", "cv_b", "cnk_w")}
        self._sst_host = []          # per-block scale_shift_table fp32 hosts

        for l in range(DIT_L):
            bp = B.format(l)
            qw = permute_qk_rows(g32(bp + "attn1.to_q.weight"), DIT_NH)
            kw = permute_qk_rows(g32(bp + "attn1.to_k.weight"), DIT_NH)
            vw = g32(bp + "attn1.to_v.weight")
            qkv = torch.cat([qw, kw, vw], dim=0)          # [9216, 3072]
            qb = permute_qk_rows(g32(bp + "attn1.to_q.bias"), DIT_NH)
            kb = permute_qk_rows(g32(bp + "attn1.to_k.bias"), DIT_NH)
            vb = g32(bp + "attn1.to_v.bias")
            self._blk["qkv_b"].append(d16(torch.cat([qb, kb, vb])))
            self._blk["nq_w"].append(d16(permute_qk_rows(
                g32(bp + "attn1.norm_q.weight"), DIT_NH)))
            self._blk["nk_w"].append(d16(permute_qk_rows(
                g32(bp + "attn1.norm_k.weight"), DIT_NH)))

            fp8_sites = [
                ("qkv_w", qkv),
                ("o_w", g32(bp + "attn1.to_out.0.weight")),
                ("cq_w", g32(bp + "attn2.to_q.weight")),
                ("co_w", g32(bp + "attn2.to_out.0.weight")),
                ("ff1_w", g32(bp + "ffn.net.0.proj.weight")),
                ("ff2_w", g32(bp + "ffn.net.2.weight")),
            ]
            for name, w in fp8_sites:
                if self.precision == "fp8" and name not in self.fp16_families:
                    q, s = to_kn_fp8(w)
                    self._blk[name].append(q)
                    w_scales.append(s)
                else:
                    self._blk[name].append(to_kn_fp16(w))
                    if self.precision == "fp8":
                        w_scales.append(1.0)     # slot unused (fp16 family), keeps indexing uniform
            self._blk["o_b"].append(d16(g32(bp + "attn1.to_out.0.bias")))
            self._blk["cq_b"].append(d16(g32(bp + "attn2.to_q.bias")))
            self._blk["cnq_w"].append(d16(g32(bp + "attn2.norm_q.weight")))
            self._blk["co_b"].append(d16(g32(bp + "attn2.to_out.0.bias")))
            self._blk["n2_w"].append(d16(g32(bp + "norm2.weight")))
            self._blk["n2_b"].append(d16(g32(bp + "norm2.bias")))
            self._blk["ff1_b"].append(d16(g32(bp + "ffn.net.0.proj.bias")))
            self._blk["ff2_b"].append(d16(g32(bp + "ffn.net.2.bias")))

            for hk, ck in (("ck_w", "attn2.to_k.weight"),
                           ("ck_b", "attn2.to_k.bias"),
                           ("cv_w", "attn2.to_v.weight"),
                           ("cv_b", "attn2.to_v.bias"),
                           ("cnk_w", "attn2.norm_k.weight")):
                self._cross[hk].append(dbf(bp + ck))
            self._sst_host.append(g32(bp + "scale_shift_table"))

        assert len(w_scales) in (0, DIT_L * W_SLOTS)
        self._w_scales = (torch.tensor(w_scales, dtype=torch.float32, device=dev)
                          if w_scales else torch.zeros(1, dtype=torch.float32, device=dev))

        # heads (engine fp16 cuBLAS path)
        self._proj_w = to_kn_fp16(g32("proj_out.weight"))          # [3072, 192]
        self._proj_b = d16(g32("proj_out.bias"))
        self._aproj_w = to_kn_fp16(g32("action_proj_out.weight"))  # [3072, 30]
        self._aproj_b = d16(g32("action_proj_out.bias"))
        # input embedders — torch-side bf16, EXACTLY the stock ops (then cast to fp16)
        self._patch_w = dbf("patch_embedding_mlp.weight")          # [3072, 192]
        self._patch_b = dbf("patch_embedding_mlp.bias")
        self._act_w = dbf("action_embedder.weight")                # [3072, 30]
        self._act_b = dbf("action_embedder.bias")
        # text_embedder — the VIDEO copy only (the _action copy's text_embedder is dead at
        # inference, model.py:843); bf16 stock ops at set_prompt
        tp = "condition_embedder.text_embedder."
        self._text1_w = dbf(tp + "linear_1.weight"); self._text1_b = dbf(tp + "linear_1.bias")
        self._text2_w = dbf(tp + "linear_2.weight"); self._text2_b = dbf(tp + "linear_2.bias")
        # time chains → step-table build inputs (host)
        self._emb_video = self._embedder_host(sf, "condition_embedder.")
        self._emb_action = self._embedder_host(sf, "condition_embedder_action.")
        self._root_sst = g32("scale_shift_table")        # [1, 2, 3072]
        logger.info("Weights loaded + repacked in %.1fs (%s)", time.time() - t0, self.precision)

    @staticmethod
    def _embedder_host(sf: _ShardedCkpt, prefix: str) -> EmbedderWeights:
        g = lambda k: sf.get(prefix + k).to(torch.float32)  # noqa: E731
        return EmbedderWeights(
            g("time_embedder.linear_1.weight"), g("time_embedder.linear_1.bias"),
            g("time_embedder.linear_2.weight"), g("time_embedder.linear_2.bias"),
            g("time_proj.weight"), g("time_proj.bias"))

    # ────────────────────────────────────────────────────────────
    # Tables, slabs, buffers
    # ────────────────────────────────────────────────────────────

    def _build_step_tables(self):
        """Per-stream AdaLN tables at THE BUILD'S grids (the point is the table)."""
        self._t_values = {s: self.point.t_values(s) for s in ("video", "action")}
        self._tables = {
            "video": build_stream_tables(self._emb_video, self._sst_host, self._root_sst,
                                         self._t_values["video"]),
            "action": build_stream_tables(self._emb_action, self._sst_host, self._root_sst,
                                          self._t_values["action"]),
        }
        self._sigmas = {"video": flow_match_sigmas(self.point.video_steps, self.point.video_shift),
                        "action": flow_match_sigmas(self.point.action_steps, self.point.action_shift)}
        if not self.torch_ref_only:
            for s in ("video", "action"):
                self._tables[s]["mod"] = self._tables[s]["mod"].to(self.device).contiguous()
                self._tables[s]["out"] = self._tables[s]["out"].to(self.device).contiguous()

    def _alloc_slabs(self):
        """Self-KV linear episode slab [L, B, slab_rows, D] fp16 x (K, V) (11.3 GB at B=2,
        5.6 GB at B=1) + cross-KV slab [L, B, 512, D]. Host bookkeeping lives in
        LinearSlabRef — never inside a captured region."""
        if self.torch_ref_only:
            self._Kc = self._Vc = self._cK = self._cV = None
            return
        shape = (DIT_L, self.B, self.slab_rows, DIT_D)
        self._Kc = torch.zeros(shape, dtype=fp16, device=self.device)
        self._Vc = torch.zeros(shape, dtype=fp16, device=self.device)
        self._cK = torch.zeros(DIT_L, self.B, TEXT_LEN, DIT_D, dtype=fp16, device=self.device)
        self._cV = torch.zeros_like(self._cK)

    def _alloc_buffers(self):
        if self.torch_ref_only:
            self._gate_fn = None
            return
        dev = self.device
        Rmax = self.B * self.geometry.max_tokens
        self._Rmax = Rmax
        e = lambda *shape: torch.empty(*shape, dtype=fp16, device=dev)  # noqa: E731
        self._x = e(Rmax, DIT_D); self._xn = e(Rmax, DIT_D); self._fg = e(Rmax, DIT_D)
        self._qraw = e(Rmax, DIT_D); self._q = e(Rmax, DIT_D); self._kt = e(Rmax, DIT_D)
        self._attn_out = e(Rmax, DIT_D)
        self._qkv = e(Rmax, 3 * DIT_D)
        self._hid = e(Rmax, DIT_FFN)
        self._x_fp8 = torch.zeros(Rmax * DIT_FFN, dtype=torch.uint8, device=dev)
        self._head_out = e(Rmax, PATCH_FLAT)
        self._gate_bcast = e(Rmax, DIT_D)
        lse_rows = max(512, (self.geometry.max_tokens + 127) // 128 * 128)
        self._lse = torch.zeros(self.B * DIT_NH * lse_rows, dtype=torch.float32, device=dev)
        self._cos = {s: torch.zeros(self.geometry.max_tokens, 128, dtype=fp16, device=dev)
                     for s in ("video", "action")}
        self._sin = {s: torch.zeros(self.geometry.max_tokens, 128, dtype=fp16, device=dev)
                     for s in ("video", "action")}
        self._act_scales = torch.full((DIT_L * ACT_SLOTS,), 1.0, dtype=torch.float32,
                                      device=dev)
        self._act_amax_running = torch.zeros_like(self._act_scales)
        self._gate_fn = make_gate_fn(fvk, self._gate_bcast.data_ptr())
        self._rope_cache = {}

    def _build_attn(self):
        if self.torch_ref_only:
            self._attn = None
            return
        self._attn = WanFa2Attn(
            self._fa2, q=self._q.data_ptr(), o=self._attn_out.data_ptr(),
            lse=self._lse.data_ptr(), Kc=self._Kc.data_ptr(), Vc=self._Vc.data_ptr(),
            cK=self._cK.data_ptr(), cV=self._cV.data_ptr(), B=self.B,
            slab_rows=self.slab_rows, num_sms=self.num_sms)
        self._bufs = {
            "x": self._x.data_ptr(), "xn": self._xn.data_ptr(), "qkv": self._qkv.data_ptr(),
            "qraw": self._qraw.data_ptr(), "q": self._q.data_ptr(), "kt": self._kt.data_ptr(),
            "attn_out": self._attn_out.data_ptr(), "fg": self._fg.data_ptr(),
            "hid": self._hid.data_ptr(), "x_fp8": self._x_fp8.data_ptr(),
            "head_out": self._head_out.data_ptr(),
        }
        common = {k: [w.data_ptr() for w in v] for k, v in self._blk.items()}
        common.update({
            "act_scales": self._act_scales.data_ptr(), "w_scales": self._w_scales.data_ptr(),
            "Kc": self._Kc.data_ptr(), "Vc": self._Vc.data_ptr(),
            "fp16_families": self.fp16_families,
        })
        self._weights = {}
        for s, (pw, pb) in (("video", (self._proj_w, self._proj_b)),
                            ("action", (self._aproj_w, self._aproj_b))):
            d = dict(common)
            d.update({"mod": self._tables[s]["mod"].data_ptr(),
                      "out_tab": self._tables[s]["out"].data_ptr(),
                      "cos": self._cos[s].data_ptr(), "sin": self._sin[s].data_ptr(),
                      "proj_w": pw.data_ptr(), "proj_b": pb.data_ptr()})
            self._weights[s] = d
        self._out_dim = {"video": PATCH_FLAT, "action": ACTION_DIM}

    def _rope_tables(self, stream: str, frame_st_id: int, num_frames: int):
        """(cos, sin) (S, 128) fp16 for this (stream, frame_st_id, F); cached host→device."""
        key = (stream, frame_st_id, num_frames)
        t = self._rope_cache.get(key)
        if t is None:
            g = (video_grid(num_frames, self.geometry.latent_h // 2,
                            self.geometry.latent_w // 2, frame_st_id)
                 if stream == "video" else
                 action_grid(num_frames, self.geometry.action_per_frame, frame_st_id))
            cos, sin = build_cos_sin(g)
            t = (cos.to(self.device), sin.to(self.device))
            self._rope_cache[key] = t
        return t

    # ────────────────────────────────────────────────────────────
    # Episode lifecycle (Stage-A seam)
    # ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def set_prompt(self, text_emb: torch.Tensor):
        """text_emb (B, 512, 4096) bf16 — see CONDITIONING_HANDOFF_CONTRACT."""
        if text_emb.shape != (self.B, TEXT_LEN, TEXT_DIM):
            raise OperatingPointMismatch(
                f"set_prompt: text_emb {tuple(text_emb.shape)} does not match the build's "
                f"cfg_batch={self.B} (expected ({self.B}, {TEXT_LEN}, {TEXT_DIM})); a "
                f"negative row is only served by a cfg@w>1 build")
        if self.torch_ref_only:
            self._text_emb_host = text_emb
            self._prompt_set = True
            return
        te = text_emb.to(self.device, bf16)
        # stock text_embedder (PixArtAlphaTextProjection, bf16): linear_1 -> gelu(tanh) -> linear_2
        th = F.linear(F.gelu(F.linear(te, self._text1_w, self._text1_b), approximate="tanh"),
                      self._text2_w, self._text2_b)
        amax = 0.0
        for l in range(DIT_L):
            ck = _rms(F.linear(th, self._cross["ck_w"][l], self._cross["ck_b"][l]),
                      self._cross["cnk_w"][l])
            cv = F.linear(th, self._cross["cv_w"][l], self._cross["cv_b"][l])
            amax = max(amax, ck.abs().max().item(), cv.abs().max().item())
            self._cK[l].copy_(ck.to(fp16))
            self._cV[l].copy_(cv.to(fp16))
        if amax >= 65504.0:
            raise RuntimeError(f"cross-KV amax {amax} exceeds the fp16 range — fail-loud")
        self._prompt_set = True
        self.reset_episode()
        if self.use_cuda_graph and not self._warm_shapes:
            self.warmup()
        logger.info("set_prompt: cross-KV slab built (%d layers, B=%d, amax %.1f)",
                    DIT_L, self.B, amax)

    @torch.no_grad()
    def warmup(self):
        """Run one eager forward per (stream, S) shape the point can see (video 240/360, action
        32) so cuBLASLt descriptors exist before any capture; leaves the slab empty. Call after
        set_prompt, before serving (the probe protocol's run 0 also covers it, but a served
        episode must never pay a cold cuBLASLt path inside a capture)."""
        assert self._prompt_set and not self.torch_ref_only
        saved = (self.use_cuda_graph, self.frame_st_id, self.calibrating)
        self.use_cuda_graph = False
        self.calibrating = False     # warm the PRODUCTION program (static quantize path), never the calibrate path
        try:
            self.reset_episode()
            for S, stream, F_ in ((self.geometry.max_video_tokens, "video", self.geometry.frame_chunk + 1),
                                  (self.geometry.video_tokens, "video", self.geometry.frame_chunk),
                                  (self.geometry.action_tokens, "action", self.geometry.frame_chunk)):
                tok = torch.zeros(S, PATCH_FLAT if stream == "video" else ACTION_DIM,
                                  dtype=bf16, device=self.device)
                self._forward(stream, tok, num_frames=F_, spans=((0, S, 0),), kind="transient")
                self._warm_shapes.add((stream, S))
            torch.cuda.synchronize()
        finally:
            self.use_cuda_graph, self.frame_st_id, self.calibrating = saved
            self.reset_episode()

    def reset_episode(self):
        self.slab = LinearSlabRef(capacity=self.pool_slots, slab_rows=self.slab_rows)
        self.frame_st_id = 0
        self._next_tok = 0
        self.forward_log = []

    # ────────────────────────────────────────────────────────────
    # One DiT forward (host bookkeeping + the device program)
    # ────────────────────────────────────────────────────────────

    def _embed(self, stream: str, tokens_bf16: torch.Tensor) -> torch.Tensor:
        """Stock bf16 embedding: patch_embedding_mlp / action_embedder (bf16 linear)."""
        if stream == "video":
            return F.linear(tokens_bf16, self._patch_w, self._patch_b)
        return F.linear(tokens_bf16, self._act_w, self._act_b)

    def _program(self, stream: str, S: int, spans, head: int, tail: int, cuda_stream: int,
                 calibrate: bool):
        dims = {"S": S, "B": self.B, "spans": spans, "head": head, "tail": tail,
                "slab_rows": self.slab_rows, "out_dim": self._out_dim[stream]}
        dit_forward(fvk, self._ctx, self._bufs, self._weights[stream], dims, cuda_stream,
                    attn=self._attn, gate_fn=self._gate_fn, calibrate=calibrate,
                    precision=self.precision)

    @torch.no_grad()
    def _forward(self, stream: str, tokens_bf16: torch.Tensor, *, num_frames: int,
                 spans, kind: str) -> torch.Tensor:
        """tokens_bf16: (S, in_dim) for ONE stream (identical across CFG streams — only the
        cross-KV differs). kind in {transient, pred, commit}. Returns head_out (B, S, out_dim)
        fp16. Slab semantics: append (evicting oldest rows first), run, then restore the rows
        iff transient (ring_ref.LinearSlabRef == the stock allocator's live SET)."""
        assert self._prompt_set, "set_prompt first (message-order fail-loud)"
        assert not self.torch_ref_only, "engine compute needs the flash_rt extension"
        S = tokens_bf16.shape[0]
        B = self.B
        assert B * S <= self._Rmax, (S, self._Rmax)
        # host: slab bookkeeping BEFORE the program (eviction is permanent even for transient)
        self.slab.append(list(range(self._next_tok, self._next_tok + S)), kind)
        self._next_tok += S
        head, tail = self.slab.head, self.slab.tail
        # rope tables for this forward into the static per-stream buffers
        cos, sin = self._rope_tables(stream, self.frame_st_id, num_frames)
        self._cos[stream][:S].copy_(cos)
        self._sin[stream][:S].copy_(sin)
        # stock bf16 embedding → fp16 residual buffer, replicated over the B streams
        x0 = self._embed(stream, tokens_bf16.to(self.device, bf16)).to(fp16)
        xv = self._x[:B * S].view(B, S, DIT_D)
        for b in range(B):
            xv[b].copy_(x0)
        if self.debug_force_t0:
            spans = tuple((r0, n, self.point.t0_index(stream)) for (r0, n, _t) in spans)
        self._attn.text_len = self.debug_cross_len
        calibrate = bool(self.calibrating and self.precision == "fp8")
        key = (stream, S, tuple(spans), head, tail, self.debug_cross_len)
        if self.use_cuda_graph and not calibrate:
            g = self._graphs.get(key)
            if g is None:
                if (stream, S) not in self._warm_shapes:
                    # cuBLASLt creates its descriptor/algo cache on the FIRST call per
                    # (M, N, K) — never inside a capture. One eager forward warms the shape.
                    self._program(stream, S, spans, head, tail, 0, False)
                    self._warm_shapes.add((stream, S))
                    self._graph_seen[key] = 1
                else:
                    g = self._capture(key, stream, S, spans, head, tail)
            if g is not None:
                g.replay()
        else:
            self._program(stream, S, spans, head, tail, 0, calibrate)
            self._warm_shapes.add((stream, S))
        if calibrate:
            torch.cuda.synchronize()
            torch.maximum(self._act_amax_running, self._act_scales, out=self._act_amax_running)
        if kind == "transient":
            self.slab.restore(S)
        self.forward_log.append({"stream": stream, "S": S, "kind": kind, "spans": tuple(spans),
                                 "head": head, "tail": tail, "frame_st_id": self.frame_st_id})
        od = self._out_dim[stream]
        # the head GEMM writes a contiguous [R, out_dim] at the buffer start (row stride od)
        return self._head_out.view(-1)[:B * S * od].view(B, S, od).clone()

    def _capture(self, key, stream, S, spans, head, tail):
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        cs = torch.cuda.Stream()
        with torch.cuda.stream(cs):
            g.capture_begin()
            self._program(stream, S, spans, head, tail, cs.cuda_stream, False)
            g.capture_end()
        torch.cuda.synchronize()
        self._graphs[key] = g
        return g

    # ────────────────────────────────────────────────────────────
    # Calibration (fp8 arm): device act-scale slots, max over the calibration set
    # ────────────────────────────────────────────────────────────

    def begin_calibration(self):
        assert self.precision == "fp8"
        self.calibrating = True
        self._act_amax_running.zero_()

    def end_calibration(self) -> dict:
        assert self.calibrating
        self.calibrating = False
        self._act_scales.copy_(torch.clamp(self._act_amax_running, min=1e-12))
        sc = self._act_scales.view(DIT_L, ACT_SLOTS).cpu()
        return {"act_scales": sc.tolist(), "slots": ["qkv_in", "o_in", "cq_in", "co_in",
                                                     "ff1_in", "ff2_in"]}

    def load_act_scales(self, scales):
        t = torch.as_tensor(scales, dtype=torch.float32).reshape(-1)
        assert t.numel() == DIT_L * ACT_SLOTS, t.shape
        self._act_scales.copy_(t.to(self.device))

    def act_scales(self):
        return self._act_scales.view(DIT_L, ACT_SLOTS).cpu().clone()

    # ────────────────────────────────────────────────────────────
    # Cycle entries (the stock _infer / _compute_kv_cache loops, DiT → engine)
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _check_shape(tensor, expected, name):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name}: expected {expected}, got {tuple(tensor.shape)}")

    def _sched_step(self, sample, model_output, stream, i):
        """FlowMatchScheduler.step (sched:78-89), literally: prev = sample + out*(sigma_-sigma),
        sigma tensors fp32 CPU 0-dim, sigma_ = 0 (python int) at the last step. bf16 in/out."""
        sig = self._sigmas[stream]
        sigma = sig[i]
        sigma_ = sig[i + 1] if i + 1 < len(sig) else 0
        return sample + model_output * (sigma_ - sigma)

    @torch.no_grad()
    def commit_chunk(self, video_latents: torch.Tensor, action_state: torch.Tensor):
        """The 2 kv-commit forwards (t=0 rows, kind 'commit') — server:_compute_kv_cache."""
        assert self._prompt_set, "set_prompt first (message-order fail-loud)"
        g = self.geometry
        fv = int(video_latents.shape[2]) if video_latents.ndim == 5 else 0
        if not 1 <= fv <= g.frame_chunk + 1:
            raise ValueError(f"video commit frames {fv} exceed the build capacity")
        self._check_shape(video_latents, (1, LATENT_C, fv, g.latent_h, g.latent_w), "video commit")
        self._check_shape(action_state, (1, ACTION_DIM, g.frame_chunk, g.action_per_frame, 1), "action commit")
        t0 = time.perf_counter()
        self.slab.clear_pred()
        Fv = int(video_latents.shape[2]); Fa = int(action_state.shape[2])
        vt = self.point.t0_index("video"); at = self.point.t0_index("action")
        vtok = patchify(video_latents.to(self.device, bf16))[0]            # (Fv*120, 192)
        a = action_state.to(self.device, bf16).clone()
        a[:, self._action_unused] *= 0                                       # prepare's mask
        atok = actions_to_tokens(a)[0]                                       # (Fa*16, 30)
        self._forward("video", vtok, num_frames=Fv, spans=((0, vtok.shape[0], vt),),
                      kind="commit")
        self._forward("action", atok, num_frames=Fa, spans=((0, atok.shape[0], at),),
                      kind="commit")
        self.frame_st_id += Fv
        if self.profile:
            torch.cuda.synchronize()
            self.stage_ms["commit"] = (time.perf_counter() - t0) * 1000.0

    @torch.no_grad()
    def infer_cycle(self, noise_v: torch.Tensor, noise_a: torch.Tensor,
                    init_latent: Optional[torch.Tensor] = None):
        """V+1 video forwards then A+1 action forwards at the BUILD's grids (server:_infer)."""
        assert self._prompt_set, "set_prompt first (message-order fail-loud)"
        g = self.geometry
        self._check_shape(noise_v, (1, LATENT_C, g.frame_chunk, g.latent_h, g.latent_w), "video noise")
        self._check_shape(noise_a, (1, ACTION_DIM, g.frame_chunk, g.action_per_frame, 1), "action noise")
        if init_latent is not None:
            self._check_shape(init_latent, (1, LATENT_C, 1, g.latent_h, g.latent_w), "initial latent")
        cycle0 = self.frame_st_id == 0
        if cycle0 != (init_latent is not None):
            raise RuntimeError(
                f"infer_cycle: frame_st_id={self.frame_st_id} but init_latent "
                f"{'given' if init_latent is not None else 'missing'} — message-order fail-loud")
        dev = self.device
        B = self.B
        g_video = self.point.combine_scale("video")
        t_prof = time.perf_counter()
        # ── video loop ──
        latents = noise_v.to(dev, bf16).clone()                    # (1, 48, 2, 24, 20)
        latent_cond = init_latent.to(dev, bf16)[:, :, 0:1] if cycle0 else None
        tv = self._t_values["video"]
        vt0 = self.point.t0_index("video")
        for i, t in enumerate(tv):
            last = i == len(tv) - 1
            if cycle0:
                latents[:, :, 0:1] = latent_cond
                spans = ((0, g.tokens_per_frame, vt0),
                         (g.tokens_per_frame, g.video_tokens - g.tokens_per_frame, i))
            else:
                spans = ((0, g.video_tokens, i),)
            out = self._forward("video", patchify(latents)[0], num_frames=g.frame_chunk,
                                spans=spans, kind="pred" if last else "transient")
            if not last:
                pred = unpatchify(out.to(bf16), g.frame_chunk,
                                  g.latent_h, g.latent_w)
                if B == 2 and g_video > 1:
                    pred = pred[1:] + g_video * (pred[:1] - pred[1:])
                else:
                    pred = pred[:1]
                latents = self._sched_step(latents, pred, "video", i)
            if cycle0:
                latents[:, :, 0:1] = latent_cond
        if self.profile:
            torch.cuda.synchronize()
            self.stage_ms["video"] = (time.perf_counter() - t_prof) * 1000.0
            t_prof = time.perf_counter()
        # ── action loop ──
        actions = noise_a.to(dev, bf16).clone()                    # (1, 30, 2, 16, 1)
        ta = self._t_values["action"]
        at0 = self.point.t0_index("action")
        for i, t in enumerate(ta):
            last = i == len(ta) - 1
            if cycle0:
                actions[:, :, 0:1] = 0
                spans = ((0, g.action_per_frame, at0),
                         (g.action_per_frame, g.action_tokens - g.action_per_frame, i))
            else:
                spans = ((0, g.action_tokens, i),)
            actions[:, self._action_unused] *= 0                   # prepare's channel mask
            if last and self.action_terminal_elision:
                # P010: the action pred-commit forward has no reader (output guarded out, its
                # provisional K/V dropped by clear_pred at the next kv message). Replay ONLY its
                # slab allocation so the eviction side effect past saturation is preserved
                # (the naive skip diverged at the ring wrap, 2026-08-09) — no DiT compute.
                self.slab.append(list(range(self._next_tok, self._next_tok + g.action_tokens)), "pred")
                self._next_tok += g.action_tokens
                self.forward_log.append({"stream": "action", "S": g.action_tokens, "kind": "pred-elided",
                                         "spans": tuple(spans), "head": self.slab.head,
                                         "tail": self.slab.tail, "frame_st_id": self.frame_st_id})
                break
            out = self._forward("action", actions_to_tokens(actions)[0], num_frames=g.frame_chunk,
                                spans=spans, kind="pred" if last else "transient")
            if not last:
                pred = tokens_to_actions(out.to(bf16), g.frame_chunk)[:1]   # action positive-only
                actions = self._sched_step(actions, pred, "action", i)
            if cycle0:
                actions[:, :, 0:1] = 0
        actions[:, self._action_unused] *= 0
        if self.profile:
            torch.cuda.synchronize()
            self.stage_ms["action"] = (time.perf_counter() - t_prof) * 1000.0
        return actions, latents

    # ────────────────────────────────────────────────────────────
    # Introspection
    # ────────────────────────────────────────────────────────────

    def forwards_last_cycle(self, n_events=None) -> dict:
        kinds = [f["kind"] for f in self.forward_log]
        return {"total_forwards": len([k for k in kinds if k != "pred-elided"]),
                "transient": kinds.count("transient"), "pred": kinds.count("pred"),
                "pred_elided": kinds.count("pred-elided"), "commit": kinds.count("commit")}

    def launch_estimate(self, stream: str = "video", spans: int = 1) -> int:
        return launches_per_forward(self.B, spans, self.precision == "fp8",
                                    getattr(self._gate_fn, "kind", "") == "kernel")


__all__ = ["WanVaTorchFrontendThor", "CONDITIONING_HANDOFF_CONTRACT", "_ShardedCkpt",
           "_quantize_fp8", "digest", "patchify", "unpatchify", "actions_to_tokens",
           "tokens_to_actions", "ROBOTWIN_USED_ACTION_CHANNELS"]
