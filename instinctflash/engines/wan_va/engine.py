"""GI LingBot-VA transformer: a drop-in for upstream ``WanTransformer3DModel`` at inference.

The native ``VA_Server`` keeps its own control loop (schedules, CFG combine, Euler,
noise, VAE, frame ids) and calls ``transformer(input_dict, update_cache, cache_name,
action_mode)`` plus the cache hooks. This object answers those calls with FP8 kernels:

* The self-attention KV pool has upstream's exact slot semantics (``pool.SlotPool``);
  attention gathers the live slots through a device index list, so one CUDA graph per
  (stream, batch, tokens) serves every history position and there is no episode cap.
* Work nothing can observe is skipped: action forwards and the trailing video forward
  compute only the conditional branch (the unconditional result is never read at
  action guidance 1), and the trailing action forward only performs its slot allocation.
* Flow-time modulation rows are tabulated per timestep value and selected per token.
"""
from __future__ import annotations

import numpy as np
import torch

from instinctflash.engines.common import Arena, GraphCache
from instinctflash.engines.wan_va.pool import SlotPool
from instinctflash.engines.wan_va.weights import D, FF, H, HD, time_rows
from instinctflash.kernels.attention import attention
from instinctflash.kernels.fp8 import E4M3, gemm
from instinctflash.kernels.norm import residual_norm_quant, row_quant
from instinctflash.kernels.wan import wan_qkv

EPS = 1e-6
TEXT = 512
TABLE_CAP = 96
ATTENTION_SITES = {"self": (64, 64, 4, 2, True), "cross": (64, 64, 4, 2, True)}


class _TimeTable:
    """Per-stream modulation rows [cap, 6, D] per block plus head rows, keyed by timestep value."""

    def __init__(self, weights, stream, device):
        self.w, self.stream = weights, stream
        n = len(weights.blocks)
        self.block = torch.zeros(n, TABLE_CAP, 6, D, dtype=torch.float32, device=device)
        self.head = torch.zeros(TABLE_CAP, 6, D, dtype=torch.float32, device=device)
        self.index = {}

    def entries(self, values):
        new = [v for v in dict.fromkeys(values) if v not in self.index]
        if new:
            if len(self.index) + len(new) > TABLE_CAP:
                raise RuntimeError("too many distinct VA timestep values for the modulation table")
            t = torch.tensor(new, dtype=torch.float32, device=self.head.device)
            temb, proj = time_rows(self.w.time[self.stream], t)
            rows = torch.arange(len(self.index), len(self.index) + len(new), device=t.device)
            for i, blk in enumerate(self.w.blocks):
                self.block[i, rows] = blk.sst[None] + proj.float()
            head = (self.w.head_sst[None] + temb[:, None]).float()        # BF16 add, as upstream
            one_plus = (1.0 + head[:, 1].to(torch.bfloat16)).float()       # (1 + scale) rounds to BF16
            self.head[rows, 0] = head[:, 0]
            self.head[rows, 1] = one_plus - 1.0
            for v, r in zip(new, rows.tolist()):
                self.index[v] = r
        return [self.index[v] for v in values]


class GiWanTransformer:
    def __init__(self, weights, device, *, pool_slots, max_tokens=512, max_batch=2):
        self.w = weights
        self.dev = torch.device(device)
        self.pool = SlotPool(pool_slots)
        self.pool_slots = int(pool_slots)
        self.max_tokens, self.max_batch = int(max_tokens), int(max_batch)
        self.attn = {k: (v[:4], v[4]) for k, v in ATTENTION_SITES.items()}
        dev, f32, f16, bf16 = self.dev, torch.float32, torch.float16, torch.bfloat16
        L, B, R = len(weights.blocks), self.max_batch, self.max_batch * self.max_tokens
        self.kpool = torch.zeros(L, B, pool_slots, D, dtype=f16, device=dev)
        self.vpool = torch.zeros_like(self.kpool)
        self.tk = torch.zeros(L, B, TEXT, D, dtype=f16, device=dev)
        self.tv = torch.zeros_like(self.tk)
        self.q8 = Arena(R * FF, E4M3, dev)
        self.raw = Arena(R * 3 * D, bf16, dev)
        self.raw_ff = Arena(R * FF, bf16, dev)
        self.scale = [torch.zeros(R, dtype=f32, device=dev) for _ in range(2)]
        # BF16 residual stream: upstream rounds the residual to BF16 after every add.
        self.x = torch.zeros(R, D, dtype=bf16, device=dev)
        self.y = torch.zeros(R, D, dtype=f32, device=dev)
        self.q16 = torch.zeros(R, D, dtype=f16, device=dev)
        self.o16 = torch.zeros(R, D, dtype=f16, device=dev)
        self.inp = torch.zeros(R, 192, dtype=f32, device=dev)
        self.out = torch.zeros(R, 192, dtype=f32, device=dev)
        self.pos = torch.zeros(self.max_tokens, 3, dtype=f32, device=dev)
        self.slots = torch.zeros(self.max_tokens, dtype=torch.int32, device=dev)
        self.live = torch.zeros(pool_slots, dtype=torch.int32, device=dev)
        self.n_live = torch.zeros(1, dtype=torch.int32, device=dev)
        self.mod_index = torch.zeros(R, dtype=torch.int32, device=dev)
        self.tables = {s: _TimeTable(weights, s, dev) for s in ("video", "action")}
        self._graphs = GraphCache(dev)
        self._text_ready = False
        self.forwards = 0
        self.skipped = 0

    # --------------------------------------------------------------- cache API
    def clear_cache(self, cache_name):
        self.pool.reset()
        self._text_ready = False

    def clear_pred_cache(self, cache_name):
        self.pool.clear_pred()

    def create_empty_cache(self, cache_name, attn_window, video_tokens, action_tokens, device, dtype, batch_size):
        slots = (attn_window // 2) * video_tokens + (attn_window // 2) * action_tokens
        if slots != self.pool_slots or batch_size > self.max_batch:
            raise ValueError(f"VA cache geometry {slots} slots x{batch_size} differs from the built engine")
        self.pool.reset()

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def __call__(self, input_dict, update_cache=0, cache_name="pos", action_mode=False, train_mode=False):
        if train_mode:
            raise RuntimeError("the GI VA engine serves inference only")
        lat = input_dict["noisy_latents"]
        B = lat.shape[0]
        if action_mode:
            tokens = lat.permute(0, 2, 3, 4, 1).reshape(B, -1, lat.shape[1])     # b (f h w) c
            per_frame = lat.shape[3] * lat.shape[4]
        else:
            b, c, f, h, w = lat.shape
            tokens = lat.view(b, c, f, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(b, f * (h // 2) * (w // 2), c * 4)
            per_frame = (h // 2) * (w // 2)
        T = tokens.shape[1]
        if T > self.max_tokens:
            raise ValueError(f"VA engine built for {self.max_tokens} tokens per forward, got {T}")
        if not self._text_ready:
            self._prepare_text(input_dict["text_emb"])
        slots = self.pool.append(T, is_pred=update_cache == 1)
        # Only the conditional branch of action forwards and of the trailing video
        # forward is ever read (action guidance 1); the trailing action forward is unread.
        skip = action_mode and update_cache == 1
        batch = 1 if (action_mode and update_cache in (0, 1)) or (not action_mode and update_cache == 1) else B
        out_dim = lat.shape[1] if action_mode else 192
        if skip:
            self.skipped += 1
            result = torch.zeros(B, T, out_dim, dtype=lat.dtype, device=lat.device)
        else:
            self._stage(tokens[:batch], input_dict, slots, per_frame, T, batch, action_mode)
            self._graphs.run((action_mode, batch, T), self._forward)
            y = self.out[: batch * T, :out_dim].view(batch, T, out_dim)
            result = y.to(lat.dtype).expand(B, T, out_dim).clone() if batch < B else y.to(lat.dtype).clone()
            self.forwards += 1
        if update_cache == 0:
            self.pool.restore(slots)
        if action_mode:
            return result
        return result.reshape(B, T * 4, 48)

    def _stage(self, tokens, input_dict, slots, per_frame, T, batch, action_mode):
        n_in = tokens.shape[-1]
        self.inp[: batch * T, :n_in].copy_(tokens.reshape(batch * T, n_in))
        grid = input_dict["grid_id"][0, :3].t()
        self.pos[:T].copy_(grid)
        live = self.pool.live()
        self.slots[:T].copy_(torch.from_numpy(slots.astype(np.int32)), non_blocking=True)
        self.live[: live.size].copy_(torch.from_numpy(live.astype(np.int32)), non_blocking=True)
        self.n_live.fill_(int(live.size))
        stream = "action" if action_mode else "video"
        t = input_dict["timesteps"][0].float().cpu().tolist()
        entries = self.tables[stream].entries(t)
        per_token = np.repeat(np.asarray(entries, dtype=np.int32) * 6, per_frame)
        idx = torch.from_numpy(np.tile(per_token, batch))
        self.mod_index[: batch * T].copy_(idx, non_blocking=True)

    def _prepare_text(self, text_emb):
        """Text embedder + per-layer cross K/V for both branches (once per episode)."""
        w = self.w
        B = text_emb.shape[0]
        th = torch.nn.functional.gelu(text_emb.float() @ w.text_w1.t() + w.text_b1, approximate="tanh")
        th = (th @ w.text_w2.t() + w.text_b2).reshape(B * TEXT, D)
        q8 = self.q8(B * TEXT, D)
        s = torch.empty(B * TEXT, dtype=torch.float32, device=self.dev)
        row_quant(th, q8, s)
        raw = torch.empty(B * TEXT, 2 * D, dtype=torch.bfloat16, device=self.dev)
        dummy_q = torch.empty(B * TEXT, D, dtype=torch.float16, device=self.dev)
        for i, blk in enumerate(w.blocks):
            gemm(q8, blk.ckv, raw)
            # Parts [K | V]: K gets the full-width norm; the "Q" output slot receives K.
            wan_qkv(raw, dummy_q, heads=H, head_dim=HD, tokens=TEXT, row=s, col=blk.ckv.scale,
                    bias=blk.ckv.bias, q_norm=blk.cnorm_k, v_pool=None, eps=EPS)
            self.tk[i, :B].copy_(dummy_q.view(B, TEXT, D))
            self.tv[i, :B].view(B * TEXT, D).copy_(
                (raw[:, D:].float() * s[:, None] * blk.ckv.scale[D:] + blk.ckv.bias[D:]).to(torch.float16))
        self._text_ready = True

    def _forward(self, key):
        action_mode, B, T = key
        w, q8, raw, s = self.w, self.q8, self.raw, self.scale
        R = B * T
        x = self.x[:R]
        stream = "action" if action_mode else "video"
        tbl = self.tables[stream]
        if action_mode:
            x.copy_(torch.addmm(w.action_b, self.inp[:R, :w.action_w.shape[1]], w.action_w.t()))
        else:
            x.copy_(torch.addmm(w.patch_b, self.inp[:R], w.patch_w.t()))
        mi = self.mod_index[:R]
        t0 = tbl.block[0]
        residual_norm_quant(x, layernorm=True, mod_scale=t0[0, 1], mod_shift=t0[0, 0], mod_index=mi,
                            q=q8(R, D), q_scale=s[0], eps=EPS)
        qv = self.q16[:R].view(B, T, H, HD)
        ov = self.o16[:R].view(B, T, H, HD)
        scfg, stc5 = self.attn["self"]
        ccfg, ctc5 = self.attn["cross"]
        last = len(w.blocks) - 1
        for i, blk in enumerate(w.blocks):
            tb = tbl.block[i]
            kp = self.kpool[i, :B]
            vp = self.vpool[i, :B]
            gemm(q8(R, D), blk.qkv, raw(R, 3 * D))
            wan_qkv(raw(R, 3 * D), self.q16[:R], heads=H, head_dim=HD, tokens=T, k_pool=kp, v_pool=vp,
                    slots=self.slots, row=s[0], col=blk.qkv.scale, bias=blk.qkv.bias, q_norm=blk.norm_q,
                    k_norm=blk.norm_k, pos=self.pos, sections=w.sections, inv_freq=w.inv_freq, eps=EPS)
            attention(qv, kp.view(B, self.pool_slots, H, HD), vp.view(B, self.pool_slots, H, HD), out=ov,
                      kv_len=self.n_live, kv_index=self.live, config=scfg, tc5=stc5)
            row_quant(self.o16[:R], q8(R, D), s[1], num_warps=4)
            gemm(q8(R, D), blk.o, raw(R, D))
            residual_norm_quant(x, delta=raw(R, D), delta_row=s[1], delta_col=blk.o.scale, bias=blk.o.bias,
                                gate=tb[0, 2], mod_index=mi, layernorm=True, weight=blk.norm2_w, beta=blk.norm2_b,
                                q=q8(R, D), q_scale=s[0], eps=EPS)
            gemm(q8(R, D), blk.cq, raw(R, D))
            wan_qkv(raw(R, D), self.q16[:R], heads=H, head_dim=HD, tokens=T, row=s[0], col=blk.cq.scale,
                    bias=blk.cq.bias, q_norm=blk.cnorm_q, eps=EPS)
            attention(qv, self.tk[i, :B].view(B, TEXT, H, HD), self.tv[i, :B].view(B, TEXT, H, HD), out=ov,
                      config=ccfg, tc5=ctc5)
            row_quant(self.o16[:R], q8(R, D), s[1], num_warps=4)
            gemm(q8(R, D), blk.co, raw(R, D))
            residual_norm_quant(x, delta=raw(R, D), delta_row=s[1], delta_col=blk.co.scale, bias=blk.co.bias,
                                layernorm=True, mod_scale=tb[0, 4], mod_shift=tb[0, 3], mod_index=mi,
                                q=q8(R, D), q_scale=s[0], eps=EPS)
            gemm(q8(R, D), blk.ff1, self.raw_ff(R, FF))
            row_quant(self.raw_ff(R, FF), q8(R, FF), s[1], row=s[0], col=blk.ff1.scale, bias=blk.ff1.bias,
                      gelu=True, num_warps=8)
            gemm(q8(R, FF), blk.ff2, raw(R, D))
            if i < last:
                nt = tbl.block[i + 1]
                residual_norm_quant(x, delta=raw(R, D), delta_row=s[1], delta_col=blk.ff2.scale, bias=blk.ff2.bias,
                                    gate=tb[0, 5], layernorm=True, mod_scale=nt[0, 1], mod_shift=nt[0, 0],
                                    mod_index=mi, q=q8(R, D), q_scale=s[0], eps=EPS)
            else:
                residual_norm_quant(x, delta=raw(R, D), delta_row=s[1], delta_col=blk.ff2.scale, bias=blk.ff2.bias,
                                    gate=tb[0, 5], layernorm=True, mod_scale=tbl.head[0, 1],
                                    mod_shift=tbl.head[0, 0], mod_index=mi, y=self.y[:R], eps=EPS)
        if action_mode:
            torch.addmm(w.action_out_b, self.y[:R], w.action_out_w.t(), out=self.out[:R, :w.action_out_w.shape[0]])
        else:
            torch.addmm(w.proj_out_b, self.y[:R], w.proj_out_w.t(), out=self.out[:R])
