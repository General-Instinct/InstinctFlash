"""Serve a PDD heads-only student through LingBot-VA's own denoise loop.

WHY NO NEW SAMPLER IS NEEDED, which is the whole trick here.

A PDD block step (Eq 10) advances L intervals from grid point n with one network evaluation:

    x <- x + sum_{l=n}^{n+L-1} h_l * u_l          u_l = head l's mean velocity

and because sum_{l=n}^{n+L-1} h_l = sigma_{n+L} - sigma_n, that is identical to

    x <- x + v_eff * (sigma_{n+L} - sigma_n)      v_eff = sum(h_l * u_l) / sum(h_l)

which is exactly the form `FlowMatchScheduler.step` already computes. So the student needs no bespoke
sampler: run the server at `num_inference_steps = N/L` and have the video head return `v_eff`.

AND THE GRIDS LINE UP EXACTLY. The N=256 grid's block boundary is sigma_128 = shift(1 - 128/256) =
shift(0.5) = 0.8333, and a 2-step scheduler's sigmas are shift([1.0, 0.5]) = [1.0, 0.8333]. Same
numbers, because the shift is applied pointwise to a linspace that both share. Verified in
tests/test_pdd_serve_parity.py rather than asserted.

THE HEADS COLLAPSE INTO ONE LINEAR PER BLOCK. Every head is a copy of `proj_out`, i.e. affine, so

    v_eff = sum_l w_l (W_l f + b_l) = (sum_l w_l W_l) f + (sum_l w_l b_l),   w_l = h_l / sum h

is a single affine map. Folding it once at load time means the student costs the SAME per forward as
the teacher -- L=128 head evaluations would otherwise be paid on every step for a result that is a
fixed linear combination. This is only valid because the heads are linear; a non-linear head would
have to be evaluated L times.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class _BlockHead(torch.nn.Module):
    """Replaces `transformer.proj_out` with one folded affine map per PDD block.

    Which block is active is derived from the conditioning timestep rather than from a call counter:
    a counter would silently desynchronise the moment the server inserts an extra forward, and it
    does -- the video loop runs one final cache-only forward whose output is discarded
    (wan_va_server.py:502-508). Reading the timestep cannot drift.
    """

    def __init__(self, folded, cond_at_block_start, fallback):
        super().__init__()
        self.folded = torch.nn.ModuleList(folded)
        self.register_buffer("starts", torch.tensor(cond_at_block_start, dtype=torch.float64))
        self.fallback = fallback          # the teacher's own proj_out, for the action stream
        self.current_t = None
        self.misses = 0

    def set_timestep(self, t) -> None:
        """`None` means "the caller is not on a video step" -- the action loop calls
        _prepare_latent_input with latent_in=None, and float(None) would raise on the first action
        step of the first chunk. Falling back to the teacher's proj_out is the correct behaviour
        there anyway, since only the video stream was distilled."""
        self.current_t = None if t is None else float(t)

    def forward(self, x):
        if self.current_t is None:
            self.misses += 1
            return self.fallback(x)
        d = (self.starts - self.current_t).abs()
        i = int(torch.argmin(d))
        if float(d[i]) > 1.0:             # timestep is not a block start: not a student step
            self.misses += 1
            return self.fallback(x)
        return self.folded[i](x)


def fold_heads(state_dict, grid, n_blocks: int, template: torch.nn.Linear):
    """Collapse each block's L heads into one affine map, weighted by interval width."""
    L = grid.block
    out = []
    for b in range(n_blocks):
        n = b * L
        hs = list(range(n, min(n + L, grid.n_intervals)))
        w = torch.tensor([grid.h(l) for l in hs], dtype=torch.float64)
        w = w / w.sum()                                  # sum h_l cancels against the sampler's dsigma
        lin = torch.nn.Linear(template.in_features, template.out_features,
                              bias=template.bias is not None)
        with torch.no_grad():
            # Accumulate on CPU: the state dict is loaded with map_location="cpu" while the template
            # lives on the GPU, and mixing them is a device error. Folding is a one-off at load time,
            # so doing it in fp64 on the host costs nothing and avoids a partial-sum rounding path.
            W = torch.zeros(template.weight.shape, dtype=torch.float64)
            B = (torch.zeros(template.bias.shape, dtype=torch.float64)
                 if template.bias is not None else None)
            for wi, l in zip(w, hs):
                W += state_dict[f"{l}.weight"].double() * wi
                if B is not None:
                    B += state_dict[f"{l}.bias"].double() * wi
            lin.weight.copy_(W.to(dtype=template.weight.dtype))
            if B is not None:
                lin.bias.copy_(B.to(dtype=template.bias.dtype))
        out.append(lin.to(device=template.weight.device, dtype=template.weight.dtype)
                   .requires_grad_(False))
    return out


def install_pdd_video_heads(server_module, server, ckpt_dir: str) -> list[str]:
    """Load a heads-only PDD student and serve its video stream at NFE = N/L.

    THE SIGN FLIP IS UNDONE HERE. The adapter negates velocities to express LingBot's descending-sigma
    field in instinct-pdd's ascending-t convention (dt = -dsigma). The server integrates in sigma, so
    the folded weights are negated back on the way in. Leaving it out would run the sampler backwards
    and produce noise, which is at least loud rather than subtle.
    """
    from instinctwm.adapter.lingbot_velocity import LingBotChunk0Video

    d = Path(ckpt_dir)
    meta = json.loads((d / "delta.json").read_text())
    if not meta.get("coverage_gate_pass", False):
        raise RuntimeError(
            f"{d}: delta.json says the coverage gate FAILED, so some heads are undertrained. "
            f"Refusing to serve it -- a checkpoint that cannot be defended should not produce a "
            f"benchmark number.")
    n_intervals, block = int(meta["n_intervals"]), int(meta["block"])
    nfe = n_intervals // block

    adapter = LingBotChunk0Video(server, guidance=float(meta["guidance"]["video"]))
    grid = adapter.grid(n_intervals, block)

    sd = torch.load(d / "heads.pt", map_location="cpu")
    sd = {k.split("head_list.")[-1] if "head_list." in k else k: v for k, v in sd.items()}

    template = server.transformer.proj_out
    folded = fold_heads(sd, grid, nfe, template)
    # NO SIGN FLIP HERE, and this is the subtle part. The adapter's `_Student.heads` returns `-y` so
    # that instinct-pdd sees an ascending-t velocity. The training loss therefore drove `-y` onto the
    # t-velocity, which means the head's RAW output y is already the SIGMA-velocity -- exactly what
    # FlowMatchScheduler.step consumes. Negating the folded weights (as a first version did) served
    # v_t where v_sigma was wanted, integrating away from the data: 0/100 success on RoboTwin against
    # 92/100 for the untrained 2-step control.

    starts = [grid.cond(b * block) for b in range(nfe)]
    head = _BlockHead(folded, starts, template).to(template.weight.device)
    server.transformer.proj_out = head

    # The video loop must take exactly `nfe` steps, and the action loop is untouched: only the video
    # stream was distilled, and the action stream reads the KV the video stream commits.
    for cfg in server_module.VA_CONFIGS.values():
        if hasattr(cfg, "num_inference_steps"):
            cfg.num_inference_steps = nfe
    server.job_config.num_inference_steps = nfe

    # Feed the head the timestep the server is currently on. `_prepare_latent_input` is the one place
    # that sees it for both streams, so hooking there cannot miss a call.
    orig_prepare = server._prepare_latent_input

    def prepare_with_t(latent_in, action_in, latent_t=0, action_t=0, *a, **k):
        head.set_timestep(float(latent_t) if latent_in is not None else None)
        return orig_prepare(latent_in, action_in, latent_t, action_t, *a, **k)

    server._prepare_latent_input = prepare_with_t
    return [f"pdd_video_heads(nfe={nfe},N={n_intervals},L={block})"]
