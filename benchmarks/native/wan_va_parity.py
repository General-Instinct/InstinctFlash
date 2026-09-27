"""LingBot-VA transformer parity: upstream WanTransformer3DModel vs the GI drop-in.

Both receive an identical scripted call sequence for N control cycles at 2V/4A with
CFG (B=2): video denoise forwards, the trailing video pred forward, action forwards,
the trailing action forward, then the commit (clear_pred + video/action t=0 with
update_cache=2). Every output the native server actually reads is compared.

    python -m benchmarks.native.wan_va_parity --cycles 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

LINGBOT = Path.home() / "lingbot-va"


def transformer_dir():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download("robbyant/lingbot-va-posttrain-robotwin", local_files_only=True)) / "transformer"


def _import_upstream():
    from instinctflash.runtime.lingbot_install import _ensure_flash_attn_importable
    _ensure_flash_attn_importable()
    if str(LINGBOT) not in sys.path:
        sys.path.insert(0, str(LINGBOT))


def script(cycles, device, seed=0):
    _import_upstream()
    from wan_va.utils.utils import get_mesh_id
    g = torch.Generator(device="cpu").manual_seed(seed)
    text = torch.randn(2, 512, 4096, generator=g).to(device, torch.bfloat16) * 0.1
    text[:, 200:] = 0          # T5 output is zero past the sequence length
    text[1] = 0                # negative prompt "" embeds to (near) zeros past its tokens
    steps = []

    def video(frame_st, t, uc, first):
        lat = torch.randn(1, 48, 2, 24, 20, generator=g).to(device, torch.bfloat16)
        ts = torch.full((2,), float(t), device=device)
        if first:
            ts[0] = 0
        grid = get_mesh_id(2, 12, 10, 0, 1, frame_st).to(device)
        return dict(noisy_latents=lat.repeat(2, 1, 1, 1, 1), timesteps=ts[None].repeat(2, 1),
                    grid_id=grid[None].repeat(2, 1, 1), text_emb=text), uc, False

    def action(frame_st, t, uc, first):
        lat = torch.randn(1, 30, 2, 16, 1, generator=g).to(device, torch.bfloat16)
        ts = torch.full((2,), float(t), device=device)
        if first:
            lat[:, :, 0:1] = 0
            ts[0] = 0
        grid = get_mesh_id(2, 16, 1, 1, 1, frame_st, action=True).to(device)
        return dict(noisy_latents=lat.repeat(2, 1, 1, 1, 1), timesteps=ts[None].repeat(2, 1),
                    grid_id=grid[None].repeat(2, 1, 1), text_emb=text), uc, True

    for c in range(cycles):
        fst = 2 * c
        vt = [1000.0, 833.3333129882812, 0.0]
        at = [1000.0, 750.0, 500.0, 250.0, 0.0]
        for i, t in enumerate(vt):
            steps.append(("video", c, i) + video(fst, t, 1 if i == len(vt) - 1 else 0, c == 0))
        for i, t in enumerate(at):
            steps.append(("action", c, i) + action(fst, t, 1 if i == len(at) - 1 else 0, c == 0))
        steps.append(("clear_pred", c, 0, None, None, None))
        steps.append(("commit_video", c, 0) + video(fst, 0.0, 2, False))
        steps.append(("commit_action", c, 0) + action(fst, 0.0, 2, False))
    return steps


def run_model(model, steps, reads):
    outs, lat = [], []
    for name, c, i, inp, uc, act in steps:
        if name == "clear_pred":
            model.clear_pred_cache("pos")
            continue
        inp = {k: v.clone() for k, v in inp.items()}
        torch.cuda.synchronize()
        t = time.perf_counter()
        y = model(inp, update_cache=uc, cache_name="pos", action_mode=act)
        torch.cuda.synchronize()
        lat.append(((name, c, i), (time.perf_counter() - t) * 1e3))
        if reads(name, uc):
            outs.append(((name, c, i), y.float().cpu()))
    return outs, lat


def reads(name, uc):
    return name in ("video", "action") and uc == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--save")
    args = p.parse_args()
    dev = torch.device("cuda")
    steps = script(args.cycles, dev)
    tdir = transformer_dir()
    _import_upstream()
    from wan_va.modules.model import WanTransformer3DModel
    ref = WanTransformer3DModel.from_pretrained(str(tdir), torch_dtype=torch.bfloat16).to(dev).eval()
    ref.create_empty_cache("pos", 72, 240, 32, dev, torch.bfloat16, 2)
    with torch.no_grad():
        ref_out, ref_lat = run_model(ref, steps, reads)
    del ref
    torch.cuda.empty_cache()
    from instinctflash.engines.wan_va.engine import GiWanTransformer
    from instinctflash.engines.wan_va.weights import load_wan_weights
    gi = GiWanTransformer(load_wan_weights(tdir, dev), dev, pool_slots=36 * 240 + 36 * 32)
    gi.create_empty_cache("pos", 72, 240, 32, dev, torch.bfloat16, 2)
    gi_out, gi_lat = run_model(gi, steps, reads)
    rows = []
    for (key, r), (_, g) in zip(ref_out, gi_out):
        name = key[0]
        if name == "action":
            r, g = r[:1], g[:1]          # the server reads the conditional branch only
        d = (g - r).abs()
        rows.append({"step": list(key), "max_rel": float(d.max() / r.abs().max()),
                     "mean_rel": float(d.mean() / r.abs().mean())})
    print(json.dumps({"forwards": rows, "gi_latency_ms": [(list(k), round(v, 2)) for k, v in gi_lat],
                      "ref_latency_ms": [(list(k), round(v, 2)) for k, v in ref_lat]}, indent=1))
    if args.save:
        torch.save({"ref": ref_out, "gi": gi_out}, args.save)


if __name__ == "__main__":
    main()
