#!/usr/bin/env python
"""Assert that the text conditioning the SERVER computes equals what TRAINING baked in.

Why this check exists
---------------------
LingBot-VA never runs T5 during training. `wan_va/dataset/lerobot_latent_dataset.py`
reads a *precomputed* `text_emb` out of each dataset `.pth`, and for the classifier-free
guidance drop it substitutes a precomputed `empty_emb.pt`. The server does the opposite:
`wan_va_server.py::_reset` calls `encode_prompt(...)` which runs T5 live, applies
`prompt_clean()`, and right-zero-pads to `max_sequence_length=512`.

So there are two entirely separate code paths producing the tensor that conditions every
denoising step, and nothing in the repo asserts they agree. This is the same shape as a
failure that previously voided a 22.7-hour, 2000-rollout run on this box: training used
one prompt format, all eight eval servers silently resolved another, and the only symptom
was a warning nobody read.

Two things are checked, and both must hold or no number from this box is reportable:

  A. live T5(action_text, len=512)  ==  dataset `.pth` `text_emb`
  B. live T5("",         len=512)   ==  `empty_emb.pt`      (the CFG-negative branch)

Note on (B): `guidance_scale=5 > 1` for RoboTwin, so the negative branch is not
decorative -- it is subtracted from every video-stream prediction with a weight of 5.
A wrong empty embedding therefore corrupts the output *more* than a wrong positive one.

The check deliberately calls the server's OWN bound method rather than reimplementing
the ~20 lines of tokenize/encode/pad, because a reimplementation would test the
reimplementation. VA_Server.__new__ is used to get an instance without loading the
10 GB transformer.

Usage:
    python check_prompt_parity.py --latent <a training .pth> --empty-emb <empty_emb.pt>
Exit code 0 = parity holds. Non-zero = do not run an eval.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("LINGBOT_CKPT"))
    ap.add_argument("--latent", required=True, help="a training latent .pth with a baked text_emb")
    ap.add_argument("--empty-emb", required=True, help="empty_emb.pt from the training dataset")
    ap.add_argument("--lingbot-root", default=os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va"))
    ap.add_argument("--cos-tol", type=float, default=0.999)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.lingbot_root, "wan_va"))

    from configs import VA_CONFIGS
    from modules.utils import load_text_encoder, load_tokenizer
    from wan_va_server import VA_Server

    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = args.ckpt

    # Build a VA_Server shell WITHOUT __init__ (which would load the 10 GB transformer),
    # then attach only what _get_t5_prompt_embeds touches. This keeps us on the real
    # server code path: same prompt_clean, same padding, same dtype handling.
    srv = VA_Server.__new__(VA_Server)
    srv.job_config = cfg
    srv.dtype = cfg.param_dtype
    srv.device = torch.device("cuda:0")
    srv.tokenizer = load_tokenizer(os.path.join(args.ckpt, "tokenizer"))
    srv.text_encoder = load_text_encoder(
        os.path.join(args.ckpt, "text_encoder"),
        torch_dtype=cfg.param_dtype,
        torch_device=srv.device,
    )

    train = torch.load(args.latent, map_location="cpu", weights_only=False)
    text = train["text"]
    train_emb = train["text_emb"].float()
    empty_emb = torch.load(args.empty_emb, map_location="cpu", weights_only=False)
    if isinstance(empty_emb, dict):
        empty_emb = empty_emb.get("text_emb", next(iter(empty_emb.values())))
    empty_emb = empty_emb.float().squeeze(0) if empty_emb.dim() == 3 else empty_emb.float()

    print(f"training action_text: {text!r}")

    failures = []

    def compare(name, live, ref):
        live = live.detach().float().cpu().squeeze(0)
        ref = ref.detach().float().cpu().squeeze(0) if ref.dim() == 3 else ref.detach().float().cpu()
        print(f"\n--- {name} ---")
        print(f"  live shape {tuple(live.shape)}   train shape {tuple(ref.shape)}")
        if live.shape != ref.shape:
            # A 226-vs-512 mismatch lands here. encode_prompt's own default is 226 and
            # only the _reset call site passes 512, so this is one edit away from happening.
            failures.append(f"{name}: SHAPE MISMATCH {tuple(live.shape)} vs {tuple(ref.shape)}")
            print("  FAIL: shape mismatch")
            return
        max_abs = (live - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            live.flatten()[None], ref.flatten()[None]
        ).item()
        # Compare only the non-padding rows too: zero padding inflates cosine similarity
        # and could hide a real divergence in the few tokens that carry the prompt.
        nz = ref.abs().sum(-1) > 0
        cos_nz = torch.nn.functional.cosine_similarity(
            live[nz].flatten()[None], ref[nz].flatten()[None]
        ).item() if nz.any() else float("nan")
        print(f"  non-pad rows      : {int(nz.sum())} / {nz.numel()}")
        print(f"  max |live - train|: {max_abs:.3e}")
        print(f"  cosine (all)      : {cos:.6f}")
        print(f"  cosine (non-pad)  : {cos_nz:.6f}")
        if not (cos_nz >= args.cos_tol):
            failures.append(f"{name}: cosine(non-pad) {cos_nz:.6f} < {args.cos_tol}")
            print("  FAIL")
        else:
            print("  OK")

    with torch.no_grad():
        live_pos = srv._get_t5_prompt_embeds(prompt=text, max_sequence_length=512)
        live_neg = srv._get_t5_prompt_embeds(prompt="", max_sequence_length=512)

    compare("A: positive prompt (live T5 vs dataset text_emb)", live_pos, train_emb)
    compare("B: CFG negative '' (live T5 vs empty_emb.pt)", live_neg, empty_emb)

    print("\n" + "=" * 70)
    if failures:
        print("PROMPT PARITY: FAIL")
        for f in failures:
            print("  -", f)
        print("Do NOT run an evaluation. A model served a conditioning signal it was")
        print("never trained on produces a number that is noise, not a weak baseline.")
        return 1
    print("PROMPT PARITY: PASS -- serving reproduces the training text conditioning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
