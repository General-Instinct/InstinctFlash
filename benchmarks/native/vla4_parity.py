"""LingBot-VLA-4B action parity: upstream server vs GI engine.

    python -m benchmarks.native.vla4_parity bundle --out vla4_bundle.pt      # upstream reference
    python -m benchmarks.native.vla4_parity run --engine gi --bundle vla4_bundle.pt --out gi.pt
    python -m benchmarks.native.vla4_parity compare vla4_bundle.pt gi.pt

The bundle records, per recorded observation, exactly what upstream passes to
``sample_actions`` (Qwen patches, masks, tokens, normalized state) together with
the noise it was given, the native BF16 vision embeddings and upstream's
normalized output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL = "robbyant/lingbot-vla-4b-posttrain-robotwin"


def _paths():
    from huggingface_hub import snapshot_download
    model = Path(snapshot_download(MODEL, local_files_only=True))
    root = Path(os.environ.get("LINGBOT_VLA_ROOT", str(Path.home() / "lingbot-vla-repo")))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("QWEN25_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
    return model, root


def _observations(archive, count):
    with np.load(archive, allow_pickle=False) as data:
        frames = data["frames"].copy()
    keys = ("observation.images.cam_high", "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist")
    out = []
    for i in range(count):
        out.append({**dict(zip(keys, frames[1 + i % 12])),
                    "observation.state": np.full(14, .05 * (i % 3), np.float32),
                    "prompt": "pick up the object" if (i // 4) % 2 == 0 else "place the object down"})
    return out


def bundle(args):
    model, root = _paths()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/lingbot_vla"))
    from lingbot_vla_iwm.adapter import _LingBotVLA4BLoop, _project_cwd
    from deploy.lingbot_vla_policy import LingbotVLAServer
    with _project_cwd(root):
        server = LingbotVLAServer(str(model), use_length=25,
                                  robot_norm_path=str(root / "assets/norm_stats/robotwin_50.json"),
                                  num_denoising_step=10, use_compile=False)
    policy_model = server.vla.model
    original = policy_model.sample_actions
    embedder = policy_model.qwenvl_with_expert
    embed_image = embedder.embed_image
    captured = {}

    def hooked_embed(images, *a, **k):
        out = embed_image(images, *a, **k)
        captured["vision"] = out.detach().float().reshape(-1, out.shape[-1]).cpu()
        return out
    embedder.embed_image = hooked_embed
    gen = torch.Generator().manual_seed(args.seed)
    records = []

    def hooked(images, img_masks, lang_tokens, lang_masks, state, *a, noise=None, **k):
        noise = torch.randn((1, 50, 75), generator=gen).to(device=state.device, dtype=state.dtype)
        out = original(images, img_masks, lang_tokens, lang_masks, state, *a, noise=noise, **k)
        records.append({"images": images.detach().cpu(), "img_masks": img_masks.detach().cpu(),
                        "lang_tokens": lang_tokens.cpu(), "lang_masks": lang_masks.cpu(),
                        "state": state.float().cpu(), "noise": noise.float().cpu(),
                        "ref": out.float().cpu(), "vision": captured.pop("vision")})
        return out
    policy_model.sample_actions = hooked
    loop = _LingBotVLA4BLoop(server, root, robot="robotwin")
    obs = _observations(args.inputs, args.count)
    loop.reset(prompt=obs[0]["prompt"])
    with torch.no_grad():
        for o in obs:
            loop.predict(o)
    torch.save({"records": records, "device": torch.cuda.get_device_name()}, args.out)
    print(json.dumps({"calls": len(records)}))


def run(args):
    data = torch.load(args.bundle)
    model, _ = _paths()
    dev = torch.device("cuda")
    acts, lat = [], []
    from instinctflash.engines.vla4.engine import Vla4Engine
    from instinctflash.engines.vla4.weights import load_vla4_weights
    eng = Vla4Engine(load_vla4_weights(model, dev), dev)
    for r in data["records"]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        eng.set_prompt(r["lang_tokens"][r["lang_masks"].bool()].numpy())
        a = eng.infer(r["vision"].to(dev), r["state"].to(dev), r["noise"][0].to(dev)).float().cpu()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)
        acts.append(a)
    torch.save({"engine": args.engine, "actions": torch.stack(acts), "latency_ms": lat,
                "device": torch.cuda.get_device_name()}, args.out)
    print(json.dumps({"engine": args.engine, "latency_ms": [round(v, 2) for v in lat]}))


def compare(args):
    data = torch.load(args.bundle)
    ref = torch.stack([r["ref"][0] for r in data["records"]])
    joint = list(range(12)) + [14, 15]   # the dims upstream keeps (unapply joint mask)
    ref = ref[..., joint]
    out = {"ref_mean_abs": float(ref.abs().mean())}
    for p in args.results:
        r = torch.load(p)
        d = (r["actions"][..., joint] - ref).abs()
        out[f"{r['engine']}@{r['device']}"] = {"mean_abs": float(d.mean()), "max_abs": float(d.max()),
                                                "p99_abs": float(d.flatten().quantile(0.99))}
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--inputs", default="/tmp/pi05_recorded_inputs_v1.npz")
    b.add_argument("--count", type=int, default=12)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", choices=("gi",), default="gi")
    r.add_argument("--bundle", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("bundle")
    c.add_argument("results", nargs="*")
    args = p.parse_args()
    {"bundle": bundle, "run": run, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
