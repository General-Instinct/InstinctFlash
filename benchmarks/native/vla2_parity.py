"""LingBot-VLA-V2 action parity: upstream server vs GI engine.

    python -m benchmarks.native.vla2_parity bundle --out vla2_bundle.pt      # upstream reference
    python -m benchmarks.native.vla2_parity run --engine gi --bundle vla2_bundle.pt --out gi.pt
    python -m benchmarks.native.vla2_parity compare vla2_bundle.pt gi.pt

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

MODEL = "robbyant/lingbot-vla-v2-6b-robotwin"


def _paths():
    from huggingface_hub import snapshot_download
    model = Path(snapshot_download(MODEL, local_files_only=True)) / "checkpoints/global_step_50000/hf_ckpt"
    root = Path(os.environ.get("LINGBOT_VLA_V2_ROOT", str(Path.home() / "lingbot-vla-v2-repo")))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("QWEN3VL_PATH", "Qwen/Qwen3-VL-4B-Instruct")
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/lingbot_vla_v2"))
    import transformers.tokenization_utils_base as tub
    if hasattr(tub.PreTrainedTokenizerBase, "_patch_mistral_regex"):
        tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, tok, *a, **k: tok)
    from lingbot_vla_v2_iwm.adapter import _LingBotVLAV2Loop, _project_cwd
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    with _project_cwd(root):
        server = LingbotVLAv2Server(str(model), use_length=50, chunk_ret=True, use_bf16=True,
                                    use_fp32=False, use_compile=False)
    embedder = server.vla.model.qwenvl_with_expert
    get_features = embedder.get_image_features
    captured = {}

    def hooked_features(*a, **k):
        merged, deep = get_features(*a, **k)
        captured["merged"] = merged.detach().float().reshape(-1, merged.shape[-1]).cpu()
        captured["deep"] = [d.detach().float().reshape(-1, d.shape[-1]).cpu() for d in deep]
        return merged, deep
    embedder.get_image_features = hooked_features
    original = server.sample_actions_fn
    gen = torch.Generator().manual_seed(args.seed)
    records = []

    def hooked(images, img_masks, lang_tokens, lang_masks, state, *a, noise=None, **k):
        noise = torch.randn((1, 50, 55), generator=gen).to(device=state.device, dtype=state.dtype)
        out = original(images, img_masks, lang_tokens, lang_masks, state, *a, noise=noise, **k)
        records.append({"images": images.detach().cpu(), "lang_tokens": lang_tokens.cpu(), "lang_masks": lang_masks.cpu(),
                        "state": state.float().cpu(), "noise": noise.float().cpu(), "ref": out.float().cpu(),
                        "merged": captured.pop("merged"), "deep": captured.pop("deep")})
        return out
    server.sample_actions_fn = hooked
    loop = _LingBotVLAV2Loop(server, root, robot="robotwin")
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
    from instinctflash.engines.vla2.engine import Vla2Engine
    from instinctflash.engines.vla2.weights import load_vla2_weights
    eng = Vla2Engine(load_vla2_weights(model, dev), dev)
    for r in data["records"]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        eng.set_prompt(r["lang_tokens"][r["lang_masks"].bool()].numpy())
        a = eng.infer(r["merged"].to(dev), [d.to(dev) for d in r["deep"]], r["state"].to(dev),
                      r["noise"][0].to(dev)).float().cpu()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)
        acts.append(a)
    torch.save({"engine": args.engine, "actions": torch.stack(acts), "latency_ms": lat,
                "device": torch.cuda.get_device_name()}, args.out)
    print(json.dumps({"engine": args.engine, "latency_ms": [round(v, 2) for v in lat]}))


def compare(args):
    data = torch.load(args.bundle)
    ref = torch.stack([r["ref"][0] for r in data["records"]])
    joint = list(range(12)) + [28, 29]   # the dims upstream keeps (arm + effectors)
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
