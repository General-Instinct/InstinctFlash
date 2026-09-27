"""Paired pi05 action parity on identical processed inputs and noise.

    # 1. build a bundle with native references (any CUDA host that runs LeRobot)
    python -m benchmarks.native.pi05_parity bundle --count 12 --out pi05_bundle.pt
    # 2. run the GI engine on the bundle
    python -m benchmarks.native.pi05_parity run --engine gi --bundle pi05_bundle.pt --out gi.pt
    # 3. compare
    python -m benchmarks.native.pi05_parity compare pi05_bundle.pt gi.pt

Noise is rounded to FP16 (the precision earlier engine arms drew); every arm
consumes exactly the same values. Differences are reported on the
checkpoint's real action dimensions (normalized space).
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from benchmarks.native.pi05_engine_ab import CHECKPOINT, REVISION, _observations


def _checkpoint():
    from huggingface_hub import snapshot_download
    return snapshot_download(CHECKPOINT, revision=REVISION, local_files_only=True)


def processed_inputs(path, observations, device):
    from types import SimpleNamespace
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    cfg = PI05Config.from_pretrained(path)
    cfg.device = str(device)
    pre, _ = make_pre_post_processors(
        cfg, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}})
    anchor = torch.empty(0, device=device)
    shim = SimpleNamespace(config=cfg, parameters=lambda: iter([anchor]))
    out = []
    for obs in observations:
        batch = {k: torch.as_tensor(v).to(device) for k, v in obs.items() if k.startswith("observation.")}
        batch["task"] = obs["prompt"]
        batch = pre(batch)
        images, img_masks = PI05Policy._preprocess_images(shim, batch)
        out.append({"images": [i.cpu() for i in images], "img_masks": [m.cpu() for m in img_masks],
                    "tokens": batch["observation.language.tokens"].cpu(),
                    "masks": batch["observation.language.attention_mask"].bool().cpu()})
    return cfg, out


def bundle(args):
    dev = torch.device("cuda")
    path = _checkpoint()
    cfg, inputs = processed_inputs(path, _observations(args.inputs, args.count), dev)
    gen = torch.Generator().manual_seed(args.seed)
    noise = [torch.randn(cfg.chunk_size, 32, generator=gen).half().float() for _ in inputs]
    refs = {}
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    for dtype in ("float32", "bfloat16"):
        cfg.dtype = dtype
        policy = PI05Policy.from_pretrained(path, config=cfg).to(dev).eval()
        acts = []
        with torch.no_grad():
            for x, n in zip(inputs, noise):
                acts.append(policy.model.sample_actions(
                    [i.to(dev) for i in x["images"]], [m.to(dev) for m in x["img_masks"]],
                    x["tokens"].to(dev), x["masks"].to(dev), noise=n[None].to(dev)).float().cpu()[0])
        refs[dtype] = torch.stack(acts)
        del policy
        torch.cuda.empty_cache()
    action_dim = int(cfg.output_features["action"].shape[0])
    torch.save({"inputs": inputs, "noise": noise, "refs": refs, "action_dim": action_dim,
                "device": torch.cuda.get_device_name()}, args.out)
    d = (refs["bfloat16"] - refs["float32"])[..., :action_dim].abs()
    print(json.dumps({"calls": len(inputs), "bf16_vs_fp32_mean": float(d.mean()), "bf16_vs_fp32_max": float(d.max())}))


def _active(x):
    return [img[0] for img, m in zip(x["images"], x["img_masks"]) if bool(m.item())]


def run(args):
    data = torch.load(args.bundle)
    path = _checkpoint()
    dev = torch.device("cuda")
    acts, lat = [], []
    from instinctflash.engines.pi05.engine import Pi05Engine
    from instinctflash.engines.pi05.weights import load_pi05_weights
    weights = load_pi05_weights(path, dev)
    engines = {}
    for x, n in zip(data["inputs"], data["noise"]):
        views = _active(x)
        eng = engines.get(len(views)) or engines.setdefault(len(views), Pi05Engine(weights, len(views), dev))
        torch.cuda.synchronize()
        t = time.perf_counter()
        eng.set_prompt(x["tokens"][x["masks"]].numpy())
        a = eng.infer(torch.stack(views).to(dev), n.to(dev)).float().cpu()
        lat.append((time.perf_counter() - t) * 1e3)
        acts.append(a)
    torch.save({"engine": args.engine, "actions": torch.stack(acts), "latency_ms": lat,
                "device": torch.cuda.get_device_name()}, args.out)
    print(json.dumps({"engine": args.engine, "latency_ms": [round(v, 2) for v in lat]}))


def compare(args):
    data = torch.load(args.bundle)
    k = data["action_dim"]
    ref = data["refs"]["float32"][..., :k]
    rows = {"bf16_native": data["refs"]["bfloat16"][..., :k]}
    for p in args.results:
        r = torch.load(p)
        rows[f"{r['engine']}@{r['device']}"] = r["actions"][..., :k]
    out = {}
    for name, a in rows.items():
        d = (a - ref).abs()
        out[name] = {"mean_abs": float(d.mean()), "max_abs": float(d.max()),
                     "p99_abs": float(d.flatten().quantile(0.99)),
                     "per_call_mean": [round(float(v), 5) for v in d.mean((1, 2))]}
    print(json.dumps({"reference": "native fp32", "ref_mean_abs": float(ref.abs().mean()), **out}, indent=1))


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
