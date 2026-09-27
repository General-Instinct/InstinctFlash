"""GR00T N1.7 action-head parity: upstream head vs GI engine.

    python -m benchmarks.native.groot_parity bundle --out groot_bundle.pt     # native reference
    python -m benchmarks.native.groot_parity run --engine gi --bundle groot_bundle.pt --out gi.pt
    python -m benchmarks.native.groot_parity compare groot_bundle.pt gi.pt

The bundle records, for each recorded observation, what the native backbone hands the
action head (features, image/attention masks, state, embodiment), the noise upstream
drew, and upstream's ``action_pred`` in normalized model space.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

MODEL = "nvidia/GR00T-N1.7-3B"


def _observations(archive, count):
    with np.load(archive, allow_pickle=False) as data:
        frames = data["frames"].copy()
    out = []
    for i in range(count):
        out.append({"images": [frames[1 + i % 12][0], frames[1 + i % 12][1]], "state": {
            "eef_9d": np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], np.float32) + 0.01 * (i % 3),
            "gripper_position": np.zeros(1, np.float32),
            "joint_position": np.zeros(7, np.float32) + 0.02 * (i % 3)},
            "prompt": "pick up the object" if (i // 4) % 2 == 0 else "place the object down"})
    return out


def _checkpoint_dir():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, local_files_only=True)


def bundle(args):
    from instinctflash import Runtime
    rt = Runtime.from_pretrained(MODEL, device="cuda:0", precision="native")
    obs = _observations(args.inputs, args.count)
    rt.reset(prompt=obs[0]["prompt"])
    backend = rt._backend
    loop = getattr(backend, "_loop", None) or backend._impl
    head = loop._policy.model.action_head
    native = head.get_action
    records = []
    real_randn = torch.randn

    def hooked(backbone_output, action_input, options=None):
        drawn = []

        def randn(*a, **k):
            t = real_randn(*a, **k)
            if tuple(t.shape) == (1, 40, 132):
                drawn.append(t.detach().float().cpu())
            return t
        torch.randn = randn
        try:
            out = native(backbone_output, action_input, options)
        finally:
            torch.randn = real_randn
        records.append({"features": backbone_output.backbone_features[0].float().cpu(),
                        "image_mask": backbone_output.image_mask[0].bool().cpu(),
                        "attention_mask": backbone_output.backbone_attention_mask[0].bool().cpu(),
                        "state": action_input.state.float().cpu(),
                        "embodiment": int(action_input.embodiment_id.item()),
                        "noise": drawn[0], "ref": out["action_pred"].float().cpu()[0]})
        return out
    head.get_action = hooked
    for o in obs:
        rt.predict(o)
    torch.save({"records": records, "device": torch.cuda.get_device_name(),
                "embodiment_tag": loop._policy.embodiment_tag.value}, args.out)
    print(json.dumps({"calls": len(records), "seq": [int(r["features"].shape[0]) for r in records]}))
    rt.close()


def run(args):
    data = torch.load(args.bundle)
    path = _checkpoint_dir()
    dev = torch.device("cuda")
    acts, lat = [], []
    from instinctflash.engines.groot_n17.engine import GrootN17Engine
    from instinctflash.engines.groot_n17.weights import load_groot_weights
    eng = GrootN17Engine(load_groot_weights(path, dev, data["records"][0]["embodiment"]), dev)
    for r in data["records"]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        a = eng.infer(r["features"].to(dev), r["image_mask"].to(dev), r["attention_mask"].to(dev),
                      r["state"].to(dev), r["noise"][0].to(dev)).float().cpu()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)
        acts.append(a)
    torch.save({"engine": args.engine, "actions": torch.stack(acts), "latency_ms": lat,
                "device": torch.cuda.get_device_name()}, args.out)
    print(json.dumps({"engine": args.engine, "latency_ms": [round(v, 2) for v in lat]}))


def compare(args):
    data = torch.load(args.bundle)
    ref = torch.stack([r["ref"] for r in data["records"]])[..., :17]
    out = {"ref_mean_abs": float(ref.abs().mean())}
    for p in args.results:
        r = torch.load(p)
        d = (r["actions"][..., :17] - ref).abs()
        out[f"{r['engine']}@{r['device']}"] = {"mean_abs": float(d.mean()), "max_abs": float(d.max()),
                                                "p99_abs": float(d.flatten().quantile(0.99))}
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--inputs", default="/tmp/pi05_recorded_inputs_v1.npz")
    b.add_argument("--count", type=int, default=12)
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
