"""Engine state isolation: every record's actions must not depend on what ran before it.

Runs a parity bundle through the GI engine in recorded order and in a permuted order
(prompt changes, same-length and changed-length prompts, camera counts, repeated
observations) and requires bit-identical actions per record. A stale prompt buffer,
KV extent, graph binding or per-token scale buffer shows up as a mismatch.

    python -m benchmarks.native.order_check --family vla4 --bundle vla4_bundle.pt
"""
import argparse
import importlib
import json
import os
import tempfile

import torch


def _permute(data, order):
    out = dict(data)
    for key in ("records", "inputs", "noise"):
        if key in data:
            out[key] = [data[key][i] for i in order]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("pi05", "vla4", "vla2", "groot"), required=True)
    p.add_argument("--bundle", required=True)
    args = p.parse_args()
    parity = importlib.import_module(f"benchmarks.native.{args.family}_parity")
    data = torch.load(args.bundle)
    n = len(data.get("records") or data["inputs"])
    # Reverse, then interleave halves: every record gets a different predecessor.
    orders = {"recorded": list(range(n)), "reversed": list(range(n))[::-1],
              "interleaved": [i for pair in zip(range(n // 2), range(n // 2, n)) for i in pair] + list(range(2 * (n // 2), n))}
    actions = {}
    with tempfile.TemporaryDirectory() as td:
        for name, order in orders.items():
            bundle = os.path.join(td, f"{name}.pt")
            torch.save(_permute(data, order), bundle)
            out = os.path.join(td, f"{name}_out.pt")
            parity.run(argparse.Namespace(engine="gi", bundle=bundle, out=out))
            got = torch.load(out)["actions"]
            restored = torch.empty_like(got)
            restored[torch.tensor(order)] = got
            actions[name] = restored
    ref = actions["recorded"]
    report = {name: {"bit_identical": bool(torch.equal(a, ref)),
                     "max_abs": float((a - ref).abs().max())} for name, a in actions.items() if name != "recorded"}
    print(json.dumps({"family": args.family, "records": n, **report}, indent=1))
    raise SystemExit(0 if all(r["bit_identical"] for r in report.values()) else 1)


if __name__ == "__main__":
    main()
