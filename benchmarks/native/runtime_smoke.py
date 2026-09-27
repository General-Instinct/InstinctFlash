"""Public-Runtime FP8 smoke for one family on the recorded fixture: build, reset, N predicts.

Reports the backend the Runtime actually built, steady latency and action finiteness.

    python -m benchmarks.native.runtime_smoke --family vla4 --inputs recorded_inputs_v1.npz
"""
import argparse
import json
import time

import numpy as np

MODELS = {
    "pi05": ("lerobot/pi05_libero_finetuned_v044", "8e174154ef5f6c60a8da12ae99c303d8963138c1"),
    "vla4": ("robbyant/lingbot-vla-4b-posttrain-robotwin", "fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e"),
    "vla2": ("robbyant/lingbot-vla-v2-6b-robotwin", "0451855729ec904f970600e0aec8b84661423afe"),
    "groot": ("nvidia/GR00T-N1.7-3B", "2fc962b973bccdd5d8ce4f67cc63b264d6886495"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=sorted(MODELS), required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--calls", type=int, default=12)
    p.add_argument("--output", help="also write the report as JSON")
    args = p.parse_args()
    import torch
    from benchmarks.regression.user_e2e import RecordedInputs, prompt
    from instinctflash import Runtime
    model, revision = MODELS[args.family]
    t0 = time.perf_counter()
    rt = Runtime.from_pretrained(model, revision=revision, device="cuda:0", precision="fp8",
                                 tier_ceiling="numeric")
    build_s = time.perf_counter() - t0
    inputs = RecordedInputs(args.inputs)
    rt.reset(prompt=prompt(0))
    lat, finite = [], True
    for i in range(args.calls):
        obs = inputs.observation(args.family, i, 0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = rt.predict(obs)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)
        finite &= bool(np.isfinite(np.asarray(out["action"], np.float64)).all())
    backend = type(getattr(rt._backend, "_loop", rt._backend)).__module__
    stats = getattr(getattr(rt._backend, "_loop", None), "backend_stats", None)
    report = {"family": args.family, "model": model, "revision": revision, "build_s": round(build_s, 1),
              "first_ms": round(lat[0], 1), "calls_ms": [round(v, 1) for v in lat], "finite": finite,
              "loop_module": backend, "backend_stats": stats if isinstance(stats, dict) else None}
    text = json.dumps(report, default=str, indent=1)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    rt.close()


if __name__ == "__main__":
    main()
