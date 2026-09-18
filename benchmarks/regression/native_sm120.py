"""Recorded-input validation for the existing native model path on RTX 5090.

Never selects FP8 or the experimental GEMM backend. A functional screen is not
a task-success certificate. Keep one fresh process per comparison arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from .gemm_guard import idle_admission, gpu_snapshot, require_uncontended
from .user_e2e import RecordedInputs, observed_schedule, prompt, request_hash, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("groot", "vla2", "dreamzero"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", choices=("upstream", "native"), default="native")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--numeric", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite previous evidence")
    if args.family == "dreamzero" and args.arm == "upstream":
        parser.error("Unmodified full-GPU DreamZero cannot fit; use native residency and its component parity gate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(ok=False, family=args.family, arm=args.arm, precision="native", calls=[],
                  task_quality="unqualified", input_kind="recorded cameras, synthetic robot states",
                  environment={k: v for k, v in os.environ.items() if k.startswith("IFL_")})
    api = None
    actions = {}
    try:
        report["gpu_admission"] = idle_admission("0")
        import numpy as np
        import torch
        from instinctflash import Runtime
        from instinctflash.descriptors.known import lookup
        from instinctflash.descriptors.package import _declared_view, from_pretrained
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("Expose exactly one SM120 GPU")
        if str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-") != report["gpu_admission"][-1]["gpu_uuid"].removeprefix("GPU-"):
            raise ValueError("CUDA device does not match the admission GPU")
        view = _declared_view(args.checkpoint, args.model_id, lookup(args.model_id))
        checkpoint = from_pretrained(view)
        report["checkpoint"] = str(args.checkpoint.resolve())
        report["config_sha256"] = sha(view / "config.json")
        start = time.perf_counter()
        if args.arm == "upstream":
            from .native_reference import build
            api = build(args.family, checkpoint, output_dir=args.output.parent)
        else:
            api = Runtime.from_pretrained(view, device="cuda:0", precision="native",
                placement="in_process", tier_ceiling="numeric" if args.numeric else "bitexact", seed=707)
        api.reset(prompt=prompt(0))
        torch.cuda.synchronize()
        report["setup_seconds"] = time.perf_counter() - start
        fixture = Path(__file__).parent / "fixtures/recorded_inputs_v1.npz"
        report["fixture_sha256"] = sha(fixture)
        inputs = RecordedInputs(fixture)
        expected = dict(checkpoint.execution.nfe)
        report["observed_nfe"] = observed_schedule(api, args.family, expected)
        for episode in range(args.episodes):
            api.reset(prompt=prompt(episode))
            for cycle in range(args.calls):
                require_uncontended(gpu_snapshot("0"), own_pids=(os.getpid(),), max_idle_utilization=100)
                i = episode * args.calls + cycle
                seed = 707 + i
                torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
                obs = inputs.observation(args.family, i, cycle % 3)
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = api.predict(obs)
                torch.cuda.synchronize()
                action = np.asarray(result["action"])
                if not np.isfinite(action).all() or not action.size:
                    raise ValueError("Invalid public action")
                key = f"episode_{episode}_call_{cycle}"
                actions[key] = action.copy()
                row = dict(episode=episode, cycle=cycle, seed=seed, shape=list(action.shape),
                           ms=1000*(time.perf_counter()-start),
                           input_sha256=request_hash(dict(observation=obs, prompt=prompt(episode))))
                report["calls"].append(row)
                print(json.dumps(row), flush=True)
        report["observed_nfe_after"] = observed_schedule(api, args.family, expected)
        stats = getattr(api, "backend_stats", {})
        report["backend_stats"] = stats() if callable(stats) else stats
        report["graph_stats"] = getattr(getattr(api._backend, "_impl", None), "graph_stats", {})
        report["peak_GiB"] = torch.cuda.max_memory_allocated()/2**30
        report["steady_GiB"] = torch.cuda.memory_allocated()/2**30
        report["experimental_gemm_loaded"] = "instinctflash_sm120_tma" in sys.modules
        if report["experimental_gemm_loaded"]:
            raise RuntimeError("Native validation must never load experimental GEMM")
        report["ok"] = True
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if api is not None:
            try:
                api.close()
            except BaseException:
                report.update(ok=False, close_error=traceback.format_exc())
        if actions:
            import numpy as np
            np.savez_compressed(args.output.with_suffix(".npz"), **actions)
        report["sources"] = {str(Path(m.__file__).resolve()): sha(m.__file__)
            for name, m in list(sys.modules.items()) if getattr(m, "__file__", None)
            and Path(m.__file__).is_file() and name.startswith(("instinctflash", "groot_n17_iwm", "lingbot_vla_v2_iwm", "dreamzero_iwm"))}
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
