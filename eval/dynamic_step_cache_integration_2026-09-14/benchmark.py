"""Bounded Thor cache qualification and paired latency screen; no quality admission."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
from types import MethodType

import numpy as np
from PIL import Image
import torch

from instinctflash import Runtime
from load_memory import LoadingHeapTrim


MODEL = "GEAR-Dreams/DreamZero-DROID"
REVISION = "96ad344138c66e82536422432ad742f015784942"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def tensor_receipt(value):
    tensor = value.detach().contiguous()
    assert torch.isfinite(tensor).all(), "Nonfinite retained output or KV"
    raw = tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=digest(raw))


def rng_receipt():
    """Read generator states after diagnostics, never inside latency samples."""
    numpy_state = np.random.get_state()
    return dict(torch_cpu=digest(torch.random.get_rng_state().numpy().tobytes()),
                torch_cuda=digest(torch.cuda.get_rng_state().numpy().tobytes()),
                numpy=dict(algorithm=numpy_state[0], state_sha256=digest(numpy_state[1].tobytes()),
                           position=int(numpy_state[2]), has_gauss=int(numpy_state[3]),
                           cached_gaussian=float(numpy_state[4])),
                python=digest(json.dumps(random.getstate(), separators=(",", ":")).encode()))


def snapshot(head, raw_output):
    result = {"current_start_frame": int(head.current_start_frame)}
    for key in ("video_pred", "action_pred"):
        result[key] = tensor_receipt(raw_output[key])
    for key in ("kv_cache1", "kv_cache_neg", "crossattn_cache", "crossattn_cache_neg"):
        result[key] = [tensor_receipt(t) for t in getattr(head, key)]
    return result


class Observer:
    """Diagnostic-only wrappers; removed for every timing sample."""

    def __init__(self, head):
        from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler
        self.head, self.scheduler = head, FlowUniPCMultistepScheduler
        self.names = ("lazy_joint_video_action", "should_run_model", "_run_diffusion_steps")
        self.methods = {name: getattr(head, name) for name in self.names}
        self.existing = {name: vars(head).get(name) for name in self.names}
        self.scheduler_step = self.scheduler.step
        self.reset()

    def reset(self):
        self.output = None
        self.mask = []
        self.denoiser_branches = self.kv_branches = 0
        self.solvers = {}

    def __enter__(self):
        def generate(bound_head, *args, **kwargs):
            self.output = self.methods["lazy_joint_video_action"](*args, **kwargs)
            return self.output

        def decide(bound_head, index, current_timestep, history):
            answer = bool(self.methods["should_run_model"](index, current_timestep, history))
            assert index == len(self.mask)
            self.mask.append(answer)
            return answer

        def forward(bound_head, *args, **kwargs):
            result = self.methods["_run_diffusion_steps"](*args, **kwargs)
            if kwargs["kv_cache_metadata"]["update_kv_cache"]:
                self.kv_branches += len(result)
            else:
                self.denoiser_branches += len(result)
            return result

        def solver_step(scheduler, *args, **kwargs):
            key = id(scheduler)
            self.solvers.setdefault(key, []).append(int(kwargs["step_index"]))
            return self.scheduler_step(scheduler, *args, **kwargs)

        for name, method in zip(self.names, (generate, decide, forward)):
            object.__setattr__(self.head, name, MethodType(method, self.head))
        self.scheduler.step = solver_step
        return self

    def __exit__(self, *exc):
        for name, method in self.existing.items():
            if method is None:
                object.__delattr__(self.head, name)
            else:
                object.__setattr__(self.head, name, method)
        self.scheduler.step = self.scheduler_step

    def report(self):
        slots = list(self.solvers.values())
        assert len(self.mask) == 16
        assert len(slots) == 2 and all(s == list(range(16)) for s in slots)
        assert self.denoiser_branches == 2 * sum(self.mask)
        assert self.kv_branches == 2
        return dict(compute_mask=self.mask.copy(), computed_steps=sum(self.mask),
                    denoiser_branch_forwards=self.denoiser_branches,
                    kv_update_branch_forwards=self.kv_branches, solver_steps=slots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("precision", choices=("native", "fp8"))
    parser.add_argument("schedule", choices=("dynamic", "checkpoint"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-archive", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert torch.cuda.get_device_capability() == (11, 0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(9173)
    np.random.seed(9173)
    random.seed(9173)
    fixture = np.load(args.input_archive, allow_pickle=True)

    def decode(values):
        return [np.asarray(Image.open(io.BytesIO(bytes(value))).convert("RGB")) for value in values]

    frames = [decode(fixture["frame0_0"])] + [decode(value) for value in fixture["jpeg_0"][:12]]
    fixture.close()
    keys = ("observation/exterior_image_0_left", "observation/exterior_image_1_left",
            "observation/wrist_image_left")

    def observation(cycle, episode):
        indices = [0] if cycle == 0 else list(range(1, 5)) if cycle == 1 else list(range(5, 9))
        result = {key: np.stack([frames[i][camera] for i in indices]) for camera, key in enumerate(keys)}
        result["observation/joint_position"] = np.full(7, 0.01 * episode, np.float32)
        result["observation/gripper_position"] = np.zeros(1, np.float32)
        return result

    report = dict(status="running", precision=args.precision, schedule=args.schedule,
                  model=MODEL, revision=REVISION, torch=torch.__version__,
                  device=torch.cuda.get_device_name(), fixture_sha256=digest(args.input_archive.read_bytes()),
                  scope="Bounded Thor API latency and native-loop parity SCREEN; no task-quality certificate",
                  timing_scope={"included": "public predict and NumPy action conversion, synchronized before/after",
                                "excluded": ["model loading", "episode reset", "observation assembly",
                                             "seeding", "diagnostic observers", "tensor hashing", "receipt writes"],
                                "warmup_episodes": 1, "measured_episodes": 2,
                                "continuation_cycles_per_episode": 2},
                  quality_certified=False, timings=[], captures={}, competing_processes=[])
    outputs, api, trim = {}, None, None

    def save_progress():
        path = args.output.with_suffix(".progress.json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    def check_gpu():
        ids = subprocess.check_output(["/usr/sbin/nvidia-smi", "--query-compute-apps=pid",
                                       "--format=csv,noheader,nounits"], text=True).splitlines()
        other = [int(pid) for pid in ids if int(pid) != os.getpid()]
        if other:
            report["competing_processes"].append(other)
            raise RuntimeError(f"Concurrent GPU processes: {other}")

    def call(episode, cycle):
        if cycle == 0:
            api.reset(prompt="pick up the object" if episode % 2 == 0 else "place the object down")
        seed = 17000 + episode * 17 + cycle
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        result = api.predict(observation(cycle, episode))["action"]
        action = np.asarray(result)
        assert action.shape == (24, 8) and np.isfinite(action).all()
        return action.copy(), seed

    try:
        check_gpu()
        save_progress()
        trim = LoadingHeapTrim(args.output.with_suffix(".load-memory.jsonl")).start()
        start = time.perf_counter()
        api = Runtime.from_pretrained(MODEL, revision=REVISION, precision=args.precision,
                                      step_cache=args.schedule, tier_ceiling="behavioral", device="cuda:0",
                                      placement="in_process")
        report["preload_execution_policy"] = api.execution_policy
        api.reset(prompt="pick up the object")
        report["loading"] = trim.close()
        trim = None
        report["setup_seconds"] = time.perf_counter() - start
        loop = api._backend._loop if args.precision == "fp8" else api._backend._impl
        head = loop._wrapper._policy.trained_model.action_head
        assert head.num_inference_steps == 16 and float(head.cfg_scale) == 5.0
        assert head.dynamic_cache_schedule == (args.schedule == "dynamic")
        assert sum(head.dit_step_mask) == 8
        report["native_head_seed"] = getattr(head, "seed", None)
        if args.precision == "fp8":
            assert len(loop._fp8_recipe["projections"]) == 200
            report["fp8_recipe"] = loop._fp8_recipe
        report["stage"] = "public_timing"
        save_progress()
        for episode in range(3):
            for cycle in range(3):
                check_gpu()
                # Reset outside latency; predict remains the complete public API.
                if cycle == 0:
                    api.reset(prompt="pick up the object" if episode % 2 == 0 else "place the object down")
                seed = 17000 + episode * 17 + cycle
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                obs = observation(cycle, episode)
                torch.cuda.synchronize()
                start = time.perf_counter()
                action = np.asarray(api.predict(obs)["action"])
                torch.cuda.synchronize()
                ms = (time.perf_counter() - start) * 1000
                assert action.shape == (24, 8) and np.isfinite(action).all()
                stats = loop.backend_stats["step_cache"]
                if args.schedule == "dynamic":
                    assert stats["installed"]
                    assert stats["last_generation"]["denoiser_calls"] == stats["last_generation"]["computed_steps"]
                outputs[f"timing_{episode}_{cycle}"] = action.copy()
                report["timings"].append(dict(episode=episode, cycle=cycle, seed=seed, ms=ms,
                                              phase="warmup" if episode == 0 else "measured",
                                              step_cache=stats["last_generation"] if stats else None))
                save_progress()
                print("TIMING", episode, cycle, round(ms, 3), flush=True)

        if args.schedule == "dynamic":
            from dreamzero_iwm.dynamic_cache import install
            for role in ("shared", "native_oracle", "shared_restored"):
                if role == "native_oracle":
                    loop._step_cache_hook.close()
                    loop._step_cache_hook = None
                elif role == "shared_restored":
                    loop._step_cache_hook = install(head)
                report["stage"] = role
                bank = []
                report["captures"][role] = bank
                with Observer(head) as observer:
                    for episode in range(2):
                        for cycle in range(3):
                            check_gpu()
                            observer.reset()
                            action, seed = call(episode, cycle)
                            outputs[f"{role}_{episode}_{cycle}"] = action
                            values = snapshot(head, observer.output)
                            bank.append(dict(episode=episode, cycle=cycle, seed=seed,
                                             execution=observer.report(), values=values,
                                             rng_after=rng_receipt(),
                                             action_sha256=digest(action.tobytes())))
                            save_progress()
                            print("CAPTURE", role, episode, cycle, bank[-1]["execution"]["computed_steps"], flush=True)
                if role != "shared":
                    assert bank == report["captures"]["shared"], f"Native/shared mismatch in {role}"
            report["native_dynamic_parity"] = dict(passed=True, paired_requests=6,
                comparisons=12,
                kv_arrays_per_request=sum(len(report["captures"]["shared"][0]["values"][name])
                                          for name in ("kv_cache1", "kv_cache_neg", "crossattn_cache", "crossattn_cache_neg")),
                fields="full actions, raw video/action endpoints, all recorded KV arrays, compute masks, both solver clocks, CFG/prefill branch counts, post-call RNG states")
        samples = [row["ms"] for row in report["timings"] if row["phase"] == "measured" and row["cycle"] > 0]
        report["continuation_p50_ms"] = float(np.median(samples))
        report["continuation_sample_count"] = len(samples)
        report["execution_policy"] = api.execution_policy
        report["backend_stats"] = loop.backend_stats
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        already_failed = report["status"] == "failed"
        cleanup_errors = []
        for name, resource in (("loading_trim", trim), ("runtime", api)):
            if resource is not None:
                try:
                    resource.close()
                except BaseException:
                    cleanup_errors.append(dict(resource=name, error=traceback.format_exc()))
        if cleanup_errors:
            report["status"] = "failed"
            report["cleanup_errors"] = cleanup_errors
        if outputs:
            path = args.output.with_suffix(".npz")
            np.savez_compressed(path, **outputs)
            report["actions_sha256"] = digest(path.read_bytes())
        report["completed_utc_epoch"] = time.time()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        save_progress()
        if cleanup_errors and not already_failed:
            raise RuntimeError("Benchmark cleanup failed; see the saved receipt")


if __name__ == "__main__":
    main()
