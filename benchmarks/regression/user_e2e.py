"""Installed-package public API latency capture, one fresh process per matrix cell.

The fixture contains recorded cameras and feedback, with synthetic robot states.
This is observation-to-action wall time, not a camera driver or task-success test.
Run with ``python -m benchmarks.regression.user_e2e --help``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .hardware import bound_target, probe_device, validate_device_receipt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_hash(value):
    """Bind nested public inputs including array shape/dtype, without pickle."""
    import numpy as np
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"list:{len(item)}:".encode())
            for child in item:
                add(child)
        elif isinstance(item, np.ndarray):
            digest.update(json.dumps(["array", item.dtype.str, item.shape]).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        else:
            digest.update(json.dumps([type(item).__name__, item], sort_keys=True).encode())
    add(value)
    return digest.hexdigest()


class RecordedInputs:
    def __init__(self, archive):
        import numpy as np
        from PIL import Image
        with np.load(archive, allow_pickle=False) as data:
            modern = {"frames", "feedback"} <= set(data.files)
            if modern:
                frames, feedback = data["frames"].copy(), data["feedback"].copy()
                if (frames.dtype != np.uint8 or frames.ndim != 5 or frames.shape[0] != 13
                        or frames.shape[1] != 3 or frames.shape[-1] != 3
                        or feedback.dtype != np.float32 or feedback.shape != (8, 16, 2, 16)
                        or not np.isfinite(feedback).all()):
                    raise ValueError("Unexpected recorded fixture shape or dtype")
                # Preserve the original nested-list observation contract (GR00T
                # consumes a list of camera arrays rather than one stacked array).
                self.frames = [[image for image in views] for views in frames]
                self.feedback = feedback
        if not modern:
            # Compatibility for the single hash-pinned historical fixture. Arbitrary
            # user archives must never reach numpy's object-array/pickle decoder.
            if sha(archive) != "d6f08f968287b78eadd0ef3001e90f0a46397b17294dbd2c46c854feb5b93172":
                raise ValueError("Use the public non-pickle frames/feedback fixture")
            with np.load(archive, allow_pickle=True) as data:
                def decode(values):
                    return [np.asarray(Image.open(io.BytesIO(bytes(v))).convert("RGB")) for v in values]
                self.frames = [decode(data["frame0_0"])] + [decode(v) for v in data["jpeg_0"][:12]]
                self.feedback = data["actions_0"].copy()

    def observation(self, family, i, cycle):
        import numpy as np
        from PIL import Image
        frames = self.frames
        images = frames[1 + i % 12]
        if family == "pi05":
            return {"observation.images.image": images[0].transpose(2, 0, 1).astype(np.float32)/255,
                    "observation.images.image2": images[1].transpose(2, 0, 1).astype(np.float32)/255,
                    "observation.state": np.zeros(8, np.float32)}
        if family in ("vla4", "vla2"):
            keys = ("observation.images.cam_high", "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist")
            return {**dict(zip(keys, images)), "observation.state": np.full(14, .05*(i % 3), np.float32)}
        if family == "groot":
            return {"images": images[:2], "state": {
                "eef_9d": np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], np.float32),
                "gripper_position": np.zeros(1, np.float32),
                "joint_position": np.zeros(7, np.float32)}}
        if family in ("edge", "nano"):
            return {"image": np.asarray(Image.fromarray(images[0]).resize((640, 540))),
                    "state": np.full(8, .01*(i % 3), np.float32), "prompt": prompt(i)}
        if family == "va":
            keys = ("observation.images.cam_high", "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist")
            indices = [0] if cycle == 0 else list(range(1, 5)) if cycle == 1 else list(range(5, 13))
            return {"obs": [dict(zip(keys, frames[j])) for j in indices]}
        if family == "dreamzero":
            keys = ("observation/exterior_image_0_left", "observation/exterior_image_1_left",
                    "observation/wrist_image_left")
            indices = [0] if cycle == 0 else list(range(1, 5)) if cycle == 1 else list(range(5, 9))
            return {**{key: np.stack([frames[j][v] for j in indices]) for v, key in enumerate(keys)},
                    "observation/joint_position": np.zeros(7, np.float32),
                    "observation/gripper_position": np.zeros(1, np.float32)}
        raise ValueError(f"Unsupported fixture family: {family}")


def prompt(episode):
    return "pick up the object" if episode % 2 == 0 else "place the object down"


def _loop(api):
    backend = api._backend  # Instrumentation only; requests use the public API.
    return getattr(backend, "_loop", None) or getattr(backend, "_impl", None)


def _generator_loop(api):
    """Read the loaded generator beneath a generic FP8 execution wrapper."""
    loop = _loop(api)
    return getattr(loop, "inner", loop)


def queue_length(api):
    loop = _generator_loop(api)
    if hasattr(loop, "_queue"):
        return len(loop._queue)
    return len(loop._p._action_queue)


def fp8_weights(root):
    """Read packed-weight metadata after timing; never convert or mutate tensors."""
    import torch
    seen, found = set(), []
    # Plain policy/server objects bridge the public loop to its nn.Module. Use
    # their defining namespaces: sys.modules aliases do not change __module__.
    packages = (
        "instinctflash", "flash_rt", "lingbotvla", "lingbot_vla_iwm",
        "lingbot_vla_v2_iwm", "groot_n17_iwm", "gr00t", "groot",
        "cosmos3_iwm", "cosmos_framework", "dreamzero_iwm", "pi05_iwm",
        "eval_utils", "wan_va",
    )
    modules = {"wan_va_server", "deploy.lingbot_vla_policy",
               "deploy.lingbot_vla_v2_policy"}

    def visit(value, path, depth=0):
        if id(value) in seen or depth > 128:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float8_e4m3fn and value.ndim == 2:
                found.append({"path": path, "shape": list(value.shape)})
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, path+"."+str(key), depth+1)
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                visit(child, path+f"[{i}]", depth+1)
        elif (isinstance(value, torch.nn.Module)
              or type(value).__module__ in modules
              or any(type(value).__module__ == package
                     or type(value).__module__.startswith(package + ".")
                     for package in packages)):
            if hasattr(value, "__dict__"):
                visit(vars(value), path, depth+1)
    visit(root, "backend")
    return found


def gpu_competitors(torch):
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise RuntimeError("nvidia-smi is required on PATH for isolated GPU timing")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    output = subprocess.check_output([smi, "--query-compute-apps=gpu_uuid,pid",
                                      "--format=csv,noheader,nounits"], text=True, timeout=15)
    return [int(parts[1]) for line in output.splitlines()
            if len(parts := line.split(",")) == 2 and parts[0].strip() == uuid
            and int(parts[1]) != os.getpid()]


def dreamzero_contract(api, *, dynamic):
    from dreamzero_iwm.adapter import SHIPPED_DIT_MASK, _head_declaration
    declaration = _head_declaration(_generator_loop(api)._wrapper._policy.trained_model.action_head)
    assert declaration["steps"] == {"video_action": 16, "kv_commit": 1}
    assert declaration["guidance"]["video_action"] == ("cfg", 5.0)
    assert tuple(declaration["dit_step_mask"]) == SHIPPED_DIT_MASK
    assert declaration["dynamic_cache_schedule"] is dynamic
    return declaration


def observed_schedule(api, family, expected):
    """Read loaded generator settings independently of Runtime's request metadata."""
    inner = _generator_loop(api)
    if family == "pi05":
        config = inner.config if hasattr(inner, "_queue") else inner._p.config
        result = {"action": int(config.num_inference_steps)}
    elif family == "vla4":
        result = {"action": int(inner._server.num_denoising_step)}
    elif family == "vla2":
        result = {"action": int(inner._server.vla.model.config.num_steps)}
    elif family == "groot":
        result = {"action": int(inner._policy.model.action_head.num_inference_timesteps)}
    elif family in ("edge", "nano"):
        config = inner._service.cfg
        assert float(config.guidance) == 3.0 and float(config.shift) == 5.0
        result = dict(inner.declaration()["steps"])
    elif family == "va":
        if hasattr(inner, "declaration"):
            declaration = inner.declaration()
            result = dict(declaration["steps"])
            assert declaration["guidance"]["video"] == ("cfg", 5.0)
            assert float(declaration["guidance"]["action"][1]) == 1.0
        else:
            config = inner._server.job_config
            result = {"video": int(config.num_inference_steps),
                      "action": int(config.action_num_inference_steps)}
            assert float(config.guidance_scale) == 5.0 and float(config.action_guidance_scale) == 1.0
    else:
        result = {"video_action": inner.declaration()["steps"]["video_action"]}
    assert result and all(expected.get(key) == value for key, value in result.items()), (result, expected)
    return result


def native_schedule_override(cell):
    """Validate a declared upstream schedule separately from Runtime options."""
    if "native_nfe" not in cell:
        return None
    if cell.get("arm") != "eager_native":
        raise ValueError("native_nfe is only valid for the eager_native arm")
    override = cell["native_nfe"]
    if not isinstance(override, dict):
        raise ValueError("native_nfe must be a nonempty video/action mapping")
    expected = cell.get("effective_schedule", {}).get("nfe")
    from .native_reference import resolve_native_nfe
    effective = resolve_native_nfe(cell.get("family"), expected, override)
    if effective != expected:
        raise ValueError("native_nfe differs from the frozen effective schedule")
    return dict(override)


def capture(matrix_path, cell_id, output_root, fixture):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    from instinctflash import Runtime

    matrix = json.loads(Path(matrix_path).read_text())
    target = bound_target(matrix.get("target"))
    cell = next(row for row in matrix["cells"] if row["id"] == cell_id)
    native_nfe = native_schedule_override(cell)
    output = Path(output_root) / cell["receipt"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(f"Capture is immutable: {output}")
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("Installed-package validation requires an unset PYTHONPATH")
    optimizer_environment = {k: v for k, v in os.environ.items() if k.startswith("IFL_")}
    if optimizer_environment != cell.get("expected_optimizer_environment", {}):
        raise RuntimeError("Optimizer environment does not match the frozen matrix")
    options = dict(cell.get("expected_runtime_kwargs", {}))
    family = cell["family"]
    history = family in ("va", "dreamzero")
    report = dict(schema=1, target=target, cell_id=cell_id, family=family, arm=cell["arm"],
        runtime_kwargs=options, optimizer_environment=optimizer_environment,
        model_id=cell["model_id"], revision=cell["revision"],
        precision=options.get("precision", "native"), device=torch.cuda.get_device_name(),
        torch=torch.__version__, cases=[], calls=[], ok=False,
        quality_certified=False, interpreter=sys.executable, cwd=os.getcwd(),
        input_archive_sha256=sha(fixture), matrix_sha256=sha(matrix_path),
        input_contract={"family": family, "fixture_sha256": sha(fixture),
                        "fixture_protocol": "recorded_cameras_synthetic_states_v1",
                        "action_shape": cell["action_shape"]},
        setup_scope="Checkpoint resolution, construction and initial reset; first predict reported separately",
        scope="Public predict including input processing, generation, output CPU transfer and feedback commit; prepared cameras, synthetic states; no network or simulator")
    if native_nfe is not None:
        report["native_nfe_override"] = native_nfe
    api = None
    try:
        report["hardware"] = probe_device(target, torch)
        validate_device_receipt(report["hardware"], target)
        report["nvidia_smi"] = shutil.which("nvidia-smi")
        assert not gpu_competitors(torch), "Unexpected competing GPU process"
        snapshot = Path(snapshot_download(cell["model_id"], revision=cell["revision"], local_files_only=True))
        assert snapshot.name == cell["revision"]
        report["checkpoint_snapshot"] = str(snapshot)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(9173)
        np.random.seed(9173)
        random.seed(9173)
        start = time.perf_counter()
        if cell["arm"] == "eager_native":
            from instinctflash.descriptors.package import from_pretrained

            from .native_reference import build
            checkpoint = from_pretrained(cell["model_id"], revision=cell["revision"])
            if native_nfe is None:
                api = build(family, checkpoint, output_dir=output.parent,
                            target=target, hardware=report["hardware"])
            else:
                api = build(family, checkpoint, output_dir=output.parent, nfe=native_nfe,
                            target=target, hardware=report["hardware"])
        else:
            api = Runtime.from_pretrained(cell["model_id"], revision=cell["revision"], **options)
            report["observation_contract"] = api.observation.describe()
        report["package_path"] = str(api._checkpoint.path)
        # Keep declared loading/residency changes even if the first reset fails.
        report["execution_policy"] = api.execution_policy
        api.reset(prompt=prompt(0))
        torch.cuda.synchronize()
        report["setup_seconds"] = time.perf_counter() - start
        report["observed_nfe_before"] = observed_schedule(api, family, cell["effective_schedule"]["nfe"])
        if family == "dreamzero":
            report["observed_schedule_before"] = dreamzero_contract(api, dynamic=options.get("step_cache") == "dynamic")
        inputs = RecordedInputs(fixture)
        count = 21 if history else 25
        actions = []

        def request(i, episode, cycle, phase, *, queue=False):
            assert not gpu_competitors(torch), f"Unexpected GPU process at request {i}"
            observation = inputs.observation(family, i, cycle)
            instruction = prompt(episode)
            if queue:
                instruction = prompt(0)
            feedback = ({"executed_action": inputs.feedback[cycle % len(inputs.feedback)].copy()}
                        if family == "va" else {})
            seed = 2707+i if queue else 1300+episode*3+cycle if history else 707+i
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            case = dict(i=i, episode=episode, cycle=cycle, phase=phase, seed=seed,
                        input_sha256=request_hash({"observation": observation, "prompt": instruction}),
                        feedback_sha256=request_hash(feedback), call_kind="generation")
            before = queue_length(api) if family == "pi05" else None
            if before:
                case["call_kind"] = "queue_hit"
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = api.predict(observation, **feedback)
            action = np.asarray(result["action"])
            torch.cuda.synchronize()
            ms = 1000*(time.perf_counter()-start)
            if list(action.shape) != cell["action_shape"] or not np.isfinite(action).all():
                raise RuntimeError(f"Unexpected/nonfinite public action: {action.shape}")
            call = dict(i=i, ms=ms, shape=list(action.shape))
            if family == "pi05":
                call.update(queue_before=before, queue_after=queue_length(api))
            if family == "dreamzero":
                call["step_cache"] = _generator_loop(api).backend_stats.get("step_cache")
            print(json.dumps({**case, **call}), flush=True)
            return case, call, action.copy()

        for i in range(count):
            episode, cycle = divmod(i, 3) if history else (i, 0)
            phase = "warmup" if (episode == 0 if history else i < 5) else "measured"
            if cycle == 0:
                api.reset(prompt=prompt(episode))
            case, call, action = request(i, episode, cycle, phase)
            report["cases"].append(case)
            report["calls"].append(call)
            actions.append(action)

        if family == "pi05":
            api.reset(prompt=prompt(0))
            queue_cases, queue_calls, queue_actions = [], [], []
            for i in range(51):
                case, call, action = request(i, 0, i, "queue_drain", queue=True)
                queue_cases.append(case)
                queue_calls.append(call)
                queue_actions.append(action)
            assert [c["i"] for c in queue_cases if c["call_kind"] == "generation"] == [0, 50]
            queue_file = output.with_suffix(".queue.npz")
            np.savez_compressed(queue_file, actions=np.stack(queue_actions))
            report["queue_drain"] = dict(archive=str(queue_file.relative_to(Path(output_root))), actions_sha256=sha(queue_file),
                actions_key="actions", cases=queue_cases, calls=queue_calls,
                reset_between_calls=False, expected_calls=51)

        loop = _loop(api)
        if report["precision"] == "fp8":
            report["e4m3_tensors"] = fp8_weights(loop)
            if not report["e4m3_tensors"]:
                raise RuntimeError("FP8 selected but no actual packed E4M3 projection weights found")
        stats = getattr(api, "backend_stats", {})
        report["backend_stats"] = stats() if callable(stats) else stats
        if family == "dreamzero":
            report["observed_schedule_after"] = dreamzero_contract(api, dynamic=options.get("step_cache") == "dynamic")
        report["graph_stats"] = getattr(loop, "graph_stats", {})
        checkpoint = api._checkpoint
        report["default_schedule"] = dict(checkpoint.execution.nfe or {})
        report["observed_nfe_after"] = observed_schedule(api, family, cell["effective_schedule"]["nfe"])
        report["guidance"] = str(checkpoint.execution.guidance)
        report["effective_schedule"] = {**cell["effective_schedule"],
            "nfe": {**report["default_schedule"],
                    **(native_nfe if native_nfe is not None else options.get("nfe", {}))}}
        report["execution_policy"] = api.execution_policy
        if cell["arm"] != "eager_native":
            if dict(report["execution_policy"].get("nfe", {})) != report["effective_schedule"]["nfe"]:
                raise RuntimeError("Runtime effective NFE differs from frozen schedule")
        report["plan"] = api.plan.explain()
        report["applied_passes"] = [r.name for r in api.plan.results if r.applies]
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["numeric_environment"] = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark)
        archive = output.with_suffix(".npz")
        np.savez_compressed(archive, actions=np.stack(actions))
        report["actions_sha256"] = sha(archive)
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
        report["sources"] = {str(Path(module.__file__).resolve()): sha(module.__file__)
            for name, module in list(sys.modules.items())
            if name.startswith(("instinctflash", "flash_rt", "pi05_iwm", "lingbot_vla", "groot_n17_iwm",
                "cosmos3_iwm", "dreamzero_iwm", "benchmarks.regression", "lerobot", "gr00t", "groot",
                "cosmos_framework", "wan_va", "deploy", "eval_utils"))
            and getattr(module, "__file__", None) and str(module.__file__).endswith((".py", ".so"))
            and Path(module.__file__).is_file()}
        output.write_text(json.dumps(report, indent=2, default=str)+"\n")
    return 0 if report["ok"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True,
                        help="Trusted repository-owned recorded-camera archive")
    args = parser.parse_args(argv)
    return capture(args.matrix, args.cell, args.output_root, args.fixture)


if __name__ == "__main__":
    raise SystemExit(main())
