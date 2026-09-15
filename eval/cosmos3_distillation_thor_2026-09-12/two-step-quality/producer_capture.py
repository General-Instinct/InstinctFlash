#!/usr/bin/env python3
"""Compare native Cosmos DROID samplers on recorded observations and actions.

Control arms use the supplied native checkpoint's original network. An optional
declared candidate binds trained output-head tensors only during its own arm.
Generation goes through the native RoboLab processors and generate_samples_from_batch,
including native noise, CFG, conditioning masks, padded action channels, and decoding.
These open-loop action diagnostics do not measure task success or establish that
agreement with a recorded action chunk or a higher-step sampler is better policy.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


RF_GRID = [1.0, 0.9375, 5.0 / 6.0, 0.625, 0.0]
BASELINE = "unipc4_cfg3"
DIAGNOSTIC = "unipc30_cfg3"
CANDIDATE = "candidate_unipc4_cfg1"


def arms(include_fixed_cfg1=False, include_candidate=False):
    values = [
        {"name": BASELINE, "sampler": "unipc", "steps": 4, "guidance": 3.0,
         "role": "primary_operational_baseline", "max_callbacks": 4},
        {"name": DIAGNOSTIC, "sampler": "unipc", "steps": 30, "guidance": 3.0,
         "role": "higher_step_diagnostic", "max_callbacks": None},
        {"name": "unipc4_cfg1", "sampler": "unipc", "steps": 4, "guidance": 1.0,
         "role": "guidance_control", "max_callbacks": 4},
        {"name": "fixed4_cfg3", "sampler": "fixed", "steps": 4, "guidance": 3.0,
         "role": "sampler_control_original_weights", "max_callbacks": 4},
    ]
    if include_fixed_cfg1:
        values.append({"name": "fixed4_cfg1", "sampler": "fixed", "steps": 4,
                       "guidance": 1.0, "role": "sampler_and_guidance_control_original_weights",
                       "max_callbacks": 4})
    if include_candidate:
        values.append({"name": CANDIDATE, "sampler": "unipc", "steps": 4,
                       "guidance": 1.0, "role": "trained_output_head_candidate",
                       "max_callbacks": 4})
    for arm in values:
        arm["shift"] = 5.0 if arm["sampler"] == "unipc" else None
        arm["rf_grid"] = RF_GRID if arm["sampler"] == "fixed" else None
    return values


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_candidate_declaration(state_path, declaration_path, checkpoint_files, train_data):
    """Check the declared local training artifacts before loading any tensors."""
    if train_data is None:
        raise ValueError("A trained candidate requires --train-data for split and provenance validation")
    declaration = json.loads(Path(declaration_path).read_text())
    required = {"base_checkpoint_files", "training_state_sha256", "train_data_sha256", "head_names",
                "sampler", "steps", "guidance", "shift", "protocol_sha256"}
    if not isinstance(declaration, dict) or set(declaration) != required:
        raise ValueError(f"Candidate declaration must contain exactly {sorted(required)}")
    for key in ("training_state_sha256", "train_data_sha256", "protocol_sha256"):
        if not isinstance(declaration[key], str) or not re.fullmatch(r"[0-9a-f]{64}", declaration[key]):
            raise ValueError(f"Invalid candidate {key}")
    expected_execution = {"head_names": ["llm2action"], "sampler": "unipc", "steps": 4,
                          "guidance": 1.0, "shift": 5.0}
    if any(declaration[key] != value for key, value in expected_execution.items()):
        raise ValueError("Candidate must declare output-head-only native UniPC4 CFG1 shift5")
    if type(declaration["steps"]) is not int or any(
            type(declaration[key]) not in (int, float) for key in ("guidance", "shift")):
        raise ValueError("Candidate execution values must be numbers, not booleans")
    state_identity = file_identity(state_path)
    if declaration["training_state_sha256"] != state_identity["sha256"]:
        raise ValueError("Candidate training-state SHA256 differs from its declaration")
    if declaration["train_data_sha256"] != train_data["sha256"]:
        raise ValueError("Candidate training data differs from the supplied train-data")
    # Match the native-file scope captured by the training source_identity helper.
    suffixes = {".json", ".safetensors", ".pth", ".pt", ".model"}
    expected_files = {entry["relative_path"]: {"sha256": entry["sha256"], "bytes": entry["bytes"]}
                      for entry in checkpoint_files if Path(entry["relative_path"]).suffix in suffixes}
    declared_files = declaration["base_checkpoint_files"]
    if not expected_files or declared_files != expected_files:
        raise ValueError("Candidate native base checkpoint file inventory differs from the loaded checkpoint")
    for entry in declared_files.values():
        if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"} or type(entry["bytes"]) is not int:
            raise ValueError("Malformed candidate base-checkpoint file entry")
    folder = Path(state_path).resolve().parent
    configuration_path, identity_path = folder / "configuration.json", folder / "source_identity.json"
    configuration = json.loads(configuration_path.read_text())
    identity = json.loads(identity_path.read_text())
    provenance = configuration["provenance"]
    if (identity["native_checkpoint_files"] != declared_files
            or provenance["train_data_sha256"] != declaration["train_data_sha256"]
            or provenance["protocol_sha256"] != declaration["protocol_sha256"]
            or provenance["source_identity_sha256"] != file_identity(identity_path)["sha256"]):
        raise ValueError("Candidate configuration, source identity and declaration disagree")
    target_cache_path = folder / "target_cache.json"
    optional_files = []
    if "target_cache_sha256" in provenance:
        target_identity = file_identity(target_cache_path)
        if target_identity["sha256"] != provenance["target_cache_sha256"]:
            raise ValueError("Candidate teacher target-cache receipt differs from training provenance")
        optional_files.append(target_identity)
    expected_sampler = {"kind": "native_unipc", "num_steps": 4, "guidance": 1.0, "shift": 5.0}
    if provenance.get("student_sampler") != expected_sampler:
        raise ValueError("Candidate training provenance does not declare native UniPC4 CFG1 shift5")
    protocol_path = Path(configuration["arguments"]["protocol"])
    if file_identity(protocol_path)["sha256"] != declaration["protocol_sha256"]:
        raise ValueError("Candidate protocol differs from the frozen training declaration")
    plan = json.loads(protocol_path.read_text())["candidate"]
    arguments = configuration["arguments"]
    if any(arguments.get(key) != plan.get(key) for key in ("iterations", "learning_rate", "seed")):
        raise ValueError("Candidate training arguments differ from the frozen protocol")
    expected_plan = {"num_steps": 4, "shift": 5.0, "teacher_guidance": 3.0, "student_guidance": 1.0,
                     "scope": "output_head", "gradients": "last_native_velocity_callback_only",
                     "train_data_sha256": declaration["train_data_sha256"]}
    if (any(plan.get(key) != value for key, value in expected_plan.items())
            or type(plan.get("iterations")) is not int or plan["iterations"] < 1):
        raise ValueError("Candidate frozen protocol does not match the supported final-output-head recipe")
    return {"declaration": declaration, "configuration": configuration, "frozen_candidate_recipe": plan,
            "files": [file_identity(declaration_path), state_identity, file_identity(configuration_path),
                      file_identity(identity_path), file_identity(protocol_path), *optional_files]}


def load_candidate(adapter, state_path, evidence):
    """Load only strict, finite FP32 output-head tensors from a verified local state."""
    import torch

    # Standard v2 snapshots also contain Python/NumPy RNG state. The complete
    # trusted local file was verified above before deserializing that format.
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("format") != "instinct_compress_training" or state.get("version") != 2:
        raise ValueError("Candidate requires a standard instinct_compress v2 training snapshot")
    if state.get("provenance") != evidence["configuration"]["provenance"]:
        raise ValueError("Candidate state provenance differs from its verified configuration")
    if state.get("config") != evidence["configuration"]["training"]:
        raise ValueError("Candidate state training configuration differs from its verified receipt")
    technique = state.get("technique", {})
    expected = {"sampler": "native_unipc", "num_steps": 4, "shift": 5.0,
                "student_guidance": 1.0, "teacher_guidance": 3.0, "scope": "output_head",
                "gradient": "last_velocity_callback"}
    if any(technique.get(key) != value for key, value in expected.items()):
        raise ValueError("Candidate state technique is incompatible with the declared evaluation arm")
    final_iteration = evidence["frozen_candidate_recipe"]["iterations"]
    if any(type(state.get(key)) is not int or state[key] != final_iteration
           for key in ("student_updates", "iteration")):
        raise ValueError(f"Candidate must be the frozen final checkpoint at update {final_iteration}")
    candidate = adapter.fork(guidance=1.0, trainable=False)
    if list(candidate.module) != ["llm2action"]:
        raise ValueError("Only the native output head may be replaced")
    expected_state = candidate.module.state_dict()
    values = state.get("student")
    if not isinstance(values, dict) or set(values) != set(expected_state):
        raise ValueError("Candidate output-head state keys do not match the native model")
    for name, value in values.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected_state[name].shape
                or value.dtype != expected_state[name].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Invalid candidate output-head tensor: {name}")
    candidate.module.load_state_dict(values, strict=True)
    candidate.module.eval()
    evidence["loaded_state"] = {"format": state["format"], "version": state["version"],
                                "iteration": state["iteration"], "student_updates": state["student_updates"],
                                "tensor_keys": sorted(values), "strict": True, "all_finite": True,
                                "binding": "Temporary native-precision output-head binding during candidate calls only"}
    return candidate


def validate_baseline_repeat(reference_path, receipt, action_path):
    """Require untouched controls to reproduce a prior successful baseline run."""
    reference = json.loads(Path(reference_path).read_text())
    if reference.get("status") != "success":
        raise ValueError("Baseline reference report did not finish successfully")
    if reference["data"]["sha256"] != receipt["data"]["sha256"] or reference["requests"] != receipt["requests"]:
        raise ValueError("Baseline reference has different input observations or seeds")
    def inventory(entries):
        return {e["relative_path"]: e["sha256"] for e in entries}
    if inventory(reference["checkpoint_files"]) != inventory(receipt["checkpoint_files"]):
        raise ValueError("Baseline reference has a different native checkpoint")
    if inventory(reference["numerical_source"]["native"]["files"]) != inventory(receipt["numerical_source"]["native"]["files"]):
        raise ValueError("Native source changed since the reference baseline")
    if reference["numerical_source"]["adapter_loader"]["sha256"] != receipt["numerical_source"]["adapter_loader"]["sha256"]:
        raise ValueError("Native adapter loader changed since the reference baseline")
    if reference["environment"] != receipt["environment"]:
        raise ValueError("Native interpreter, packages or device identity changed since the reference baseline")
    def calls_by_identity(report):
        return {(call["arm"], call["request_index"], call["is_null_repeat"]): call
                for call in report["calls"] if call["arm"] != CANDIDATE}
    old_calls, new_calls = calls_by_identity(reference), calls_by_identity(receipt)
    if set(old_calls) != set(new_calls):
        raise ValueError("Untreated native request identities changed since the reference baseline")
    for key, before in old_calls.items():
        after = new_calls[key]
        for field in ("native_initial_noise", "native_condition_reference", "native_condition_mask",
                      "sampler_denoiser_callbacks", "model_velocity_branch_calls", "callback_timesteps"):
            if before[field] != after[field]:
                raise RuntimeError(f"Untreated native request {key} differs in {field}")
    old_artifact = reference["action_artifact"]
    if file_identity(old_artifact["path"])["sha256"] != old_artifact["sha256"]:
        raise ValueError("Baseline reference action artifact hash changed")
    matched = []
    with np.load(old_artifact["path"], allow_pickle=False) as old, np.load(action_path, allow_pickle=False) as new:
        for key in ("episode_id", "frame_index", "seed", "measured_action", "normalized_measured_action"):
            if not np.array_equal(old[key], new[key]):
                raise ValueError(f"Baseline reference differs in {key}")
        old_names, new_names = old["arm"].tolist(), new["arm"].tolist()
        expected = [arm["name"] for arm in receipt["arms"] if arm["name"] != CANDIDATE]
        if set(old_names) != set(expected):
            raise ValueError("Reference must contain exactly the untreated arms being repeated")
        for name in expected:
            before, after = old_names.index(name), new_names.index(name)
            for key in ("action", "normalized_action", "null_action", "null_normalized_action"):
                if not np.array_equal(old[key][before], new[key][after]):
                    raise RuntimeError(f"Untreated {name} {key} changed after candidate integration")
            matched.append(name)
    return {"reference_report": file_identity(reference_path), "reference_actions": old_artifact,
            "arms": matched, "bitexact": True, "raw_and_model_space_max_abs": 0.0,
            "scope": "Every untreated raw/model-space action and null output; identical checkpoint, inputs, native source, environment, noise, conditioning and callback evidence"}


def qualified(value):
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def source_inventory(package_root):
    package_root = Path(package_root).resolve()
    files = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}:
            files.append({**file_identity(path), "relative_path": str(path.relative_to(package_root))})
    try:
        head = subprocess.check_output(["git", "-C", str(package_root), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {"root": str(package_root), "git_head": head, "files": files,
            "note": "Hashes identify the working tree; git_head does not imply a clean checkout."}


def load_data(path):
    """Require complete identities and the same 32 by 8 DROID action contract."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"image", "state", "prompt", "episode_id", "frame_index", "measured_action"}
        if missing := required - set(archive.files):
            raise ValueError(f"Missing NPZ fields: {sorted(missing)}")
        data = {key: archive[key].copy() for key in required}
    image, state = data["image"], data["state"]
    count = len(image)
    if not count or image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be nonempty uint8 [N,H,W,3]")
    if min(image.shape[1:3]) < 1 or state.shape != (count, 8) or not np.isfinite(state).all():
        raise ValueError("Images must have positive size; state must be finite [N,8]")
    for key in ("prompt", "episode_id"):
        array = data[key]
        if array.shape != (count,) or array.dtype.kind not in "US":
            raise ValueError(f"{key} must contain one string per observation")
        data[key] = array.astype(str)
        if any(not value.strip() for value in data[key]):
            raise ValueError(f"{key} cannot contain empty strings")
    frames = data["frame_index"]
    if frames.shape != (count,) or frames.dtype.kind not in "iu" or (frames < 0).any():
        raise ValueError("frame_index must contain one nonnegative integer per observation")
    if len(set(zip(data["episode_id"].tolist(), frames.tolist()))) != count:
        raise ValueError("Duplicate episode_id/frame_index observations are not independent samples")
    action = data["measured_action"]
    if action.shape != (count, 32, 8) or action.dtype.kind not in "fiu" or not np.isfinite(action).all():
        raise ValueError("measured_action must contain finite raw DROID action chunks [N,32,8]")
    return data


def split_evidence(data, train_path=None):
    episodes = sorted(set(data["episode_id"].tolist()))
    result = {"evaluation_episode_ids": episodes, "episode_disjoint_verified": False,
              "pretrained_checkpoint_contamination": "unknown"}
    if train_path is not None:
        train = load_data(train_path)
        train_episodes = sorted(set(train["episode_id"].tolist()))
        if overlap := set(episodes) & set(train_episodes):
            raise ValueError(f"Training and evaluation episodes overlap: {sorted(overlap)}")
        result.update(episode_disjoint_verified=True, train_data=file_identity(train_path),
                      train_episode_ids=train_episodes)
    return result


def finite_stats(array):
    array = np.asarray(array)
    mask = np.isfinite(array)
    finite = array[mask].astype(np.float64)
    return {"elements": int(array.size), "finite_elements": int(mask.sum()),
            "finite_fraction": float(mask.mean()), "all_finite": bool(mask.all()),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "max_abs": float(np.abs(finite).max()) if finite.size else None}


def error_metrics(prediction, reference, episode_ids=None):
    """Average squared/absolute errors within episodes, then weight episodes equally."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[1] != 32:
        raise ValueError("Error metrics require matching [N,32,D] arrays")
    if not prediction.size or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("Error metrics require nonempty finite arrays")
    delta = prediction.astype(np.float64) - reference.astype(np.float64)
    episodes = (np.arange(len(delta)).astype(str) if episode_ids is None else np.asarray(episode_ids).astype(str))
    if episodes.shape != (len(delta),):
        raise ValueError("episode_ids must identify every metric request")
    unique, inverse, counts = np.unique(episodes, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(unique) * counts[inverse])

    def reduced(values, axes, weighted=True):
        square = np.mean(np.square(values), axis=axes)
        absolute = np.mean(np.abs(values), axis=axes)
        if weighted:
            square = np.average(square, weights=weights, axis=0)
            absolute = np.average(absolute, weights=weights, axis=0)
        return {"rmse": np.sqrt(square).tolist(), "mae": absolute.tolist()}

    per_episode = []
    for episode in unique:
        values = delta[episodes == episode]
        per_episode.append({"episode_id": str(episode), "requests": len(values),
                            "rmse": float(np.sqrt(np.square(values).mean())),
                            "mae": float(np.abs(values).mean())})
    return {"all_32": reduced(delta, (1, 2)), "first_step": reduced(delta[:, :1], (1, 2)),
            "per_horizon": reduced(delta, (2,)),
            "per_request": reduced(delta, (1, 2), weighted=False),
            "per_channel": reduced(delta, (1,)), "per_episode": per_episode,
            "episode_count": len(unique), "request_count": len(delta)}


def comparison_metrics(raw, normalized, reference_raw, reference_normalized, episode_ids=None, normalizer_present=True):
    normalized_unit = ("native configured model-space action normalization" if normalizer_present else
                       "mixed units: joint radians and native 1-gripper scalar; normalizer=None, no standardization")
    return {"normalized_action": {"unit": normalized_unit,
                                  **error_metrics(normalized, reference_normalized, episode_ids)},
            "joint_position": {"unit": "radian", "channels": list(range(7)),
                               **error_metrics(raw[..., :7], reference_raw[..., :7], episode_ids)},
            "gripper_position": {"unit": "published DROID scalar, dimensionless",
                                 "channels": [7],
                                 **error_metrics(raw[..., 7:], reference_raw[..., 7:], episode_ids)}}


def validate_call(call, arm):
    callbacks = call["sampler_denoiser_callbacks"]
    branches = call["model_velocity_branch_calls"]
    if call["sampler_calls"] != 1 or call["preparation_calls"] != 1:
        raise RuntimeError(f"Expected one native preparation and sampler call: {call}")
    if callbacks != arm["steps"] or (arm["max_callbacks"] is not None and callbacks > arm["max_callbacks"]):
        raise RuntimeError(f"{arm['name']} executed {callbacks} callbacks; expected {arm['steps']}")
    if branches != callbacks * (1 if arm["guidance"] == 1.0 else 2):
        raise RuntimeError(f"{arm['name']} has unexpected CFG branch count: {branches}/{callbacks}")
    times = np.asarray(call["callback_timesteps"], dtype=np.float64)
    if times.shape != (callbacks,) or not np.isfinite(times).all():
        raise RuntimeError("Incomplete native callback timestep evidence")
    if arm["sampler"] == "fixed":
        expected = np.asarray(RF_GRID[:-1]) * call["num_train_timesteps"]
        if not np.allclose(times, expected, rtol=0, atol=1e-4):
            raise RuntimeError(f"Native fixed sampler used unexpected times: {times.tolist()}")
        if call["sample_type"] != "sde":
            raise RuntimeError("Expected the native SDE fixed sampler")


def tensor_identity(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest}


class NativeCapture:
    """Observe original methods, restoring every temporary override in finally."""

    def __init__(self, model, sampler, method):
        self.model, self.sampler, self.method = model, sampler, method
        self.prepared = None
        self.latents = None
        self.initial = None
        self.times = []
        self.record = {"sampler_calls": 0, "preparation_calls": 0,
                       "sampler_denoiser_callbacks": 0, "model_velocity_branch_calls": 0,
                       "sampler_class": qualified(sampler),
                       "sample_type": getattr(sampler, "sample_type", None),
                       "rf_grid": getattr(sampler, "t_list", None),
                       "num_train_timesteps": float(model.config.rectified_flow_inference_config.num_train_timesteps)}
        config = getattr(sampler, "cfg", None)
        if config is not None:
            self.record["sampler_config"] = {key: getattr(config, key) for key in
                                             ("num_train_timesteps", "shift", "use_dynamic_shifting")}

    @contextmanager
    def instrument(self):
        patches = []

        def patch(owner, name, replacement):
            own = name in vars(owner)
            original = getattr(owner, name)
            patches.append((owner, name, own, original))
            setattr(owner, name, replacement(original))

        def preparing(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                self.record["preparation_calls"] += 1
                self.prepared = original(*args, **kwargs)
                return self.prepared
            return wrapped

        def velocity(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                self.record["model_velocity_branch_calls"] += 1
                return original(*args, **kwargs)
            return wrapped

        def sampling(original):
            @functools.wraps(original)
            def wrapped(sampler, callback, noise, *args, **kwargs):
                if sampler is not self.sampler:
                    return original(sampler, callback, noise, *args, **kwargs)
                self.record["sampler_calls"] += 1
                # Retain unmodified tensors. Native samplers update latents out of
                # place; identity is read after generation, outside its hot loop.
                self.initial = list(noise)
                self.record["effective_sampler_arguments"] = {
                    key: kwargs.get(key) for key in ("num_steps", "shift", "seed")}
                self.record["native_conditioning_passed"] = (
                    kwargs.get("condition_reference") is not None and kwargs.get("condition_mask") is not None)

                @functools.wraps(callback)
                def counted(*values, **named):
                    self.record["sampler_denoiser_callbacks"] += 1
                    timestep = values[1] if len(values) > 1 else named["timestep"]
                    self.times.append(timestep.detach())
                    return callback(*values, **named)

                started = time.perf_counter()
                try:
                    self.latents = original(sampler, counted, noise, *args, **kwargs)
                    return self.latents
                finally:
                    self.record["sampler_host_seconds"] = time.perf_counter() - started
            return wrapped

        try:
            patch(self.model, "_prepare_inference_data", preparing)
            patch(self.model, "_get_velocity", velocity)
            patch(type(self.sampler), self.method, sampling)
            yield self
        finally:
            for owner, name, own, original in reversed(patches):
                if own:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)

    def finalize(self):
        self.record["callback_timesteps"] = [float(t.float().cpu().item()) for t in self.times]
        self.record["native_initial_noise"] = [tensor_identity(t) for t in self.initial or []]
        if self.prepared is not None:
            self.record["native_condition_reference"] = [tensor_identity(t) for t in self.prepared[5]]
            self.record["native_condition_mask"] = [tensor_identity(t) for t in self.prepared[6]]
        return self.record


def normalize_recorded(action, record, device):
    import torch

    value = torch.as_tensor(action, device=device, dtype=torch.float32).clone()
    value[:, -1] = 1.0 - value[:, -1]
    normalizer = record.action_normalizer
    if normalizer is not None:
        value = normalizer.normalize_action(value)
    return value.detach().float().cpu().numpy().copy()


def normalizer_metadata(record):
    normalizer = record.action_normalizer
    result = {"raw_action_dim": record.raw_action_dim,
              "qualified_class": qualified(normalizer) if normalizer is not None else None}
    for key in ("offset", "scale", "forward_clamp", "forward_clamp_mask"):
        value = getattr(normalizer, key, None)
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        result[key] = value
    return result


def run_native(adapter, data, index, seed, arm, sampler, *, bind_heads=False):
    import torch
    from cosmos_framework.data.generator.action.action_processing import get_action_processing_records
    from cosmos_framework.scripts.action_policy_server_robolab import _build_data_batch_from_sample

    service, model = adapter.service, adapter.model
    state = data["state"][index].astype(np.float32)
    obs = {"observation/image": data["image"][index].copy(),
           "observation/joint_position": state[:7], "observation/gripper_position": state[7:],
           "prompt": str(data["prompt"][index])}
    # Rebuild raw data per call; no cached model state, adapter mask, or synthetic
    # flattened torch.randn noise enters native generation.
    raw_batch = _build_data_batch_from_sample(service._build_sample(obs))
    action_record = get_action_processing_records(raw_batch)[0]
    capture = NativeCapture(model, sampler, "forward" if arm["sampler"] == "unipc" else "__call__")
    original_heads = {name: getattr(model.net, name) for name in adapter.module}
    started = time.perf_counter()
    with service._lock, torch.inference_mode(), capture.instrument(), (
            adapter._bound_heads() if bind_heads else nullcontext()):
        samples = model.generate_samples_from_batch(
            raw_batch, sampler=sampler, seed=[seed], guidance=arm["guidance"],
            num_steps=arm["steps"], shift=5.0, guidance_interval=None,
            normalize_cfg=False, skip_text_tokens_for_cfg=False,
        )
    if any(getattr(model.net, name) is not original for name, original in original_heads.items()):
        raise RuntimeError("Native head binding was not restored after generation")
    generation_host_seconds = time.perf_counter() - started
    call = capture.finalize()
    validate_call(call, arm)
    if arm["sampler"] == "fixed" and not call["native_conditioning_passed"]:
        raise RuntimeError("Native fixed sampling did not receive its condition reference and mask")
    plans, clean = capture.prepared[:2]
    if len(plans) != 1 or not plans[0].has_action or plans[0].has_sound:
        raise ValueError("Expected one native DROID vision/action sample, without sound")
    if int(action_record.raw_action_dim) != 8 or len(samples.get("action", [])) != 1:
        raise ValueError("Native output does not satisfy the DROID 8-channel action contract")
    action_shape = tuple(clean.x0_tokens_action[0].shape)
    # DROID has [vision | action] with no sound. Inspect the sampler's exact
    # model-space channels before native denormalization; never renormalize a
    # decoded prediction through a potentially lossy forward clamp.
    action_numel = clean.x0_tokens_action[0].numel()
    model_action = capture.latents[0][-action_numel:].reshape(action_shape)
    normalized = model_action[service.cfg.history_length:, :8].detach().float().cpu().numpy().copy()
    raw = samples["action"][0][service.cfg.history_length:, :8].detach().float().cpu().numpy().copy()
    raw[:, -1] = 1.0 - raw[:, -1]
    recorded_normalized = normalize_recorded(data["measured_action"][index], action_record, model_action.device)
    if raw.shape != (32, 8) or normalized.shape != (32, 8):
        raise ValueError(f"Unexpected native output shapes: {raw.shape}, {normalized.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise FloatingPointError("Nonfinite native DROID actions")
    call.update(generation_host_seconds=generation_host_seconds,
                trained_head_binding=bind_heads, native_heads_restored=True,
                raw_action_stats=finite_stats(raw), normalized_action_stats=finite_stats(normalized),
                output_shapes={key: [list(value.shape) for value in values] for key, values in samples.items()},
                normalizer=normalizer_metadata(action_record))
    return raw, normalized, recorded_normalized, call


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local pinned native DROID checkpoint directory")
    parser.add_argument("--data", type=Path, required=True, help="Prepared NPZ with measured_action and episode/frame IDs")
    parser.add_argument("--train-data", type=Path, help="Reject overlapping evaluation/training episode IDs")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1301, 1302])
    parser.add_argument("--include-fixed-cfg1", action="store_true")
    parser.add_argument("--candidate-training-state", type=Path,
                        help="Trusted local v2 training.pt; adds a trained output-head UniPC4 CFG1 arm")
    parser.add_argument("--candidate-declaration", type=Path,
                        help="Strict candidate.json declaration (default: sibling of training.pt)")
    parser.add_argument("--baseline-report", type=Path,
                        help="Require all untreated arms to reproduce a prior successful report bitexact")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds must be distinct integers in [0, 2**32)")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This bounded evaluator supports one process/GPU; launch arms sequentially")
    if not args.checkpoint.is_dir():
        parser.error("checkpoint must be an existing local directory")
    if args.candidate_declaration is not None and args.candidate_training_state is None:
        parser.error("--candidate-declaration requires --candidate-training-state")
    if args.candidate_training_state is not None and args.train_data is None:
        parser.error("--candidate-training-state requires --train-data")
    data = load_data(args.data)
    split = split_evidence(data, args.train_data)
    args.output.mkdir(parents=True, exist_ok=False)
    arm_specs = arms(args.include_fixed_cfg1, args.candidate_training_state is not None)
    requests = [{"data_index": index, "episode_id": str(data["episode_id"][index]),
                 "frame_index": int(data["frame_index"][index]), "seed": seed,
                 "prompt": str(data["prompt"][index])}
                for index in range(len(data["image"])) for seed in args.seeds]
    receipt = {
        "schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Paired open-loop DROID action diagnostics on recorded observations; task success is unmeasured.",
        "interpretation": [
            "UniPC4 CFG3 is the primary operational baseline on the same checkpoint and task.",
            "UniPC30 CFG3 is a higher-step diagnostic, not an assumed better teacher or policy.",
            "Lower recorded-action error or higher-step agreement does not establish higher policy quality.",
            "All control arms use original checkpoint weights; fixed4 is a sampler control, not a trained DMD2 model. An optional candidate replaces only its declared output head during its own arm.",
            "Evaluation episodes are held out of the supplied train-data only when split verification passes.",
            "When native normalizer=None, the backward-compatible normalized_action field contains model-space mixed units: joint radians plus native 1-gripper scalar. It is not standardized or unitless. Primary physical comparisons keep joints and gripper separate.",
        ],
        "checkpoint": str(args.checkpoint.resolve()), "data": file_identity(args.data),
        "split": split, "arms": arm_specs, "requests": requests,
        "seed_semantics": "Explicit per-sample native diffusion seeds; no service RNG indirection. Native preparation creates the noise.",
        "count_semantics": "Per-request sampler callbacks and separate actual _get_velocity CFG branches; one process, no padding.",
        "timing_semantics": "Instrumented host wall times without added CUDA synchronization. Generation includes fresh native conditioning; no synchronized speed claim.",
        "normalization_semantics": "Exact final native action latent versus recorded raw action transformed by native gripper 1-x then native normalizer, including any configured forward clamp.",
        "metric_aggregation": "Primary aggregate: mean squared/absolute error over each episode's observations, seeds, horizons and included channels, then equal weight across episodes; RMSE takes the square root after averaging. All requests, per-episode, per-horizon and first-step metrics retained. Normalized 8-channel error, joint radians and gripper scalar are separate. No temporal alignment or angle wrapping.",
        "null_repeat": {"scope": "First observation and first seed repeated once per arm through fresh native preprocessing", "request_index": 0},
        "calls": [],
    }
    dataset_path = args.data.parent / "dataset.json"
    if dataset_path.is_file():
        receipt["dataset_provenance"] = file_identity(dataset_path)
    path = args.output / "report.json"
    started = time.perf_counter()
    try:
        import cosmos_framework
        from instinct_compress.models.cosmos3 import Cosmos3Adapter

        receipt["numerical_source"] = {
            "capture_timing": "before model construction",
            "wrapper": file_identity(__file__),
            "adapter_loader": file_identity(Path(__file__).resolve().parents[1] / "instinct_compress/models/cosmos3.py"),
            "native": source_inventory(Path(cosmos_framework.__file__).parent),
        }
        receipt["checkpoint_files"] = [
            {**file_identity(file), "relative_path": str(file.relative_to(args.checkpoint))}
            for file in sorted(args.checkpoint.rglob("*")) if file.is_file()
        ]
        receipt["checkpoint_hash_timing"] = "All native checkpoint files hashed before model construction; config/source hashes checked again after inference."
        candidate_evidence = None
        if args.candidate_training_state is not None:
            declaration_path = args.candidate_declaration or args.candidate_training_state.parent / "candidate.json"
            candidate_evidence = validate_candidate_declaration(
                args.candidate_training_state, declaration_path, receipt["checkpoint_files"], split.get("train_data"))
            receipt["candidate"] = candidate_evidence
        write_json(path, receipt)
        adapter = Cosmos3Adapter(args.checkpoint, output_dir=args.output / "native_service",
                                 image_height=data["image"].shape[1], image_width=data["image"].shape[2])
        import torch
        from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
        from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

        service, model = adapter.service, adapter.model
        if (service.cfg.domain_name != "droid_lerobot" or service.cfg.action_space != "joint_pos"
                or service.cfg.action_dim != 8 or service.cfg.action_chunk_size != 32):
            raise ValueError("Expected a native 32-step DROID joint_pos policy with 8 action channels")
        if not isinstance(model.sampler, UniPCSampler):
            raise ValueError("The supplied original checkpoint must use the native UniPC baseline sampler")
        if model.parallel_dims is not None and (model.parallel_dims.cfgp_enabled or model.parallel_dims.cp_enabled):
            raise ValueError("This evaluator requires native single-process, unsharded CFG execution")
        fixed = FixedStepSampler(RF_GRID.copy(), sample_type="sde", num_train_timesteps=float(
            model.config.rectified_flow_inference_config.num_train_timesteps))
        candidate = (load_candidate(adapter, args.candidate_training_state, candidate_evidence)
                     if candidate_evidence is not None else None)
        receipt["environment"] = {"python": sys.version, "torch": torch.__version__,
                                  "torch_cuda": torch.version.cuda,
                                  "device": torch.cuda.get_device_name(torch.cuda.current_device())}
        for package in ("numpy", "transformers", "diffusers", "safetensors"):
            receipt["environment"][package] = importlib.metadata.version(package)
        receipt["native_service"] = {"qualified_model_class": qualified(model),
                                     "qualified_service_class": qualified(service),
                                     "config": vars(service.cfg),
                                     "head_binding": "Controls retain original native heads; optional candidate binds its verified output head temporarily",
                                     "use_torch_compile": False, "guardrails": False}
        raw_values, normalized_values, measured_values, null_raw, null_normalized = [], [], [], [], []
        paired_noise, paired_reference, paired_mask = {}, {}, {}
        normalizer = None
        for arm in arm_specs:
            raw_arm, normalized_arm = [], []
            sampler = model.sampler if arm["sampler"] == "unipc" else fixed
            active_adapter = candidate if arm["name"] == CANDIDATE else adapter
            for request_index, request in enumerate(requests):
                print(f"{arm['name']} request {request_index + 1}/{len(requests)} "
                      f"frame={request['frame_index']} seed={request['seed']}", flush=True)
                raw, normalized, measured, call = run_native(
                    active_adapter, data, request["data_index"], request["seed"], arm, sampler,
                    bind_heads=arm["name"] == CANDIDATE)
                call.update(arm=arm["name"], request_index=request_index, is_null_repeat=False, status="success")
                receipt["calls"].append(call)
                for field, paired in (("native_initial_noise", paired_noise),
                                      ("native_condition_reference", paired_reference),
                                      ("native_condition_mask", paired_mask)):
                    if request_index in paired and paired[request_index] != call[field]:
                        raise RuntimeError(f"Native {field} differs between paired arms for request {request_index}")
                    paired[request_index] = call[field]
                if normalizer is not None and normalizer != call["normalizer"]:
                    raise RuntimeError("Native action normalization differs across paired requests")
                normalizer = call["normalizer"]
                if arm["name"] == BASELINE:
                    measured_values.append(measured)
                elif not np.array_equal(measured_values[request_index], measured):
                    raise RuntimeError("Normalized recorded targets differ across arms")
                raw_arm.append(raw)
                normalized_arm.append(normalized)
                write_json(path, receipt)
            # A fresh raw sample and native conditioning pass are part of the null.
            first = requests[0]
            raw, normalized, _, call = run_native(active_adapter, data, first["data_index"], first["seed"], arm, sampler,
                                                  bind_heads=arm["name"] == CANDIDATE)
            call.update(arm=arm["name"], request_index=0, is_null_repeat=True, status="success")
            receipt["calls"].append(call)
            for field, paired in (("native_initial_noise", paired_noise),
                                  ("native_condition_reference", paired_reference),
                                  ("native_condition_mask", paired_mask)):
                if call[field] != paired[0]:
                    raise RuntimeError(f"Null-repeat native {field} differs")
            if not np.array_equal(raw_arm[0], raw) or not np.array_equal(normalized_arm[0], normalized):
                raise RuntimeError(f"{arm['name']} null repeat differs: raw max abs "
                                   f"{float(np.max(np.abs(raw_arm[0] - raw)))}")
            null_raw.append(raw)
            null_normalized.append(normalized)
            raw_values.append(np.stack(raw_arm))
            normalized_values.append(np.stack(normalized_arm))
            # Persist completed arms even if a later arm fails.
            np.savez_compressed(args.output / "actions.npz",
                arm=np.asarray([spec["name"] for spec in arm_specs[:len(raw_values)]]),
                action=np.stack(raw_values), normalized_action=np.stack(normalized_values),
                measured_action=np.stack([data["measured_action"][r["data_index"]] for r in requests]),
                normalized_measured_action=np.stack(measured_values),
                episode_id=np.asarray([r["episode_id"] for r in requests]),
                frame_index=np.asarray([r["frame_index"] for r in requests]),
                seed=np.asarray([r["seed"] for r in requests]),
                prompt=np.asarray([r["prompt"] for r in requests]),
                null_action=np.stack(null_raw), null_normalized_action=np.stack(null_normalized),
                horizon_index=np.arange(32))
            write_json(path, receipt)
        recorded = np.stack([data["measured_action"][r["data_index"]] for r in requests])
        normalized_recorded = np.stack(measured_values)
        receipt["native_action_normalizer"] = normalizer
        names = [spec["name"] for spec in arm_specs]
        receipt["metrics"] = {}
        for index, arm in enumerate(arm_specs):
            references = {"recorded_action": (recorded, normalized_recorded),
                          BASELINE: (raw_values[names.index(BASELINE)], normalized_values[names.index(BASELINE)]),
                          DIAGNOSTIC: (raw_values[names.index(DIAGNOSTIC)], normalized_values[names.index(DIAGNOSTIC)])}
            receipt["metrics"][arm["name"]] = {
                "versus": {name: comparison_metrics(raw_values[index], normalized_values[index], raw, normalized,
                                                    [request["episode_id"] for request in requests],
                                                    normalizer_present=normalizer["qualified_class"] is not None)
                           for name, (raw, normalized) in references.items()},
                "raw_finite": finite_stats(raw_values[index]),
                "normalized_finite": finite_stats(normalized_values[index]), "null_max_abs": 0.0,
            }
        source = receipt["numerical_source"]
        tracked = [source["wrapper"], source["adapter_loader"], *source["native"]["files"]]
        tracked.extend(entry for entry in receipt["checkpoint_files"] if Path(entry["path"]).suffix == ".json")
        tracked.append(receipt["data"])
        if "train_data" in split:
            tracked.append(split["train_data"])
        if "dataset_provenance" in receipt:
            tracked.append(receipt["dataset_provenance"])
        if candidate_evidence is not None:
            tracked.extend(candidate_evidence["files"])
        changed = [entry["path"] for entry in tracked if file_identity(entry["path"])["sha256"] != entry["sha256"]]
        receipt["source_and_config_postcheck"] = {"files_checked": len(tracked), "changed": changed}
        if changed:
            raise RuntimeError(f"Numerical sources/configs changed during evaluation: {changed}")
        receipt["paired_native_noise_and_condition_verified"] = True
        receipt["action_artifact"] = file_identity(args.output / "actions.npz")
        if args.baseline_report is not None:
            receipt["untreated_baseline_repeat"] = validate_baseline_repeat(
                args.baseline_report, receipt, args.output / "actions.npz")
        receipt["status"] = "success"
    except BaseException as error:
        receipt.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(finished_at=datetime.now(timezone.utc).isoformat(),
                       total_host_seconds=time.perf_counter() - started)
        write_json(path, receipt)


if __name__ == "__main__":
    main()
