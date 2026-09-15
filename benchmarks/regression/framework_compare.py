"""Portable reproduction of the six measured foreign-framework Thor cells.

Planning and reporting are CPU-only. ``prepare`` downloads exact public Hub
revisions. ``run`` explicitly executes one fresh, bounded, locked GPU process.
The original schedules and continuation-only selections are fixed in the
catalog; these are latency screens, not task-quality certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time
import traceback


DATA = Path(__file__).parent / "fixtures/frameworks"
SCHEMA = "instinctflash.framework_comparison.v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_new(path, value, *, raw=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value if raw else encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def new_directory(path):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), f"destination already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def catalog():
    value = json.loads((DATA / "catalog.json").read_text())
    require(value["schema"] == SCHEMA, "unsupported framework catalog")
    require(sha(DATA / "sources.json") == value["source_inventory_sha256"], "source inventory changed")
    require(sha(DATA / value["startup_patch_file"]) == value["startup_patch_sha256"], "startup patch changed")
    require(sha(DATA / "omni_thor_startup_patch.json") == value["historical_startup_patch_sha256"], "historical startup patch changed")
    return value


def plan(cell_id):
    value = catalog()
    rows = [row for row in value["cells"] if row["id"] == cell_id]
    require(len(rows) == 1, "cell has no measured historical route; use list to inspect empty cells")
    return {"schema": SCHEMA, "cell": rows[0], "catalog_sha256": sha(DATA / "catalog.json"),
            "source_inventory_sha256": value["source_inventory_sha256"],
            "fixture_sha256": value["fixture_sha256"],
            "packages": value["packages"][rows[0]["framework"]],
            "historical_protocol": True, "task_quality_certified": False,
            "scope": "Local observation-to-CPU-actions wall time; Omni includes worker IPC. No network or simulator."}


class Inputs:
    """The safe archive is a byte-equivalent decoding of the old JPEG fixture."""

    def __init__(self, path):
        import numpy as np
        require(sha(path) == catalog()["fixture_sha256"], "recorded fixture changed")
        with np.load(path, allow_pickle=False) as data:
            self.frames, self.feedback = data["frames"].copy(), data["feedback"].copy()
        require(self.frames.dtype == np.uint8 and self.frames.shape[:2] == (13, 3)
                and self.frames.ndim == 5 and self.frames.shape[-1] == 3, "invalid cameras")
        require(self.feedback.dtype == np.float32 and self.feedback.shape == (8, 16, 2, 16)
                and np.isfinite(self.feedback).all(), "invalid feedback")


def request(cell, inputs, i):
    """Original per-call camera/state/prompt/seed contract, before tensor conversion."""
    import numpy as np
    from PIL import Image
    family = cell["family"]
    history = family in {"va", "dreamzero"}
    episode, cycle = divmod(i, 3) if history else (i, 0)
    prompt = "pick up the object" if (episode * 3 if history else i) < 8 else "place the object down"
    images = inputs.frames[1 + i % 12]
    common = {"i": i, "episode": episode, "cycle": cycle, "prompt": prompt,
              "reset": not history or cycle == 0, "seed": 1300 + cycle if history else 707 + i}
    if family == "pi05":
        obs = {"observation.images.image": images[0].transpose(2, 0, 1).astype(np.float32) / 255,
               "observation.images.image2": images[1].transpose(2, 0, 1).astype(np.float32) / 255,
               "observation.state": np.zeros(8, np.float32)}
    elif family == "groot":
        state = np.zeros(17, np.float32)
        state[3:9] = [1, 0, 0, 0, 1, 0]
        obs = {"observation.images." + key: images[j].transpose(2, 0, 1).copy()
               for j, key in enumerate(("exterior_image_1_left", "wrist_image_left"))}
        obs.update({"observation.state": state, "task": prompt})
    elif family in {"edge", "nano"}:
        state = np.full(8, .01 * (i % 3), np.float32)
        obs = {"observation/image": np.asarray(Image.fromarray(images[0]).resize((640, 540))),
               "observation/joint_position": state[:7], "observation/gripper_position": state[7:], "prompt": prompt}
        # The original harness deliberately reconstructs RNG(0) on EVERY call.
        common["seed"] = int(np.random.default_rng(0).integers(0, 2**31))
    elif family == "va":
        indices = [0] if cycle == 0 else list(range(1, 5)) if cycle == 1 else list(range(5, 13))
        obs = {"frames": inputs.frames[indices].copy(), "indices": indices,
               "feedback": None if cycle == 0 else inputs.feedback[(cycle - 1) % len(inputs.feedback)].copy()}
    elif family == "dreamzero":
        indices = [0] if cycle == 0 else [1, 2, 3, 4]
        keys = ("observation/exterior_image_0_left", "observation/exterior_image_1_left", "observation/wrist_image_left")
        obs = {key: inputs.frames[indices[0], camera].copy() if cycle == 0
               else inputs.frames[indices, camera].copy() for camera, key in enumerate(keys)}
        obs.update({"observation/joint_position": np.zeros(7, np.float32),
                    "observation/cartesian_position": np.zeros(6, np.float32),
                    "observation/gripper_position": np.zeros(1, np.float32),
                    "prompt": prompt, "session_id": f"benchmark-{episode}"})
        common["seed"] = 1140  # Upstream scheduler default, not a client RNG reset.
    else:
        raise ValueError("unknown family")
    return {**common, "observation": obs}


def request_hash(value):
    import numpy as np
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(f"list:{len(item)}".encode())
            for child in item:
                add(child)
        elif isinstance(item, np.ndarray):
            h.update(encoded([item.dtype.str, item.shape]))
            h.update(np.ascontiguousarray(item).tobytes())
        else:
            h.update(encoded([type(item).__name__, item]))
    add(value)
    return h.hexdigest()


def asset_inventory(path, filenames=None):
    root = Path(path).resolve(strict=True)
    paths = [root / name for name in filenames] if filenames is not None else sorted(root.rglob("*"))
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in paths if p.is_file() and ".cache" not in p.relative_to(root).parts}


def prepare_auxiliary(cell, root, *, cache_dir=None, local_files_only=False):
    """Bind small processor assets; aliases live only in this new owned cache."""
    from huggingface_hub import hf_hub_download
    hub = Path(root) / "auxiliary_hub"
    hub.mkdir()
    records = []
    for repo, binding in cell["auxiliary_repositories"].items():
        model = hub / ("models--" + repo.replace("/", "--"))
        snapshot = model / "snapshots" / binding["revision"]
        for filename, expected in binding["files"].items():
            source = Path(hf_hub_download(repo_id=repo, filename=filename, revision=binding["revision"],
                                          cache_dir=cache_dir, local_files_only=local_files_only)).resolve(strict=True)
            require(source.stat().st_size == expected["bytes"] and sha(source) == expected["sha256"], "auxiliary asset differs")
            destination = snapshot / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Existing content is never written; each alias is exclusive and hash-bound.
            destination.symlink_to(source)
            records.append({"repo": repo, "revision": binding["revision"], "filename": filename,
                            "path": str(destination), "source": str(source), **expected})
        (model / "refs").mkdir()
        write_new(model / "refs/main", binding["revision"].encode(), raw=True)
    return {"cache": str(hub), "files": records}


def prepare(cell_id, output, *, cache_dir=None, local_files_only=False):
    value = plan(cell_id)
    root = new_directory(output)
    write_new(root / "plan.json", value)
    fixture = DATA.parent / "recorded_inputs_v1.npz"
    require(sha(fixture) == value["fixture_sha256"], "packaged fixture changed")
    write_new(root / "inputs.npz", fixture.read_bytes(), raw=True)
    result = {"schema": SCHEMA, "status": "failed", "plan_sha256": sha(root / "plan.json"), "assets": {}}
    try:
        from huggingface_hub import snapshot_download
        for name, spec in value["cell"]["assets"].items():
            path = Path(snapshot_download(repo_id=spec["model_id"], revision=spec["revision"],
                                          cache_dir=cache_dir, local_files_only=local_files_only,
                                          allow_patterns=list(spec["files"]) if "files" in spec else None)).resolve(strict=True)
            require(path.name == spec["revision"], "asset did not resolve the pinned snapshot")
            inventory = asset_inventory(path, spec.get("files"))
            require("files" not in spec or inventory == spec["files"], "exact auxiliary tokenizer files differ")
            if "files" in spec:
                view = root / "asset_views" / name / spec["revision"]
                view.mkdir(parents=True, exist_ok=False)
                for filename in spec["files"]:
                    target = view / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to((path / filename).resolve(strict=True))
                path = view
            result["assets"][name] = {"model_id": spec["model_id"], "revision": spec["revision"], "path": str(path), "files": inventory}
        result["auxiliary"] = prepare_auxiliary(value["cell"], root, cache_dir=cache_dir, local_files_only=local_files_only)
        if value["cell"]["family"] == "va":
            write_new(root / "va_weight_mapping.json", audit_va_weights(result["assets"]["converted"]["path"],
                                                                        result["assets"]["checkpoint"]["path"]))
        result["status"] = "prepared"
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_new(root / "preparation.json", result)
    return result


def audit_va_weights(converted, original):
    """839 exact mapped tensors; disclose the two native-only legacy tensors."""
    from contextlib import ExitStack
    import mmap
    import struct

    def header(path):
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            require(0 < length < path.stat().st_size, "invalid safetensors header")
            return 8 + length, json.loads(stream.read(length))

    def tensor_sha(mm, base, entry):
        start, end = entry["data_offsets"]
        require(0 <= start <= end <= len(mm) - base, "invalid tensor byte interval")
        h = hashlib.sha256()
        for offset in range(base + start, base + end, 8 * 1024 * 1024):
            h.update(mm[offset:min(offset + 8 * 1024 * 1024, base + end)])
        return h.hexdigest()

    original = Path(original) / "transformer"
    converted = Path(converted) / "model.safetensors"
    index = json.loads((original / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    rows = []
    with ExitStack() as stack:
        def mapped(path):
            base, entries = header(path)
            stream = stack.enter_context(path.open("rb"))
            return base, entries, stack.enter_context(mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ))
        cb, ch, cm = mapped(converted)
        sources = {name: mapped(original / name) for name in set(index.values())}
        for key, entry in ch.items():
            if key == "__metadata__":
                continue
            require(key.startswith("transformer."), "unknown converted tensor namespace")
            native = key.removeprefix("transformer.")
            base, entries, mm = sources[index[native]]
            require(entry["dtype"] == entries[native]["dtype"] and entry["shape"] == entries[native]["shape"],
                    f"converted tensor layout differs: {key}")
            left, right = tensor_sha(cm, cb, entry), tensor_sha(mm, base, entries[native])
            require(left == right, f"converted tensor bytes differ: {key}")
            rows.append({"key": key, "native_key": native, "sha256": left})
    unused = sorted(set(index) - {row["native_key"] for row in rows})
    require(len(rows) == 839 and len(index) == 841 and unused == ["patch_embedding.bias", "patch_embedding.weight"],
            "VA conversion inventory differs from historical mapping")
    return {"status": "passed", "shared_tensors_bitexact": True, "tensor_count": 839,
            "native_count": 841, "native_only": unused, "rows": rows, "inference_equivalence": False}


def load_prepared(root, *, verify_assets=True):
    root = Path(root).resolve(strict=True)
    value = json.loads((root / "plan.json").read_text())
    require(value == plan(value["cell"]["id"]), "prepared plan changed")
    prepared = json.loads((root / "preparation.json").read_text())
    require(prepared["status"] == "prepared" and prepared["plan_sha256"] == sha(root / "plan.json"), "preparation failed or changed")
    require(sha(root / "inputs.npz") == value["fixture_sha256"], "prepared input changed")
    require(set(prepared["assets"]) == set(value["cell"]["assets"]), "wrong asset set")
    for key, binding in prepared["assets"].items():
        expected = value["cell"]["assets"][key]
        require(all(binding[k] == expected[k] for k in ("model_id", "revision")), "asset declaration differs")
        require(Path(binding["path"]).name == binding["revision"], "asset revision differs")
        if verify_assets:
            require(asset_inventory(binding["path"], expected.get("files")) == binding["files"], f"asset bytes changed: {key}")
        if "files" in expected:
            require(binding["files"] == expected["files"], "tokenizer differs from public bound files")
    auxiliary = prepared["auxiliary"]
    require(Path(auxiliary["cache"]) == root / "auxiliary_hub", "auxiliary cache must be owned by preparation")
    expected_rows = {(repo, binding["revision"], name): spec for repo, binding in value["cell"]["auxiliary_repositories"].items()
                     for name, spec in binding["files"].items()}
    require(len(auxiliary["files"]) == len(expected_rows), "auxiliary inventory size differs")
    observed = set()
    for row in auxiliary["files"]:
        key = row["repo"], row["revision"], row["filename"]
        require(key in expected_rows and key not in observed, "unexpected or duplicated auxiliary asset")
        observed.add(key)
        spec = expected_rows[key]
        path = Path(auxiliary["cache"]) / ("models--" + key[0].replace("/", "--")) / "snapshots" / key[1] / key[2]
        require(Path(row["path"]) == path and all(row[k] == spec[k] for k in ("bytes", "sha256")), "auxiliary binding differs")
        if verify_assets:
            require(path.is_file() and path.stat().st_size == spec["bytes"] and sha(path) == spec["sha256"], "auxiliary bytes changed")
        ref = Path(auxiliary["cache"]) / ("models--" + key[0].replace("/", "--")) / "refs/main"
        require(ref.read_text() == key[1], "auxiliary revision alias changed")
    return value, prepared


def installed_sources(framework):
    facts = json.loads((DATA / "sources.json").read_text())
    names = [framework] + (["cosmos-framework"] if framework == "vllm-omni" else [])
    verified = {}
    for name in names:
        spec = facts[name]
        found = importlib.util.find_spec(spec["package"])
        require(found is not None and found.submodule_search_locations, f"install pinned {name} package first")
        root = Path(next(iter(found.submodule_search_locations))).resolve(strict=True)
        require(root.is_relative_to(Path(sys.prefix).resolve()), f"{name} is outside the selected environment")
        for rel, expected in spec["installed_inventory"].items():
            path = root / rel
            # Deploy files are supplied from the installed source; missing ones fail before loading.
            require(path.is_file() and sha(path) == expected, f"pinned {name} source mismatch: {rel}")
        verified[name] = {"root": str(root), "revision": spec["revision"],
                          "files": spec["installed_inventory"], "runtime_patch": spec["runtime_patch"]}
    return verified


def installed_packages(expected):
    actual = {name: importlib.metadata.version(name) for name in expected}
    require(actual == expected, f"framework package versions differ: {actual}")
    return actual


def child_environment(inherited=None):
    env = {k: v for k, v in (os.environ if inherited is None else inherited).items()
           if not k.startswith(("IFL_", "BENCH_", "VLLM_")) and k not in {
               "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "DYNAMIC_CACHE_SCHEDULE", "NUM_DIT_STEPS",
               "LOAD_TRT_ENGINE", "ENABLE_TENSORRT", "TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH"}}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
    return env


def selected_report(cell, calls):
    import numpy as np
    require(len(calls) == cell["total_calls"], "incomplete capture cannot enter a latency row")
    for i, row in enumerate(calls):
        require(row["i"] == i and row["phase"] == ("warmup" if i < cell["warmup_calls"] else "measured"), "sample order/phase changed")
        require(row["shape"] in cell["expected_shapes"] and row["finite"] is True, "invalid action output")
        if cell["family"] == "va":
            require(row["shape"] == [1, 16 if i % 3 == 0 else 32, 16], "VA first/continuation return layout changed")
        require(isinstance(row["ms"], (int, float)) and math.isfinite(row["ms"]) and row["ms"] > 0, "invalid latency")
    samples = [calls[i]["ms"] for i in cell["selected_indices"]]
    return {"p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)), "selected_count": len(samples),
            "selected_indices": cell["selected_indices"], "historical_p50_ms": cell["historical"]["selected_p50_ms"],
            "all_measured_p50_ms": float(np.median([r["ms"] for r in calls if r["phase"] == "measured"])),
            "classification": "SCREEN", "task_quality_certified": False}


def build_policy(cell, assets, sources):
    import torch
    framework, family = cell["framework"], cell["family"]
    checkpoint = assets["checkpoint"]["path"]
    if framework == "vllm-omni":
        from vllm_omni import Omni
        deploy = Path(sources[framework]["root"]) / "deploy" / ("dreamzero.yaml" if family == "dreamzero" else "cosmos3_policy_droid.yaml")
        extras = {"model_paths": {"tokenizer": assets["tokenizer"]["path"]}} if family == "dreamzero" else {}
        return Omni(model=checkpoint, deploy_config=str(deploy), enforce_eager=False, dtype="bfloat16", **extras), None, None
    from lerobot.policies.factory import make_pre_post_processors
    if family == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        cfg = PI05Config.from_pretrained(checkpoint)
        cfg.compile_model, cfg.num_inference_steps = True, 1
        policy = PI05Policy.from_pretrained(checkpoint, config=cfg).eval().to("cuda:0")
        require(policy.config.num_inference_steps == policy.model.config.num_inference_steps == 1, "loaded pi05 is not NFE1")
        pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint,
                                             preprocessor_overrides={"device_processor": {"device": "cuda:0"}},
                                             postprocessor_overrides={"device_processor": {"device": "cuda:0"}})
    elif family == "groot":
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained
        features = {"observation.images." + key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
                    for key in ("exterior_image_1_left", "wrist_image_left")}
        features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(17,))
        cfg = GrootConfig(base_model_path=checkpoint, embodiment_tag="oxe_droid_relative_eef_relative_joint",
                          device="cuda", chunk_size=40, n_action_steps=40, num_inference_timesteps=4,
                          input_features=features, output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(17,))})
        policy = GrootPolicy.from_pretrained(checkpoint, config=cfg).eval().to("cuda")
        pre, post = make_groot_pre_post_processors_from_pretrained(cfg, checkpoint)
    else:
        from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
        from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
        from safetensors.torch import load_file
        converted = assets["converted"]["path"]
        cfg = LingBotVAConfig.from_pretrained(converted)
        cfg.device = cfg.text_encoder_device = "cuda"
        cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4
        cfg.wan_pretrained_path = checkpoint
        policy = LingBotVAPolicy.from_pretrained(converted, config=cfg).eval().to("cuda")
        pre, post = make_pre_post_processors(cfg, converted, preprocessor_overrides={"device_processor": {"device": "cuda"}})
        stats = load_file(str(Path(converted) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        policy._benchmark_quantiles = [torch.zeros(30, 1, 1), torch.zeros(30, 1, 1)]
        for name, target in zip(("q01", "q99"), policy._benchmark_quantiles):
            target[cfg.used_action_channel_ids] = stats["action." + name].reshape(16, 1, 1)
        policy._ensure_frozen_modules()
    return policy, pre, post


def infer(cell, policy, pre, post, req, inputs):
    """Return CPU actions and elapsed wall time with the historical timed boundary."""
    import numpy as np
    import torch
    family = cell["family"]
    obs = req["observation"]
    if cell["framework"] == "vllm-omni":
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        if family == "dreamzero":
            extra = {"robot_obs": obs, "reset": req["reset"], "session_id": obs["session_id"]}
        else:
            extra = {"robot_obs": obs, "num_steps": 4, "guidance": 3., "shift": 5., "history_length": 1,
                     "action_chunk_size": 32, "raw_action_dim": 8, "image_height": 540, "image_width": 640,
                     "conditioning_fps": 15, "domain_name": "droid_lerobot", "format_prompt_as_json": family == "edge", "seed": req["seed"]}
        sp = OmniDiffusionSamplingParams(extra_args=extra)
        start = time.perf_counter()
        output = policy.generate(req["prompt"], sampling_params_list=[sp])
        require(output, "empty Omni output")
        payload = output[0].multimodal_output
        require(isinstance(payload, dict), "invalid Omni action payload")
        value = payload.get("actions")
        if value is None and isinstance(payload.get("payload"), dict):
            value = payload["payload"].get("actions")
        require(value is not None, "Omni returned no actions")
        action = value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        return action, 1000 * (time.perf_counter() - start)
    if req["reset"]:
        policy.reset()
        if family == "va":
            policy._maybe_init_prompt({"task": req["prompt"]})
    torch.manual_seed(req["seed"])
    np.random.seed(req["seed"])
    if family != "groot":
        random.seed(req["seed"])
    # GR00T's CPU tensor preparation precedes timing; pi05's device transfer is timed.
    if family == "groot":
        batch = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in obs.items()}
    if family == "va":
        frames = [{key: torch.from_numpy(frame[camera].copy()).permute(2, 0, 1).float() / 255
                   for camera, key in enumerate(policy.config.obs_cam_keys)} for frame in inputs.frames]
    torch.cuda.synchronize()
    start = time.perf_counter()
    if family == "pi05":
        batch = {k: torch.as_tensor(v).to("cuda:0") for k, v in obs.items()}
        batch["task"] = req["prompt"]
        with torch.no_grad():
            action = post(policy.select_action(pre(batch)))
        action = action if torch.is_tensor(action) else torch.as_tensor(action)
        action = action.squeeze(0).detach().cpu().numpy()
    else:
        with torch.inference_mode():
            if family == "va":
                batch = pre(dict(frames[0], task=req["prompt"])) if req["cycle"] == 0 else None
                if req["cycle"]:
                    policy._obs_buffer = [policy._extract_raw_obs(pre(dict(frames[j]))) for j in obs["indices"]]
                    raw = torch.from_numpy(obs["feedback"].copy())
                    padded = torch.zeros(30, *raw.shape[1:])
                    padded[policy.config.used_action_channel_ids] = raw
                    q01, q99 = policy._benchmark_quantiles
                    policy._executed_actions = ((padded - q01) / (q99 - q01 + 1e-6) * 2 - 1).unsqueeze(0).unsqueeze(-1)
            action = post(policy.predict_action_chunk(batch if family == "va" else pre(batch)))
        action = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
    torch.cuda.synchronize()
    return action, 1000 * (time.perf_counter() - start)


def capture(root, output, *, allow_startup_patch=False):
    import numpy as np
    import torch
    value, prepared = load_prepared(root)
    cell = value["cell"]
    require(os.environ.get("HF_HUB_CACHE") == prepared["auxiliary"]["cache"], "use run to bind the isolated auxiliary cache")
    require(cell["framework"] != "vllm-omni" or allow_startup_patch, "Omni requires explicit --allow-startup-patch")
    cache_paths = checkpoint_cache_paths(cell, prepared)
    if cache_paths:
        require(os.environ.get("VLLM_OMNI_CHECKPOINT_PATHS") == json.dumps(cache_paths), "startup checkpoint paths differ from preparation")
    require(torch.cuda.get_device_capability() == (11, 0), "this historical comparison targets Thor SM110")
    sources = installed_sources(cell["framework"])
    packages = installed_packages(value["packages"])
    inputs = Inputs(Path(root) / "inputs.npz")
    result = {**value, "status": "failed", "calls": [], "sources": sources, "packages": packages,
              "device": torch.cuda.get_device_name(), "torch_runtime": torch.__version__,
              "runner_sha256": sha(__file__), "preparation_sha256": sha(Path(root) / "preparation.json"),
              "startup_checkpoint_paths": cache_paths}
    outputs, policy = {}, None
    try:
        require(not gpu_competitors(torch), "GPU has competing applications before model loading")
        if cell["family"] in {"pi05", "groot"}:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if cell["family"] == "pi05":
                torch.backends.cudnn.benchmark = False
            torch.manual_seed(9173)
            if cell["family"] == "pi05":
                np.random.seed(9173)
                random.seed(9173)
        start = time.perf_counter()
        policy, pre, post = build_policy(cell, prepared["assets"], sources)
        result["setup_seconds"] = time.perf_counter() - start
        for i in range(cell["total_calls"]):
            require(not gpu_competitors(torch), "GPU acquired a competing application during capture")
            req = request(cell, inputs, i)
            request_sha = request_hash(req)
            action, milliseconds = infer(cell, policy, pre, post, req, inputs)
            require(list(action.shape) in cell["expected_shapes"] and np.isfinite(action).all(), "invalid action output")
            outputs[f"action_{i}"] = action.copy()
            if cell["family"] == "pi05":
                queue = list(policy._action_queue)
                require(len(queue) == 49, "pi05 did not compute its complete 50-action chunk")
                outputs[f"queued_actions_{i}"] = torch.stack(queue).float().cpu().numpy()
                require(np.isfinite(outputs[f"queued_actions_{i}"]).all(), "pi05 queued non-finite actions")
            row = {"i": i, "phase": "warmup" if i < cell["warmup_calls"] else "measured",
                   "cycle": req["cycle"], "episode": req["episode"], "reset": req["reset"], "seed": req["seed"],
                   "request_sha256": request_sha, "ms": milliseconds, "shape": list(action.shape), "finite": True,
                   "action_sha256": hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest()}
            result["calls"].append(row)
            print(json.dumps(row), flush=True)
        result["latency"] = selected_report(cell, result["calls"])
        result["numeric_environment"] = {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                         "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                                         "cudnn_benchmark": torch.backends.cudnn.benchmark}
        require(installed_sources(cell["framework"]) == sources, "framework source drift during inference")
        result["status"] = "captured"
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        # A passed capture alone cannot establish external process completion.
        # Preserve it even if a native close exits the process or crashes.
        if outputs:
            np.savez_compressed(Path(output) / "actions.npz", **outputs)
            result["actions_sha256"] = sha(Path(output) / "actions.npz")
        write_new(Path(output) / "capture.json", result)
        if policy is not None:
            close = getattr(policy, "close", None)
            if close:
                close()
        write_new(Path(output) / "closed.json", {"status": "closed", "capture_sha256": sha(Path(output) / "capture.json")})


def report(root, output):
    import numpy as np
    value, prepared = load_prepared(root, verify_assets=False)
    output = Path(output)
    completion = json.loads((output / "completion.json").read_text())
    require(completion["exit_code"] == 0 and not completion["forced_kill"] and not completion["timed_out"], "renderer/framework process did not exit cleanly")
    captured = json.loads((output / "capture.json").read_text())
    closed = json.loads((output / "closed.json").read_text())
    require(captured["status"] == "captured" and captured["cell"] == value["cell"], "capture is incomplete or wrong cell")
    require(all(captured[k] == value[k] for k in ("schema", "catalog_sha256", "source_inventory_sha256", "fixture_sha256", "packages")),
            "capture declarations differ from prepared plan")
    require(closed["status"] == "closed" and closed["capture_sha256"] == sha(output / "capture.json"), "missing clean policy close")
    require(captured["runner_sha256"] == sha(__file__) == completion["runner_sha256"], "runner source differs")
    require(captured["preparation_sha256"] == sha(Path(root) / "preparation.json"), "prepared asset binding differs")
    if value["cell"]["framework"] == "vllm-omni":
        expected_paths = checkpoint_cache_paths(value["cell"], prepared)
        require(captured["startup_checkpoint_paths"] == completion["startup_checkpoint_paths"] == expected_paths,
                "startup checkpoint path evidence differs")
    require(sha(output / "actions.npz") == captured["actions_sha256"], "saved actions changed")
    inputs = Inputs(Path(root) / "inputs.npz")
    with np.load(output / "actions.npz", allow_pickle=False) as actions:
        expected_keys = {f"action_{i}" for i in range(value["cell"]["total_calls"])}
        if value["cell"]["family"] == "pi05":
            expected_keys |= {f"queued_actions_{i}" for i in range(value["cell"]["total_calls"])}
        require(set(actions.files) == expected_keys, "saved action archive coverage differs")
        for i, row in enumerate(captured["calls"]):
            req = request(value["cell"], inputs, i)
            action = actions[f"action_{i}"]
            require(row["request_sha256"] == request_hash(req), "request stream changed")
            require(all(row[k] == req[k] for k in ("i", "episode", "cycle", "reset", "seed")), "request counters differ")
            require(list(action.shape) == row["shape"] and np.isfinite(action).all(), "saved action geometry/finiteness differs")
            require(hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest() == row["action_sha256"], "saved action bytes differ")
            if value["cell"]["family"] == "pi05":
                queued = actions[f"queued_actions_{i}"]
                require(queued.shape == (49, 1, 7) and np.isfinite(queued).all(), "pi05 queued action proof differs")
    latency = selected_report(value["cell"], captured["calls"])
    require(latency == captured["latency"], "reported latency differs from selected samples")
    return {"schema": SCHEMA, "status": "passed", "cell": value["cell"], "latency": latency,
            "completion_sha256": sha(output / "completion.json"), "capture_sha256": sha(output / "capture.json"),
            "scope": value["scope"], "task_quality_certified": False}


def gpu_competitors(torch):
    """Respect nonparticipating services as well as the advisory benchmark lock."""
    smi = shutil.which("nvidia-smi") or "/usr/sbin/nvidia-smi"
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    output = subprocess.check_output([smi, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    competitors = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or fields[0] != uuid:
            continue
        pid = int(fields[1])
        try:
            # Omni's owned model processes inherit this fresh process group.
            if os.getpgid(pid) != os.getpgrp():
                competitors.append(pid)
        except ProcessLookupError:
            pass
    return competitors


def checkpoint_cache_paths(cell, prepared):
    """Bind physical model directories independently of the small processor cache."""
    if cell["framework"] != "vllm-omni":
        return []
    return [str(Path(prepared["assets"]["checkpoint"]["path"]).resolve(strict=True))]


def run(root, output, *, python, timeout_seconds=3600, lock_path="/tmp/thor_gpu.lock", allow_startup_patch=False, ptxas=None):
    import fcntl
    value, prepared = load_prepared(root)
    require(value["cell"]["framework"] != "vllm-omni" or allow_startup_patch, "Omni startup patch acknowledgement required")
    require(60 <= timeout_seconds <= 14400, "timeout must be between 60 seconds and four hours")
    output = new_directory(output)
    environment = child_environment()
    environment.update(HF_HUB_CACHE=prepared["auxiliary"]["cache"], HUGGINGFACE_HUB_CACHE=prepared["auxiliary"]["cache"])
    cache_paths = checkpoint_cache_paths(value["cell"], prepared)
    if cache_paths:
        environment["VLLM_OMNI_CHECKPOINT_PATHS"] = json.dumps(cache_paths)
    if ptxas:
        path = Path(ptxas).resolve(strict=True)
        require(path.is_file() and os.access(path, os.X_OK), "ptxas must be executable")
        environment.update(TRITON_PTXAS_PATH=str(path), TRITON_PTXAS_BLACKWELL_PATH=str(path))
    command = [str(Path(python).absolute()), "-I", str(Path(__file__).resolve()), "_capture",
               "--prepared", str(Path(root).resolve()), "--output", str(output)]
    if allow_startup_patch:
        command.append("--allow-startup-patch")
    completion = {"schema": SCHEMA, "runner_sha256": sha(__file__), "command": command,
                  "exit_code": None, "forced_kill": False, "timed_out": False, "started": time.time(),
                  "startup_checkpoint_paths": cache_paths,
                  "compiler": {"ptxas": str(path), "sha256": sha(path)} if ptxas else None}
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "worker.log").open("xb") as log:
            child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            completion["child_pid"] = child.pid
            completion["process_group"] = child.pid
            previous = signal.getsignal(signal.SIGTERM)

            def stop(_signum, _frame):
                raise InterruptedError("comparison supervisor received SIGTERM")

            signal.signal(signal.SIGTERM, stop)
            try:
                child.wait(timeout=timeout_seconds)
            except BaseException as error:
                completion.update(timed_out=isinstance(error, subprocess.TimeoutExpired), error=repr(error))
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        completion["forced_kill"] = True
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                if not isinstance(error, subprocess.TimeoutExpired):
                    raise
            finally:
                signal.signal(signal.SIGTERM, previous)
                completion.update(exit_code=child.returncode, ended=time.time())
                write_new(output / "completion.json", completion)
    result = report(root, output)
    write_new(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("list")
    p = sub.add_parser("plan")
    p.add_argument("cell")
    p = sub.add_parser("prepare")
    p.add_argument("cell")
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir")
    p.add_argument("--local-files-only", action="store_true")
    for op in ("run", "report", "_capture"):
        p = sub.add_parser(op)
        p.add_argument("--prepared", required=True)
        p.add_argument("--output", required=True)
        if op != "report":
            p.add_argument("--allow-startup-patch", action="store_true")
        if op == "run":
            p.add_argument("--python", default=sys.executable)
            p.add_argument("--timeout-seconds", type=int, default=3600)
            p.add_argument("--lock-path", default="/tmp/thor_gpu.lock")
            p.add_argument("--ptxas", help="Thor-compatible CUDA ptxas; bound in the execution receipt")
    args = parser.parse_args(argv)
    if args.operation == "list":
        result = catalog()
    elif args.operation == "plan":
        result = plan(args.cell)
    elif args.operation == "prepare":
        result = prepare(args.cell, args.output, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    elif args.operation == "run":
        result = run(args.prepared, args.output, python=args.python, timeout_seconds=args.timeout_seconds,
                     lock_path=args.lock_path, allow_startup_patch=args.allow_startup_patch, ptxas=args.ptxas)
    elif args.operation == "report":
        result = report(args.prepared, args.output)
    else:
        return capture(args.prepared, args.output, allow_startup_patch=args.allow_startup_patch)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
