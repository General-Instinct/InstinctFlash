"""Independent CPU audit of the bounded four-job DreamZero cache screen.

No model, Torch, CUDA, SSH or benchmark code is imported or executed. By default
the result covers receipts, complete NPZ actions and manifest hash bindings.
--strict-content additionally requires all inventoried source/driver/external
files to be available locally. Missing results never count as passing results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np


JOBS = ("native-dynamic", "native-checkpoint", "fp8-dynamic", "fp8-checkpoint")
ROLES = ("shared", "native_oracle", "shared_restored")
KV_KEYS = ("kv_cache1", "kv_cache_neg", "crossattn_cache", "crossattn_cache_neg")
MASK = [i in (0, 1, 2, 6, 10, 13, 14, 15) for i in range(16)]
REVISION = "96ad344138c66e82536422432ad742f015784942"
NATIVE_SHA = "7193cd73423472aa252bee73bd80e0d673c89d773ec852e90f50154729b50845"
NATIVE_METHODS = {
    "_run_diffusion_steps": "739a3017e0c07371223ce0c831dfc8a5a00ede1aaffca5de6abe102ab28bbf1c",
    "should_run_model": "7a7f6026dba7a5f1320586f00d4220e65e972cace7d5d270bcb322a99a223b60",
    "lazy_joint_video_action": "e89f8491903e0214eb5606453d68c555cbb947a2dfc1700781b7d740044aae62",
}
SUFFIXES = {".py", ".pyi", ".cu", ".cpp", ".cc", ".c", ".h", ".hpp",
            ".yaml", ".yml", ".toml", ".so", ".pyd", ".dll"}


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def relative(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe relative manifest path: {name}")
    return path


def tensor_record(value):
    require(isinstance(value, dict), "Invalid tensor receipt")
    require(isinstance(value.get("shape"), list) and all(type(v) is int and v >= 0 for v in value["shape"]), "Invalid tensor shape")
    require(isinstance(value.get("dtype"), str) and value["dtype"].startswith("torch."), "Invalid tensor dtype")
    require(is_sha(value.get("sha256")), "Invalid tensor SHA256")


def mask_record(record, *, trace=False):
    mask = record.get("compute_mask")
    require(isinstance(mask, list) and len(mask) == 16 and all(type(v) is bool for v in mask), "Invalid 16-slot compute mask")
    require(mask[:2] == [True, True], "Dynamic history must start with two computations")
    require(record.get("computed_steps") == sum(mask), "Computed count disagrees with mask")
    require(record.get("denoiser_branch_forwards") == 2 * sum(mask), "CFG branch count disagrees with mask")
    require(record.get("kv_update_branch_forwards") == 2, "Expected two observation KV branches")
    if not trace:
        require(record.get("solver_steps") == [list(range(16)), list(range(16))], "Both native solvers must consume all 16 slots")
        return
    require(record.get("status") == "completed" and record.get("total_steps") == 16, "Incomplete dynamic generation")
    require(record.get("reused_steps") == 16 - sum(mask), "Reuse count disagrees with mask")
    require(record.get("denoiser_calls") == sum(mask) and record.get("kv_update_calls") == 1, "Native call counts disagree")
    require(record.get("pending_decision") is None, "Unconsumed final decision")
    rows = record.get("trace")
    require(isinstance(rows, list) and len(rows) == 16, "Missing complete controller trace")
    countdown, computed = 0, 0
    for index, row in enumerate(rows):
        require(row.get("index") == index and row.get("consumed") is True, "Misordered/unconsumed controller trace")
        require(row.get("countdown_before") == countdown, "Countdown before differs")
        similarity = row.get("similarity")
        if computed < 2:
            answer, reason = True, "insufficient_history"
        elif countdown > 1:
            answer, reason, countdown = False, "countdown_reuse", countdown - 1
        elif countdown == 1:
            answer, reason, countdown = True, "countdown_refresh", 0
        else:
            require(finite(similarity), "Nonfinite/missing cosine telemetry")
            answer, reason = True, "similarity_refresh"
            for threshold, skip in ((0.95, 4), (0.93, 2)):
                # Upstream compares an FP32 scalar tensor, including threshold rounding.
                if np.float32(similarity) > np.float32(threshold):
                    answer, reason, countdown = False, "similarity_reuse", skip
                    break
        if reason in ("insufficient_history", "countdown_reuse", "countdown_refresh"):
            require(similarity is None, "Unexpected cosine evaluation during countdown/history setup")
        require(row.get("compute") is answer and mask[index] is answer, "Trace decision disagrees with upstream countdown rule")
        require(row.get("reason") == reason and row.get("countdown_after") == countdown, "Trace reason/countdown differs")
        computed += answer


class Audit:
    def __init__(self, root, protocol, *, source_root=None, driver_root=None, external_map=None, strict_content=False):
        self.root = Path(root).resolve()
        self.protocol_path = Path(protocol).resolve()
        self.source_root = Path(source_root).resolve() if source_root else self.root / "source"
        self.driver_root = Path(driver_root).resolve() if driver_root else self.root
        self.external_map = external_map or {}
        self.strict_content = strict_content
        self.inputs, self.errors, self.missing = {}, [], []
        self.receipts, self.timing_samples = {}, {}
        self.content = {}

    def bind(self, path):
        path = Path(path).resolve()
        value = sha(path)
        previous = self.inputs.get(str(path))
        require(previous in (None, value), f"Input changed during audit: {path}")
        self.inputs[str(path)] = value
        return value

    def read(self, path, *, needed=True):
        path = Path(path)
        if not path.is_file():
            if needed:
                self.missing.append(str(path))
            return None
        before = self.bind(path)
        def bad_constant(value):
            raise Invalid(f"Nonfinite JSON constant: {value}")
        result = json.loads(path.read_text(), object_pairs_hook=unique_object, parse_constant=bad_constant)
        require(self.bind(path) == before, f"JSON changed while reading: {path}")
        return result

    def mapped_external(self, name):
        for remote, local in sorted(self.external_map.items(), key=lambda item: -len(item[0])):
            if name == remote or name.startswith(remote.rstrip("/") + "/"):
                return Path(local) / name[len(remote):].lstrip("/")
        return Path(name)

    def manifest_contents(self, label, entries, resolver):
        require(isinstance(entries, dict) and entries, f"Empty/invalid {label} manifest")
        verified, unavailable = [], []
        for name, expected in entries.items():
            require(is_sha(expected), f"Invalid {label} digest: {name}")
            path = resolver(name)
            if not path.is_file():
                unavailable.append(name)
                continue
            require(self.bind(path) == expected, f"{label} content mismatch: {name}")
            verified.append(name)
        self.content[label] = dict(entries=len(entries), verified_files=len(verified),
                                   unavailable_files=unavailable, complete=not unavailable)
        if unavailable and self.strict_content:
            self.missing.append(f"{label}: {len(unavailable)} inventoried files unavailable locally")

    def manifests(self, live, completion):
        manifests = {}
        for filename in ("source_manifest.json", "driver_manifest.json", "external_manifest.json"):
            doc = self.read(self.root / filename)
            if doc is None:
                continue
            manifests[filename] = doc
            actual = self.inputs[str((self.root / filename).resolve())]
            if live is not None and live.get("status") != "queued":
                require(live.get("manifests", {}).get(filename) == actual, f"Supervisor manifest binding differs: {filename}")
            if completion is not None:
                require(completion.get("artifacts", {}).get(filename) == actual, f"Completion manifest binding differs: {filename}")
        source = manifests.get("source_manifest.json")
        if source is not None:
            require(source.get("schema") == 1 and isinstance(source.get("required_binaries"), dict), "Invalid source manifest schema")
            self.manifest_contents("source", source["files"], lambda name: self.source_root / relative(name))
            for pattern, names in source["required_binaries"].items():
                require(isinstance(names, list) and names and all(name in source["files"] for name in names), f"Invalid required binary inventory: {pattern}")
            if self.source_root.is_dir():
                names = {str(path.relative_to(self.source_root)) for path in self.source_root.rglob("*")
                         if path.is_file() and "__pycache__" not in path.parts
                         and (path.suffix in SUFFIXES or re.search(r"\.so(?:\.\d+)*$", path.name))}
                require(not names - set(source["files"]), "Local source contains unmanifested code/native binaries")
                if names == set(source["files"]):
                    for pattern, expected in source["required_binaries"].items():
                        actual = sorted(str(path.relative_to(self.source_root)) for path in self.source_root.glob(pattern)
                                        if path.is_file() and (path.suffix in {".pyd", ".dll"} or re.search(r"\.so(?:\.\d+)*$", path.name)))
                        require(actual == expected, f"Required binary inventory differs: {pattern}")
        driver = manifests.get("driver_manifest.json")
        if driver is not None:
            require(set(driver) == {"benchmark.py", "run.py", "load_memory.py", "freeze.py", "protocol.json"}, "Unexpected driver inventory")
            require(driver["protocol.json"] == self.inputs[str(self.protocol_path)], "Protocol differs from prelaunch driver manifest")
            self.manifest_contents("drivers", driver, lambda name: self.driver_root / relative(name))
        external = manifests.get("external_manifest.json")
        fixture = None
        if external is not None:
            require(all(Path(name).is_absolute() for name in external), "External inventory must use absolute paths")
            heads = [value for name, value in external.items() if name.endswith("/action_head/wan_flow_matching_action_tf.py")]
            require(heads == [NATIVE_SHA], "Audited native source revision differs")
            fixtures = [value for name, value in external.items() if name.endswith("/va_eval_obs.npz")]
            require(len(fixtures) == 1, "Expected one frozen input fixture")
            fixture = fixtures[0]
            self.manifest_contents("external", external, self.mapped_external)
        return fixture

    def protocol(self):
        value = self.read(self.protocol_path)
        if value is None:
            return None
        require(value.get("schema") == 1 and value.get("quality_certified") is False, "Invalid protocol/quality claim")
        require(value.get("frozen_before_gpu_outputs") is True, "Protocol does not declare prelaunch freeze")
        require(value.get("model") == "GEAR-Dreams/DreamZero-DROID" and value.get("revision") == REVISION, "Protocol checkpoint differs")
        require(value.get("job_order") == list(JOBS) and value.get("expected_requests") == 72, "Protocol job/request count differs")
        require(value.get("grid_steps") == 16 and value.get("cfg") == 5.0, "Protocol scheduler/CFG differs")
        require(value.get("checkpoint_fixed_computed_steps") == 8 and value.get("dynamic_profile") == "dreamzero_velocity_v1", "Protocol cache profile differs")
        require(value.get("requests_per_timing_job") == 9, "Protocol timing count differs")
        timing = value.get("timing", {})
        require(timing.get("episodes") == 3 and timing.get("cycles_per_episode") == 3
                and timing.get("warmup_episodes") == [0] and timing.get("measured_episodes") == [1, 2], "Protocol timing selection differs")
        parity = value.get("same_precision_dynamic_integration_parity", {})
        require(parity.get("roles") == list(ROLES) and parity.get("requests_per_role") == 6, "Protocol parity design differs")
        return value

    def policy(self, receipt, precision, dynamic):
        policy = receipt["execution_policy"]
        require(policy == receipt["preload_execution_policy"], "Execution policy changed after lazy loading")
        require(policy.get("precision") == precision and policy.get("tier_ceiling") == "behavioral", "Precision/permission metadata differs")
        selected = policy.get("step_cache", {})
        require(selected.get("dynamic") is dynamic and selected.get("fixed_steps") == 8, "Selected schedule differs")
        require(selected.get("profile") == ("dreamzero_velocity_v1" if dynamic else None), "Selected profile differs")
        if dynamic:
            require(policy.get("category") == "OPERATING-POINT" and policy.get("transform_tier") == "BEHAVIORAL", "Dynamic reuse mislabeled as arithmetic/exact transform")
        schedule = policy.get("schedule_options", {}).get("dreamzero_schedule", {})
        require(schedule.get("grid_steps") == 16 and schedule.get("computed_steps") == (None if dynamic else 8), "Preflight invented/incorrect compute count")
        backend = receipt["backend_stats"]
        require(backend.get("precision") == precision and backend.get("dynamic_cache_schedule") is dynamic, "Backend policy differs")
        build = backend["build"]
        require(build.get("steps") == {"video_action": 16, "kv_commit": 1}, "Loaded scheduler grid differs")
        require(build.get("dit_step_mask") == MASK and build.get("guidance", {}).get("video_action") == ["cfg", 5.0], "Loaded fixed mask/CFG differs")
        if precision == "fp8":
            recipe = receipt["fp8_recipe"]
            require(recipe == backend.get("fp8_recipe"), "FP8 recipe receipts disagree")
            projections = recipe.get("projections", [])
            require(len(projections) == 200 and len({v["path"] for v in projections}) == 200, "Expected 200 distinct FP8 projections")
            require(recipe.get("scheduler_steps") == 16 and recipe.get("dit_step_mask") == MASK
                    and recipe.get("dynamic_cache_schedule") is dynamic, "FP8 recipe schedule differs")
        else:
            require(backend.get("fp8_recipe") is None and "fp8_recipe" not in receipt, "Native job unexpectedly has FP8 recipe")
        if dynamic:
            hook = backend["step_cache"]
            require(hook.get("installed") is True and hook.get("quality_certificate") is None, "Missing controller or unsupported quality claim")
            require(hook.get("profile") == "dreamzero_velocity_v1" and hook.get("decision_signal") == "cfg_video", "Wrong decision signal/profile")
            require(hook.get("source_methods") == NATIVE_METHODS, "Loaded hook source gate differs from audited native methods")
            controller = hook["controller"]
            require(controller.get("quality_certified") is False and controller.get("transform_tier") == "BEHAVIORAL", "Controller evidence mislabeled")
            require(controller.get("history_entries") == controller.get("cache_bytes") == 0 and controller.get("active") is False, "Generation retained live cache")
            mask_record(hook["last_generation"], trace=True)
        else:
            require(backend.get("step_cache") is None, "Checkpoint schedule unexpectedly installed dynamic controller")

    def job(self, name, fixture):
        precision, schedule = name.split("-")
        path = self.root / f"{name}.json"
        receipt = self.read(path)
        if receipt is None:
            progress = self.read(path.with_suffix(".progress.json"), needed=False)
            return dict(status="pending" if progress is None else "incomplete",
                        reported_stage=progress.get("stage") if progress else None,
                        available_timing_rows=len(progress.get("timings", [])) if progress else 0,
                        finite_actions_checked=0)
        self.receipts[name] = receipt
        result = dict(status="incomplete", reported_status=receipt.get("status"), finite_actions_checked=0)
        require(receipt.get("status") == "passed", f"Job did not pass: {receipt.get('status')}: {receipt.get('error') or receipt.get('cleanup_errors')}")
        require(receipt.get("quality_certified") is False and not receipt.get("cleanup_errors"), "Unsupported quality claim/cleanup failure")
        require(receipt.get("precision") == precision and receipt.get("schedule") == schedule, "Job name and receipt disagree")
        require(receipt.get("revision") == REVISION and receipt.get("model") == "GEAR-Dreams/DreamZero-DROID", "Job checkpoint differs")
        require("thor" in receipt.get("device", "").lower(), "Receipt does not identify Thor")
        require(receipt.get("native_head_seed") == 1140, "Native per-head noise seed differs")
        require(not receipt.get("competing_processes"), "Competing GPU process observed")
        timing_scope = receipt.get("timing_scope", {})
        require(timing_scope.get("warmup_episodes") == 1 and timing_scope.get("measured_episodes") == 2
                and timing_scope.get("continuation_cycles_per_episode") == 2, "Timing scope count differs")
        require(set(timing_scope.get("excluded", [])) == {"model loading", "episode reset", "observation assembly", "seeding",
                "diagnostic observers", "tensor hashing", "receipt writes"}, "Timing exclusions differ")
        require(is_sha(receipt.get("fixture_sha256")), "Invalid fixture digest")
        if fixture is not None:
            require(receipt["fixture_sha256"] == fixture, "Job input differs from external manifest")
        dynamic = schedule == "dynamic"
        self.policy(receipt, precision, dynamic)
        rows = receipt.get("timings", [])
        require(len(rows) == 9, "Expected nine public timing requests")
        samples = {}
        for index, row in enumerate(rows):
            episode, cycle = divmod(index, 3)
            require((row.get("episode"), row.get("cycle")) == (episode, cycle), "Timing order differs")
            require(row.get("seed") == 17000 + episode * 17 + cycle, "Timing seed differs")
            require(row.get("phase") == ("warmup" if episode == 0 else "measured"), "Warmup selection differs")
            require(finite(row.get("ms")) and row["ms"] > 0, "Nonfinite/nonpositive raw latency")
            if dynamic:
                mask_record(row["step_cache"], trace=True)
            else:
                require(row.get("step_cache") is None, "Fixed-mask timing has dynamic cache stats")
            if episode in (1, 2) and cycle in (1, 2):
                samples[f"{episode}:{cycle}"] = row["ms"]
        p50 = statistics.median(samples.values())
        require(receipt.get("continuation_sample_count") == 4, "Expected four continuation samples")
        require(finite(receipt.get("continuation_p50_ms")) and math.isclose(receipt["continuation_p50_ms"], p50, rel_tol=1e-12, abs_tol=1e-9), "Stored p50 differs from raw timings")
        result.update(continuation_p50_ms=p50, continuation_sample_count=4,
                      continuation_raw_ms=samples, timing_requests=9,
                      actual_timing_computed_steps=[r["step_cache"]["computed_steps"] if dynamic else 8 for r in rows])
        archive_path = path.with_suffix(".npz")
        if not archive_path.is_file():
            self.missing.append(str(archive_path))
            return result
        require(self.bind(archive_path) == receipt.get("actions_sha256"), "NPZ file digest differs")
        expected = {f"timing_{e}_{c}" for e in range(3) for c in range(3)}
        if dynamic:
            expected |= {f"{role}_{e}_{c}" for role in ROLES for e in range(2) for c in range(3)}
        with np.load(archive_path, allow_pickle=False) as archive:
            require(len(archive.files) == len(expected) and set(archive.files) == expected, "NPZ action inventory differs")
            arrays = {}
            for key in sorted(expected):
                array = archive[key]
                require(array.shape == (24, 8) and array.dtype == np.dtype("float32"), f"Wrong full action shape/dtype: {key}")
                require(bool(np.isfinite(array).all()), f"Nonfinite action: {key}")
                arrays[key] = np.ascontiguousarray(array).tobytes()
            result["finite_actions_checked"] = len(arrays)
            if dynamic:
                captures = receipt["captures"]
                require(set(captures) == set(ROLES), "Missing/extra parity roles")
                for role in ROLES:
                    bank = captures[role]
                    require(len(bank) == 6, "Expected six requests per parity role")
                    for index, item in enumerate(bank):
                        episode, cycle = divmod(index, 3)
                        require((item.get("episode"), item.get("cycle"), item.get("seed")) == (episode, cycle, 17000 + episode * 17 + cycle), "Diagnostic input/seed coordinates differ")
                        key = f"{role}_{episode}_{cycle}"
                        require(sha_bytes(arrays[key]) == item.get("action_sha256"), f"Action digest differs: {key}")
                        mask_record(item["execution"])
                        values = item["values"]
                        require(set(values) == {"current_start_frame", "video_pred", "action_pred", *KV_KEYS}, "Endpoint/KV inventory differs")
                        require(type(values["current_start_frame"]) is int and values["current_start_frame"] > 0, "Invalid native frame position")
                        for field in ("video_pred", "action_pred"):
                            tensor_record(values[field])
                        for field in KV_KEYS:
                            require(isinstance(values[field], list) and len(values[field]) == 40, "Expected forty arrays per native KV bank")
                            for value in values[field]:
                                tensor_record(value)
                        rng = item["rng_after"]
                        require(set(rng) == {"torch_cpu", "torch_cuda", "numpy", "python"}, "Incomplete RNG receipt")
                        require(all(is_sha(rng[field]) for field in ("torch_cpu", "torch_cuda", "python")), "Invalid RNG hashes")
                        numpy_rng = rng["numpy"]
                        require(numpy_rng.get("algorithm") == "MT19937" and is_sha(numpy_rng.get("state_sha256"))
                                and type(numpy_rng.get("position")) is int and 0 <= numpy_rng["position"] <= 624
                                and type(numpy_rng.get("has_gauss")) is int and numpy_rng["has_gauss"] in (0, 1)
                                and finite(numpy_rng.get("cached_gaussian")), "Invalid NumPy RNG receipt")
                        if role != "shared":
                            require(item == captures["shared"][index], f"Exact paired-role receipt mismatch: {role}/{index}")
                            require(arrays[key] == arrays[f"shared_{episode}_{cycle}"], f"Full action bytes differ: {role}/{index}")
                parity = receipt["native_dynamic_parity"]
                require(parity.get("passed") is True and parity.get("paired_requests") == 6
                        and parity.get("comparisons") == 12 and parity.get("kv_arrays_per_request") == 160, "Parity summary count differs")
                final = receipt["backend_stats"]["step_cache"]["last_generation"]
                require(final["compute_mask"] == captures["shared_restored"][-1]["execution"]["compute_mask"], "Final backend trace differs from observed mask")
                result["exact_paired_action_comparisons"] = 12
                result["diagnostic_requests"] = 18
            else:
                require(receipt.get("captures") == {} and "native_dynamic_parity" not in receipt, "Fixed mask incorrectly claims dynamic parity")
                result["diagnostic_requests"] = 0
        self.bind(archive_path)
        result["status"] = "passed"
        self.timing_samples[name] = samples
        return result

    def supervisor(self, live, completion):
        if live is None:
            return
        require(live.get("quality_certified") is False, "Supervisor quality claim differs")
        require(live.get("status") in ("queued", "running", "passed"), f"Supervisor failed: {live.get('error')}")
        jobs = live.get("jobs", [])
        require(len(jobs) <= 4, "Unexpected additional supervisor job")
        previous_end = None
        for expected, job in zip(JOBS, jobs):
            require(f"{job.get('precision')}-{job.get('schedule')}" == expected, "Supervisor job order differs")
            require(job.get("status") in ("running", "passed"), f"Supervisor job failed: {expected}")
            require(finite(job.get("started")), "Missing job start time")
            if previous_end is not None:
                require(job["started"] >= previous_end, "Supervisor job intervals overlap")
            if job.get("status") == "passed":
                require(job.get("exit_code") == 0 and finite(job.get("ended")) and job["ended"] >= job["started"], "Passed job has invalid exit/timing")
                previous_end = job["ended"]
        if live.get("status") == "passed":
            require(len(jobs) == 4 and all(job.get("status") == "passed" for job in jobs), "Supervisor passed before all jobs")
        if completion is not None:
            require(live.get("status") == "passed", "Completion exists without passed supervisor")
            require(completion.get("status") == "passed" and completion.get("fresh_jobs") == 4
                    and completion.get("expected_requests") == 72 and completion.get("quality_certified") is False, "Invalid completion summary")
            artifacts = completion.get("artifacts", {})
            mandatory = {f"{name}{suffix}" for name in JOBS for suffix in (".json", ".npz")}
            mandatory |= {"source_manifest.json", "driver_manifest.json", "external_manifest.json", "protocol.json", "environment.json"}
            require(mandatory <= set(artifacts), "Completion omits mandatory artifacts")
            for filename, expected in artifacts.items():
                require(Path(filename).name == filename and is_sha(expected), "Invalid completion artifact")
                path = self.root / filename
                if path.is_file():
                    require(self.bind(path) == expected, f"Completion artifact mismatch: {filename}")
                elif filename in mandatory:
                    self.missing.append(str(path))

    def run(self):
        report = dict(schema=1, status="pending", quality_certified=False,
                      verification_scope="independent full-action/latency checks and manifest hash bindings",
                      audit_script_sha256=sha(Path(__file__)), jobs={}, ratios={},
                      limitations=["Raw video/action latent and KV tensors are not archived: their shape/dtype/hash receipts and paired equality are checked, but finiteness cannot be independently recomputed from hashes.",
                                   "Dynamic versus fixed masks changes the policy; FP8 changes arithmetic. Latency ratios are not pure architecture speedups or task-quality evidence.",
                                   "Four continuation samples per cell are a bounded screen, not a stable tail-latency estimate."])
        def guarded(label, function):
            try:
                return function()
            except Exception as error:
                self.errors.append(f"{label}: {type(error).__name__}: {error}")
                return None
        protocol = guarded("protocol", self.protocol)
        live = guarded("supervisor receipt", lambda: self.read(self.root / "live_status.json"))
        completion = guarded("completion receipt", lambda: self.read(self.root / "completion.json"))
        fixture = guarded("manifest bindings", lambda: self.manifests(live, completion)) if protocol else None
        guarded("supervisor/completion", lambda: self.supervisor(live, completion))
        for name in JOBS:
            result = guarded(name, lambda name=name: self.job(name, fixture))
            report["jobs"][name] = result or dict(status="failed", finite_actions_checked=0)
        completed = [name for name, result in report["jobs"].items() if result["status"] == "passed"]
        if len(completed) == 4:
            require(sum(report["jobs"][name]["finite_actions_checked"] for name in JOBS) == 72, "Expected 72 finite action arrays")
            common = [(self.receipts[name].get("device"), self.receipts[name].get("torch"), self.receipts[name].get("fixture_sha256")) for name in JOBS]
            if any(item != common[0] for item in common[1:]):
                self.errors.append("Job device/Torch/input identities differ")
        pairs = {
            "native_checkpoint_over_native_dynamic": ("native-checkpoint", "native-dynamic"),
            "fp8_checkpoint_over_fp8_dynamic": ("fp8-checkpoint", "fp8-dynamic"),
            "native_checkpoint_over_fp8_checkpoint": ("native-checkpoint", "fp8-checkpoint"),
            "native_dynamic_over_fp8_dynamic": ("native-dynamic", "fp8-dynamic"),
            "native_checkpoint_over_fp8_dynamic": ("native-checkpoint", "fp8-dynamic"),
        }
        for label, (numerator, denominator) in pairs.items():
            if numerator in completed and denominator in completed:
                top, bottom = self.timing_samples[numerator], self.timing_samples[denominator]
                paired = {key: top[key] / bottom[key] for key in top}
                report["ratios"][label] = dict(numerator=numerator, denominator=denominator,
                    ratio_of_continuation_p50=statistics.median(top.values()) / statistics.median(bottom.values()),
                    per_request_ratios=paired, median_of_paired_ratios=statistics.median(paired.values()),
                    interpretation="Greater than one means the denominator cell was faster in these samples; no quality equivalence implied")
        report.update(completed_jobs=len(completed), finite_actions_checked=sum(r["finite_actions_checked"] for r in report["jobs"].values()),
                      manifest_content_verification=self.content, inputs_sha256=dict(sorted(self.inputs.items())),
                      audit_inputs_sha256=sha_bytes(json.dumps(self.inputs, sort_keys=True, separators=(",", ":")).encode()),
                      protocol_sha256=self.inputs.get(str(self.protocol_path)), errors=self.errors, missing_inputs=sorted(set(self.missing)))
        if any(not item["complete"] for item in self.content.values()):
            report["limitations"].append("Some inventoried files are unavailable locally; their manifests are hash-bound to supervisor/completion receipts, but unavailable file contents were not independently read.")
        report["status"] = ("failed" if self.errors else "passed" if len(completed) == 4 and completion is not None and not self.missing
                            else "incomplete" if completed or self.receipts else "pending")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Locally copied Thor run directory")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--driver-root", type=Path)
    parser.add_argument("--external-map", type=Path, help="JSON mapping remote absolute path prefixes to local copies")
    parser.add_argument("--strict-content", action="store_true")
    parser.add_argument("--output", type=Path, help="Write audit JSON; otherwise print it")
    args = parser.parse_args()
    audit = Audit(args.run_dir, args.protocol, source_root=args.source_root,
                  driver_root=args.driver_root, strict_content=args.strict_content)
    if args.external_map:
        mapping = audit.read(args.external_map)
        require(isinstance(mapping, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()), "External map must map path strings to path strings")
        audit.external_map = mapping
    report = audit.run()
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        require(args.output.resolve() not in {Path(name) for name in audit.inputs} | {Path(__file__).resolve()}, "Audit output cannot overwrite an input or the auditor")
        args.output.write_text(data)
    else:
        print(data, end="")
    return {"passed": 0, "failed": 1, "pending": 2, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
