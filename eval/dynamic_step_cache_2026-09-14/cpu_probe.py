"""CPU comparison of pinned, unmodified upstream step-cache decision methods.

Only method ASTs are compiled: neither model package is imported or constructed.
Synthetic velocities test proxy decisions, not model outputs, task quality, or
end-to-end performance. The common enabled configuration is compared separately
from the deliberately different disabled-mode semantics.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


NATIVE_DEFAULT = Path(
    "/home/ubuntu/dreamzero-repo/groot/vla/model/dreamzero/action_head/"
    "wan_flow_matching_action_tf.py"
)
OMNI_ROOT = Path("/home/ubuntu/work_clones/vllm-omni-benchmark-20260913")
OMNI_STATE_DEFAULT = OMNI_ROOT / "vllm_omni/diffusion/cache/stepcache/state.py"
OMNI_CONFIG_DEFAULT = OMNI_ROOT / "vllm_omni/diffusion/cache/stepcache/config.py"
SOURCE_PINS = {
    "native": "7193cd73423472aa252bee73bd80e0d673c89d773ec852e90f50154729b50845",
    "omni_state": "32b00daaf6cd6e4ac73cf9c6f40eb3c69c46386f5fc0759d6cd6f23569b75a87",
    "omni_config": "99f7fb873d2f44f1fd290261c72a6470cf569fbbe39c95178095f7acaca65721",
}
RANDOM_GENERATIONS = 160
RANDOM_HISTORIES = 192


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def scalar(value):
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def source(path, role):
    data = path.read_bytes()
    assert digest(data) == SOURCE_PINS[role], f"Re-audit changed source: {path}"
    text = data.decode()
    return text, ast.parse(text), {"path": str(path.resolve()), "sha256": digest(data)}


def extract_method(text, tree, name, path, torch):
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, (path, name, len(matches))
    node = matches[0]
    assert not node.decorator_list
    # Add postponed annotations outside the original function. Its body is unchanged.
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        copy.deepcopy(node),
    ], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    segment = ast.get_source_segment(text, node)
    return namespace[name], {
        "name": name, "start_line": node.lineno, "end_line": node.end_lineno,
        "source_sha256": digest(segment.encode()),
        "ast_sha256": digest(ast.dump(node, include_attributes=False).encode()),
        "body_modified": False,
    }


def native_shipped_mask(tree):
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (isinstance(test.left, ast.Name) and test.left.id == "num_dit_steps"
                and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == 8):
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    if any(isinstance(t, ast.Attribute) and t.attr == "dit_step_mask"
                           for t in statement.targets):
                        matches.append(ast.literal_eval(statement.value))
    assert len(matches) == 1 and len(matches[0]) == 16 and sum(matches[0]) == 8
    return matches[0]


def config_defaults(tree):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StepCacheConfig")
    wanted = {"enabled", "min_history_steps", "max_history", "sim_thresholds", "skip_countdowns"}
    result = {n.target.id: ast.literal_eval(n.value) for n in cls.body
              if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
              and n.target.id in wanted}
    assert result == dict(enabled=True, min_history_steps=2, max_history=2,
                          sim_thresholds=(0.95, 0.93), skip_countdowns=(4, 2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=Path, default=NATIVE_DEFAULT)
    parser.add_argument("--omni-state", type=Path, default=OMNI_STATE_DEFAULT)
    parser.add_argument("--omni-config", type=Path, default=OMNI_CONFIG_DEFAULT)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Launch with CUDA_VISIBLE_DEVICES=''"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ.get(name) == "1", f"Set {name}=1"

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    cpu_rng = torch.get_rng_state().clone()
    native_text, native_tree, native_meta = source(args.native_source, "native")
    omni_text, omni_tree, omni_meta = source(args.omni_state, "omni_state")
    _, config_tree, config_meta = source(args.omni_config, "omni_config")
    native, native_method = extract_method(native_text, native_tree, "should_run_model", args.native_source, torch)
    omni, omni_method = extract_method(omni_text, omni_tree, "should_run_step", args.omni_state, torch)
    omni_reset, reset_method = extract_method(omni_text, omni_tree, "reset", args.omni_state, torch)
    mask = native_shipped_mask(native_tree)
    defaults = config_defaults(config_tree)
    comparisons = 0

    def tensor_record(tensor):
        assert tensor.device.type == "cpu"
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": digest(raw)}

    def states():
        return (SimpleNamespace(dynamic_cache_schedule=True, dit_step_mask=list(mask), skip_countdown=0),
                SimpleNamespace(config=SimpleNamespace(**defaults), skip_countdown=0))

    def decide(pair, index, history):
        nonlocal comparisons
        left, right = pair
        native_history = [(i, v.clone(), a.clone()) for i, v, a in history]
        omni_history = [(v.clone(),) for _, v, _ in history]
        before = [left.skip_countdown, right.skip_countdown]
        assert before[0] == before[1]
        a = bool(native(left, index, 1000 - index * 62.5, native_history))
        b = bool(omni(right, omni_history))
        assert a == b and left.skip_countdown == right.skip_countdown, (index, before, a, b)
        comparisons += 1
        trace = {"slot": index, "history_entries": len(history), "compute": a,
                 "countdown_before": before[0], "countdown_after": left.skip_countdown}
        if len(history) >= 2:
            last, previous = history[-1], history[-2]
            cos = torch.nn.functional.cosine_similarity(last[1].flatten(1).float(), previous[1].flatten(1).float(), dim=1)
            action_cos = torch.nn.functional.cosine_similarity(last[2].flatten(1).float(), previous[2].flatten(1).float(), dim=1)
            trace.update(video_cosines=[scalar(x) for x in cos],
                         video_mean_cosine=scalar(cos.mean()),
                         action_cosines=[scalar(x) for x in action_cos])
        return a, trace

    def generation(name, predictions, pair=None, reset=True, full_trace=True):
        pair = states() if pair is None else pair
        if reset:
            pair[0].skip_countdown = 0  # Native generation entry, source lines 1226-1227.
            omni_reset(pair[1])
        history, trace = [], []
        max_video_relative_error = 0.0
        for index, (video, action) in enumerate(predictions):
            compute, row = decide(pair, index, history)
            if compute:
                history.append((index, video.clone(), action.clone()))
                history[:] = history[-defaults["max_history"]:]
            else:
                assert history
                stale = history[-1][1]
                if bool(torch.isfinite(video).all()) and bool(torch.isfinite(stale).all()):
                    norm = float(torch.linalg.vector_norm(video.float()))
                    if norm:
                        error = float(torch.linalg.vector_norm(video.float() - stale.float())) / norm
                        max_video_relative_error = max(max_video_relative_error, error)
            row["last_computed_slot"] = history[-1][0]
            trace.append(row)
        computed = [row["slot"] for row in trace if row["compute"]]
        inputs = [{"video": tensor_record(v), "action": tensor_record(a)} for v, a in predictions]
        result = {"name": name, "scheduler_slots": len(predictions), "computed_slots": computed,
                  "computed_count": len(computed), "reused_count": len(predictions) - len(computed),
                  "countdowns_after": [row["countdown_after"] for row in trace],
                  "input_sha256": digest(canonical(inputs)), "trace_sha256": digest(canonical(trace)),
                  "max_synthetic_video_relative_l2_error_on_reused_slots": max_video_relative_error,
                  "native_omni_decision_countdown_parity": True}
        if full_trace:
            result["trace"] = trace
            result["input_tensors"] = inputs
        return result, pair

    one = torch.ones((1, 2, 4), dtype=torch.float32)
    action = torch.ones((1, 3, 2), dtype=torch.float32)
    constant = [(one.clone(), action.clone()) for _ in range(16)]
    named = []
    baseline, _ = generation("constant_video", constant)
    assert baseline["computed_slots"] == [0, 1, 6, 11]
    named.append(baseline)
    unstable = [(one * (-1 if i % 2 else 1), action.clone()) for i in range(16)]
    result, _ = generation("opposing_video_all_slots_computed", unstable)
    assert result["computed_count"] == 16 > sum(mask)
    named.append(result)
    result, _ = generation("magnitude_scaling_same_direction", [(one * (2.0 ** i), action.clone()) for i in range(16)])
    assert result["computed_slots"] == baseline["computed_slots"]
    assert result["max_synthetic_video_relative_l2_error_on_reused_slots"] > 0.93
    named.append(result)
    result, _ = generation("opposing_action_velocities_ignored", [(one.clone(), action * (-1 if i % 2 else 1)) for i in range(16)])
    assert result["computed_slots"] == baseline["computed_slots"]
    assert result["trace"][2]["action_cosines"][0] < -0.99
    named.append(result)
    batched = []
    for i in range(16):
        video = one.repeat(64, 1, 1)
        video[-1].mul_(-1 if i % 2 else 1)
        batched.append((video, action.repeat(64, 1, 1)))
    result, _ = generation("batch_mean_hides_one_opposing_member", batched)
    assert result["computed_slots"] == baseline["computed_slots"]
    assert result["trace"][2]["video_mean_cosine"] > 0.95
    assert result["trace"][2]["video_cosines"][-1] < -0.99
    named.append(result)
    result, _ = generation("divergent_member_alone", [(v[-1:], a[-1:]) for v, a in batched])
    assert result["computed_count"] == 16
    named.append(result)
    for label, value in [("zero_video", 0.0), ("nan_video", float("nan")), ("infinite_video", float("inf"))]:
        result, _ = generation(label, [(torch.full_like(one, value), action.clone()) for _ in range(16)])
        assert result["computed_count"] == 16
        named.append(result)

    isolated = []
    for label, length, countdown, value in [
        ("no_history", 0, 0, 1.0), ("no_history_preserves_old_countdown", 0, 4, 1.0),
        ("one_history_preserves_old_countdown", 1, 4, 1.0),
        ("pending_reuse_does_not_check_nan", 2, 2, float("nan")),
        ("pending_reuse_does_not_check_infinity", 2, 2, float("inf")),
    ]:
        pair = states()
        pair[0].skip_countdown = pair[1].skip_countdown = countdown
        history = [(i, torch.full_like(one, value), action.clone()) for i in range(length)]
        decision, trace = decide(pair, 2, history)
        assert decision == (length < 2)
        assert trace["countdown_after"] == (countdown if length < 2 else 1)
        isolated.append({"name": label, **trace})

    first, pair = generation("first_generation", constant)
    unreset, pair = generation("second_generation_without_countdown_reset", constant, pair, reset=False)
    reset, pair = generation("generation_after_explicit_reset", constant, pair)
    assert first["countdowns_after"][-1] == 1
    assert unreset["computed_slots"] == [0, 1, 2, 7, 12]
    assert reset["computed_slots"] == first["computed_slots"]

    # Disabled Omni means all steps; disabled native dynamic mode selects its fixed mask.
    off_native, off_omni = states()
    off_native.dynamic_cache_schedule = False
    off_omni.config.enabled = False
    disabled_native = [bool(native(off_native, i, 1000 - i * 62.5, [])) for i in range(16)]
    disabled_omni = [bool(omni(off_omni, [])) for _ in range(16)]
    assert disabled_native == mask and all(disabled_omni)

    random_generations = []
    for case in range(RANDOM_GENERATIONS):
        seed = 2026091400 + case
        rng = torch.Generator(device="cpu").manual_seed(seed)
        batch = 1 + case % 4
        video = torch.randn((batch, 2, 4), generator=rng)
        predictions = []
        for slot in range(16):
            perturbation = torch.randn(video.shape, generator=rng)
            if case % 4 == 0:
                video = perturbation
            elif case % 4 == 1:
                video = video + perturbation * 0.03
            elif case % 4 == 2:
                video = video + perturbation * 0.35
            else:
                video = -video if slot % 5 == 0 else video * 1.4 + perturbation * 0.1
            dtype = torch.bfloat16 if case % 5 == 0 else torch.float32
            predictions.append((video.to(dtype), torch.randn((batch, 3, 2), generator=rng).to(dtype)))
        result, _ = generation(f"random_{case:03d}", predictions, full_trace=False)
        result.update(seed=seed, batch_size=batch)
        random_generations.append(result)

    random_histories = []
    for case in range(RANDOM_HISTORIES):
        seed = 2026091600 + case
        rng = torch.Generator(device="cpu").manual_seed(seed)
        pair = states()
        countdown = case % 7
        pair[0].skip_countdown = pair[1].skip_countdown = countdown
        history = []
        video = torch.randn((1 + case % 3, 2, 4), generator=rng)
        for i in range(case % 5):
            video = video + torch.randn(video.shape, generator=rng) * (0.01 if case % 2 else 0.8)
            history.append((i, video.clone(), torch.randn((video.shape[0], 3, 2), generator=rng)))
        _, trace = decide(pair, case % 16, history)
        inputs = [{"video": tensor_record(v), "action": tensor_record(a)} for _, v, a in history]
        random_histories.append({"seed": seed, "input_sha256": digest(canonical(inputs)), **trace})

    assert not torch.cuda.is_initialized()
    assert torch.equal(cpu_rng, torch.get_rng_state())
    model_packages_imported = any(name == package or name.startswith(package + ".")
                                  for name in sys.modules for package in ("groot", "vllm_omni"))
    assert not model_packages_imported
    for path, role in [(args.native_source, "native"), (args.omni_state, "omni_state"), (args.omni_config, "omni_config")]:
        assert digest(path.read_bytes()) == SOURCE_PINS[role]
    report = {
        "status": "passed", "schema_version": 1,
        "scope": "Synthetic CPU parity of upstream proxy decisions and countdowns; no model inference, task-quality, or performance claim",
        "probe_sha256": digest(Path(__file__).read_bytes()),
        "environment": {"torch": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(), "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                        "cuda_initialized_before": False, "cuda_initialized_after": torch.cuda.is_initialized(),
                        "global_cpu_rng_unchanged": True, "model_packages_imported": model_packages_imported},
        "sources": {"native": {**native_meta, "checkout_head_observed": "ab790c198fbce33503358efbbd4187ce9a89adf3", "method": native_method},
                    "omni_state": {**omni_meta, "checkout_head_observed": "f7d9deb45ab56e6a2ccc1690279bd9e6bdefbfe3", "method": omni_method, "reset_method": reset_method},
                    "omni_config": config_meta},
        "common_enabled_config": defaults, "native_shipped_fixed_mask": mask,
        "tuple_mapping": {"native": "(timestep, guided_video_velocity, conditional_action_velocity)",
                          "omni": "(guided_video_velocity,)", "decision_signal": "video velocity only; FP32 cosine mean over batch"},
        "parity_comparisons": comparisons,
        "structured_generations": named, "isolated_history_cases": isolated,
        "generation_reset": {"first": first, "without_reset": unreset, "after_reset": reset,
                             "scope": "without_reset is an intentionally incorrect caller lifecycle, not a claim about normal Omni request handling"},
        "disabled_semantics": {"native_computed_slots": [i for i, x in enumerate(disabled_native) if x],
                               "omni_computed_slots": [i for i, x in enumerate(disabled_omni) if x],
                               "equivalent": False},
        "random_generation_count": len(random_generations), "random_generations": random_generations,
        "random_history_count": len(random_histories), "random_histories": random_histories,
        "findings": [
            "Enabled methods agree under the recorded tuple mapping and common configuration.",
            "Constant synthetic video computes slots 0,1,6,11; unstable video can compute all 16 rather than fixed 8.",
            "Cosine is insensitive to positive magnitude scaling and ignores the action-velocity component.",
            "Batch averaging can hide one opposing member; tested singleton computes all slots.",
            "Zero/nonfinite cosine with zero countdown triggers computation; pending countdown reuse does not inspect finiteness.",
            "Per-generation countdown reset is required; empty history alone leaves old countdown unchanged.",
            "Method parity does not establish equivalent full pipelines, committed state, actions, quality, or latency.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "parity_comparisons": comparisons,
                      "random_generations": len(random_generations), "random_histories": len(random_histories),
                      "cuda_initialized": torch.cuda.is_initialized()}))


if __name__ == "__main__":
    main()
