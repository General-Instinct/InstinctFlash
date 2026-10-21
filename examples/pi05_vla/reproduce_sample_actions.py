"""Compare stock and InstinctFlash pi0.5 core latency during live LIBERO episodes.

Times sample_actions() only. Observation preprocessing, action postprocessing,
simulator stepping, and diagnostic logging are excluded. Both arms use the same
checkpoint, episode seeds, and initial-state sequence. Matching episode starts
does not establish per-call output equality across separate live rollouts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(
    0, str(HERE.parents[1])
)  # repo root, so pi05_iwm can import instinctflash

from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import bind_relative_anchor
from pi05_iwm.static_capture import WARMUP_STEPS, install_static_capture
from pi05_iwm.surface import Pi05Surface

CKPT = os.environ.get("IFL_PI05_CKPT", "lerobot/pi05_libero_finetuned_v044")
DEV = "cuda:0"


def prepare_model_inputs(policy, batch: dict[str, torch.Tensor]):
    """Prepare model images and retrieve tokenized task inputs

    Returns:
        A tuple of images, camera-validity masks, language token IDs, and language-validity masks
    """

    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]

    return images, img_masks, tokens, masks


def create_env(suite, task_id):
    """Create one synchronous environment for the selected LIBERO task.

    Returns:
        The environment configuration and suite/task-indexed environment mapping
    """

    env_cfg = make_env_config("libero", task=suite, task_ids=[task_id])
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    return env_cfg, envs


def warmup(model, images, image_masks, tokens, masks, noise, n=2):
    """Run unmeasured chunk predictions without advancing the simulator.

    Reuses the prepared observation and supplied noise. With IF installed, these
    calls also advance capture warmup; the caller must verify capture admission.
    """

    print(
        f"[warmup] Starting {n} unmeasured chunk predictions (no simulator steps)",
        flush=True,
    )
    for index in range(n):
        actions = model.sample_actions(images, image_masks, tokens, masks, noise=noise)
        if not torch.isfinite(actions).all().item():
            raise RuntimeError("Warmup produced nonfinite actions")
        print(f"[warmup] Chunk {index + 1}/{n} complete", flush=True)
    torch.cuda.synchronize(tokens.device)


def timed_sample_actions(model, images, image_masks, tokens, masks):
    """Measure one complete sample_actions() call using CUDA synchronization.

    Includes noise generation, prefix embedding/prefill, and denoising integration.
    Excludes observation preparation, action postprocessing, and simulator stepping.

    Returns:
        Elapsed wall-clock milliseconds and the predicted padded action chunk.
    """

    torch.cuda.synchronize(tokens.device)
    t0 = time.perf_counter()
    actions = model.sample_actions(images, image_masks, tokens, masks, noise=None)
    torch.cuda.synchronize(tokens.device)
    latency_ms = (time.perf_counter() - t0) * 1000
    return latency_ms, actions


def episode_success(info):
    """Extract success from direct or terminal vector-environment metadata.

    Returns False when no success information is available. Supports one environment.
    """

    value = info.get("is_success")
    if value is not None:
        return bool(np.asarray(value).reshape(-1)[0])
    final = info.get("final_info")
    if isinstance(final, dict):
        value = final.get("is_success")
        return bool(np.asarray(value).reshape(-1)[0]) if value is not None else False
    if final is not None and len(final) and isinstance(final[0], dict):
        return bool(final[0].get("is_success", False))
    return False


def run_episode(
    policy,
    env,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seed,
    arm="stock",
    episode_index=0,
    n_episodes=1,
    initial_state_index=None,
):
    """Execute one live LIBERO episode and collect fresh-chunk inference latencies.

    Resets the policy queue, seeds PyTorch, and resets the environment. Generates
    and times a new chunk only when the queue is empty, then postprocesses and
    executes queued actions until termination, truncation, or the step limit.

    Args:
        policy: Loaded pi0.5 policy using either stock or IF denoising.
        env: Single-environment vector wrapper.
        env_preprocessor: LIBERO observation-processing pipeline.
        env_postprocessor: Environment action-processing pipeline.
        preprocessor: Checkpoint observation-processing pipeline.
        postprocessor: Checkpoint action-processing pipeline.
        seed: Seed for PyTorch noise generation and environment reset.
        arm: Label identifying the benchmark variant in progress logs.
        episode_index: Zero-based episode index used for logging.
        n_episodes: Total episodes in the arm, used for logging.
        initial_state_index: Explicit LIBERO initial-state schedule index for this episode.

    Returns:
        Episode outcomes, initial-state identity, and per-prediction timing records.

    Note:
        The caller controls LIBERO's initial-state index separately from reset seeds.
    """

    label = f"[{arm} episode {episode_index + 1}/{n_episodes}]"
    policy.reset()
    for processor in (env_preprocessor, env_postprocessor, preprocessor, postprocessor):
        processor.reset()
    bind_relative_anchor(policy, preprocessor)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if initial_state_index is not None:
        env.envs[0].unwrapped.init_state_id = initial_state_index
    init_state_id = env.envs[0].unwrapped.init_state_id
    init_states = getattr(env.envs[0].unwrapped, "_init_states", None)
    resolved_init_state_id = (
        int(init_state_id % len(init_states)) if init_states is not None else None
    )
    print(f"{label} Reset: seed={seed}, init_state_index={init_state_id}", flush=True)
    observation, info = env.reset(seed=[seed])

    times = []
    samples = []
    terminated_flag = False
    truncated_flag = False
    done = False
    step = 0
    max_steps = env.call("_max_episode_steps")[0]
    success = False

    while not done and step < max_steps:
        obs = preprocess_observation(observation)
        task_desc = list(env.call("task_description"))
        obs["task"] = task_desc
        batch = preprocessor(env_preprocessor(obs))

        if len(policy._action_queue) == 0:
            images, image_masks, tokens, token_masks = prepare_model_inputs(
                policy, batch
            )
            driver = getattr(policy.model, "_ifl_static_denoiser", None)
            if arm == "IF":
                require_capture(driver)
            replays_before = driver.replays if driver is not None else 0
            latency_ms, actions = timed_sample_actions(
                policy.model, images, image_masks, tokens, token_masks
            )
            times.append(latency_ms)
            finite = bool(torch.isfinite(actions).all().item())
            expected_shape = (
                tokens.shape[0],
                policy.config.chunk_size,
                policy.config.max_action_dim,
            )
            if not finite or tuple(actions.shape) != expected_shape:
                raise RuntimeError(
                    f"{label} Invalid predicted actions: shape={tuple(actions.shape)}, finite={finite}"
                )
            if not math.isfinite(latency_ms) or latency_ms <= 0:
                raise RuntimeError(f"{label} Invalid latency: {latency_ms}")
            if arm == "IF":
                require_capture(driver)
                if driver.replays <= replays_before:
                    raise RuntimeError(
                        f"{label} IF produced a prediction without graph replay"
                    )
            samples.append(
                {
                    "arm": arm,
                    "episode": episode_index,
                    "seed": seed,
                    "control_step": step,
                    "chunk_index": len(times) - 1,
                    "latency_ms": latency_ms,
                    "output_shape": list(actions.shape),
                    "finite": finite,
                    "graph_replays": driver.replays - replays_before
                    if driver is not None
                    else 0,
                    "capture_rejected": bool(driver.rejected)
                    if driver is not None
                    else False,
                }
            )
            # Trim internal padding and retain the execution horizon.
            action_dim = policy.config.output_features["action"].shape[0]
            actions = actions[:, : policy.config.n_action_steps, :action_dim]
            policy._action_queue.extend(actions.transpose(0, 1))

        action = policy._action_queue.popleft()

        # Convert normalized model output into simulator actions.
        action = postprocessor(action)
        action = env_postprocessor({"action": action})["action"]
        action_numpy = action.cpu().numpy()
        if action_numpy.shape != (1, action_dim) or not np.isfinite(action_numpy).all():
            raise RuntimeError(f"{label} Invalid postprocessed simulator action")

        observation, _reward, terminated, truncated, info = env.step(action_numpy)
        done = bool(terminated[0] or truncated[0])
        terminated_flag = bool(terminated[0])
        truncated_flag = bool(truncated[0])
        step += 1
        success = success or episode_success(info)

    if not times:
        raise RuntimeError(f"{label} Episode produced no fresh-call samples")
    reason = (
        "success"
        if success
        else "terminated"
        if terminated_flag
        else "truncated"
        if truncated_flag
        else "step limit"
    )
    print(
        f"{label} DONE: {reason}, env_steps={step}, fresh_chunks={len(times)}, "
        f"median_sample_actions={statistics.median(times):.2f} ms",
        flush=True,
    )
    return {
        "arm": arm,
        "episode": episode_index,
        "seed": seed,
        "initial_state_index": int(init_state_id),
        "resolved_initial_state_index": resolved_init_state_id,
        "success": success,
        "environment_steps": step,
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "termination_reason": reason,
        "fresh_chunks": len(samples),
        "latency_ms": times,
        "samples": samples,
    }


def latency_summary(latencies):
    """Summarize raw fresh-call latencies without averaging episode medians."""
    if not latencies or any(
        not math.isfinite(value) or value <= 0 for value in latencies
    ):
        raise ValueError("Latency samples must be nonempty, positive, and finite")
    return {
        "samples": len(latencies),
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.mean(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def require_capture(driver):
    """Fail rather than label an eager fallback as optimized inference."""
    if driver is None or driver.rejected:
        raise RuntimeError("IF static capture is missing or was rejected")
    if (
        driver.replays < 1
        or not driver.self_check
        or driver.self_check.get("bitexact") is not True
    ):
        raise RuntimeError("IF graph has not passed its exactness check and replayed")


def save_report(output_dir, report):
    """Atomically replace this run's report inside an exclusively created directory."""
    if output_dir is None:
        return
    path = output_dir / "results.json"
    temporary = output_dir / "results.json.tmp"
    try:
        with temporary.open("x") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_evidence(driver):
    """Describe the graph mode actually installed and its correctness/replay evidence."""
    return {
        "step_tables": bool(driver._step_tables),
        "full_chunk_graph": bool(driver._full_chunk),
        "prefix_graph": bool(driver._prefix_graph_enabled),
        "denoise_replays": int(driver.replays),
        "chunk_replays": int(driver.chunk_replays),
        "prefix_replays": int(driver.prefix_replays),
        "rejected": bool(driver.rejected),
        "self_check": driver.self_check,
    }


def parse_args(argv=None):
    """Parse and validate live benchmark options without loading a model."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare stock and InstinctFlash sample_actions() latency during live "
            "LIBERO episodes. Runs stock first, then IF, on one CUDA GPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=CKPT,
        help="Hugging Face checkpoint ID or local directory",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Checkpoint revision; use a full commit SHA for reproducibility",
    )
    parser.add_argument(
        "--suite",
        choices=("libero_object", "libero_spatial", "libero_goal", "libero_10"),
        default="libero_object",
        help="LIBERO task suite",
    )
    parser.add_argument(
        "--task",
        "--task-id",
        dest="task_id",
        type=int,
        choices=range(10),
        default=0,
        help="Task index within the suite",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Measured episodes per arm"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="First episode seed; incremented for each episode",
    )
    parser.add_argument(
        "--device",
        default=DEV,
        help="CUDA device, e.g. cuda:0 (relative to CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=2,
        help="Unmeasured stock chunk predictions",
    )
    parser.add_argument(
        "--if-warmup-chunks",
        type=int,
        default=3,
        help="Unmeasured IF chunk predictions; must complete capture before measurement",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New directory for results.json, including raw per-episode latencies; omit for terminal only",
    )
    args = parser.parse_args(argv)
    for name in ("episodes", "warmup_chunks", "if_warmup_chunks"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.seed <= 2**32 - args.episodes:
        parser.error("episode seeds must fit within the unsigned 32-bit range")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError):
        parser.error("--device must be a CUDA device such as cuda:0")
    if device.type != "cuda":
        parser.error("--device must be a CUDA device such as cuda:0")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("--output-dir must not already exist; choose a new directory")
    return args


def main(argv=None):
    """Load one checkpoint and evaluate both arms, closing environments on every exit."""
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("This benchmark requires a CUDA GPU.")
        return 2
    torch.cuda.set_device(torch.device(args.device))
    # Use one checkpoint location for weights and checkpoint processors.
    if Path(args.checkpoint).is_dir():
        if args.revision is not None:
            raise ValueError(
                "--revision applies to Hub checkpoint IDs, not local directories"
            )
        checkpoint_path = str(Path(args.checkpoint).resolve())
    else:
        from huggingface_hub import snapshot_download

        checkpoint_path = snapshot_download(args.checkpoint, revision=args.revision)
    config = PI05Config.from_pretrained(checkpoint_path)
    config.compile_model = False
    config.device = args.device
    if config.num_inference_steps <= 0 or config.n_action_steps <= 0:
        raise ValueError("Inference step count and action horizon must be positive")
    if config.n_action_steps > config.chunk_size:
        raise ValueError(
            "Action execution horizon cannot exceed the predicted chunk size"
        )
    if (
        getattr(config, "use_visual_memory", False)
        or getattr(config, "use_proprioceptive_memory", False)
        or (config.rtc_config is not None and config.rtc_config.enabled)
    ):
        raise ValueError("This benchmark currently supports pi0.5 without MEM or RTC")
    minimum_if_warmup = WARMUP_STEPS // config.num_inference_steps + 1
    if args.if_warmup_chunks < minimum_if_warmup:
        raise ValueError(
            f"--if-warmup-chunks must be at least {minimum_if_warmup} "
            f"to exceed {WARMUP_STEPS} denoising warmup steps"
        )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[config] {json.dumps(vars(args), default=str)}", flush=True)
    print(f"Loading {checkpoint_path} ...", flush=True)
    policy = (
        PI05Policy.from_pretrained(checkpoint_path, config=config)
        .to(args.device)
        .eval()
    )
    model = policy.model
    envs = {}
    report = {
        "schema_version": 1,
        "status": "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "stock_episodes": [],
        "if_episodes": [],
    }
    capture_driver = None
    try:
        save_report(args.output_dir, report)
        env_cfg, envs = create_env(args.suite, args.task_id)
        env = envs[args.suite][args.task_id]
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg, policy.config
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        for processor in (
            env_preprocessor,
            env_postprocessor,
            preprocessor,
            postprocessor,
        ):
            processor.reset()
        bind_relative_anchor(policy, preprocessor)

        # This observation is used only for warmup, never as a measured live episode.
        observation, _ = env.reset(seed=[args.seed])
        obs = preprocess_observation(observation)
        obs["task"] = list(env.call("task_description"))
        batch = preprocessor(env_preprocessor(obs))
        images, image_masks, tokens, token_masks = prepare_model_inputs(policy, batch)
        shape = (1, model.config.chunk_size, model.config.max_action_dim)
        warmup_noise = model.sample_noise(shape, torch.device(args.device))
        warmup(
            model,
            images,
            image_masks,
            tokens,
            token_masks,
            noise=warmup_noise,
            n=args.warmup_chunks,
        )
        initial_state_id = int(env.envs[0].unwrapped.init_state_id)
        report["initial_state_index"] = initial_state_id
        print("WARMUP FOR STOCK DONE", flush=True)

        for episode in range(args.episodes):
            result = run_episode(
                policy,
                env,
                env_preprocessor,
                env_postprocessor,
                preprocessor,
                postprocessor,
                seed=args.seed + episode,
                arm="stock",
                episode_index=episode,
                n_episodes=args.episodes,
                initial_state_index=initial_state_id + episode,
            )
            report["stock_episodes"].append(result)
            save_report(args.output_dir, report)

        Pi05Surface(model).hoist_loop_constants()
        capture_driver = install_static_capture(model)
        warmup(
            model,
            images,
            image_masks,
            tokens,
            token_masks,
            noise=warmup_noise,
            n=args.if_warmup_chunks,
        )
        require_capture(capture_driver)
        report["if_capture_after_warmup"] = capture_evidence(capture_driver)
        print(
            "[IF] Warmup complete; graph replay and exactness check passed.", flush=True
        )
        save_report(args.output_dir, report)

        for episode in range(args.episodes):
            result = run_episode(
                policy,
                env,
                env_preprocessor,
                env_postprocessor,
                preprocessor,
                postprocessor,
                seed=args.seed + episode,
                arm="IF",
                episode_index=episode,
                n_episodes=args.episodes,
                initial_state_index=initial_state_id + episode,
            )
            stock_result = report["stock_episodes"][episode]
            if (result["seed"], result["resolved_initial_state_index"]) != (
                stock_result["seed"],
                stock_result["resolved_initial_state_index"],
            ):
                raise RuntimeError("Stock and IF episode starts are not paired")
            report["if_episodes"].append(result)
            save_report(args.output_dir, report)

        require_capture(capture_driver)
        stock_times = [
            value
            for episode in report["stock_episodes"]
            for value in episode["latency_ms"]
        ]
        if_times = [
            value
            for episode in report["if_episodes"]
            for value in episode["latency_ms"]
        ]
        stock_stats = latency_summary(stock_times)
        if_stats = latency_summary(if_times)
        ms_stock, ms_if = stock_stats["median_ms"], if_stats["median_ms"]
        speedup = ms_stock / ms_if
        summary = {
            "checkpoint": args.checkpoint,
            "checkpoint_path": str(checkpoint_path),
            "revision": args.revision,
            "protocol": (
                f"live {args.suite} task {args.task_id}; {args.episodes} episodes per arm; "
                "median of all fresh sample_actions latencies; warmup excluded"
            ),
            "stock_samples": len(stock_times),
            "if_samples": len(if_times),
            "torch_eager_chunk_ms": round(ms_stock, 1),
            "instinctflash_static_capture_chunk_ms": round(ms_if, 1),
            "speedup": round(speedup, 2),
            "time_saved_ms": ms_stock - ms_if,
            "latency_reduction_percent": 100 * (1 - ms_if / ms_stock),
            "stock_latency": stock_stats,
            "if_latency": if_stats,
            "stock_successes": sum(
                episode["success"] for episode in report["stock_episodes"]
            ),
            "if_successes": sum(
                episode["success"] for episode in report["if_episodes"]
            ),
            "step_tables": bool(capture_driver._step_tables),
        }
        report.update(
            status="complete",
            summary=summary,
            if_capture=capture_evidence(capture_driver),
        )
        save_report(args.output_dir, report)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"\npi05  {ms_stock:.1f} -> {ms_if:.1f} ms   {speedup:.2f}x", flush=True)
        if args.output_dir is not None:
            print(f"Saved {args.output_dir / 'results.json'}", flush=True)
        return 0
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if capture_driver is not None:
            report["if_capture"] = capture_evidence(capture_driver)
        try:
            save_report(args.output_dir, report)
        except (OSError, TypeError, ValueError) as save_error:
            print(
                f"Could not save failure report: {save_error}",
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        for tasks in envs.values():
            for env in tasks.values():
                env.close()


if __name__ == "__main__":
    raise SystemExit(main())
