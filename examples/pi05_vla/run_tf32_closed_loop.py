#!/usr/bin/env python3
"""Preregistered local H100 LIBERO runner for pi0.5 FP32 versus TF32.

The public process is only an orchestrator.  Every (arm, task) is evaluated in a fresh child process
so PyTorch's process-global matmul precision cannot leak across arms.  Both arms load the exact same
v044 snapshot and install the exact same replay-safe static-KV denoiser.  The only intended arithmetic
difference is:

    control_fp32   set_float32_matmul_precision("highest"), allow_tf32=False
    treatment_tf32 set_float32_matmul_precision("high"),    allow_tf32=True

With batch size one, policy.reset() is called once per episode.  We reseed the policy's torch RNG to
``1000 + 50 * task_id + episode_index`` there, independently of how early the preceding episode
terminated. LIBERO is reset with the same unique seed and uses init_state_id == episode_index, hard
resets, and the official stored initial states. This closes the otherwise subtle cross-arm RNG drift
caused by different trajectory lengths consuming different numbers of flow-noise samples.

Example smoke (one task, one pair):

    MUJOCO_GL=egl .venv-libero/bin/python examples/pi05_vla/run_tf32_closed_loop.py \
      --output-dir /tmp/pi05-tf32-smoke --tasks 0 --episodes 1 --gpu <idle-permitted-index>

Full preregistered run (do not start casually):

    MUJOCO_GL=egl .venv-libero/bin/python examples/pi05_vla/run_tf32_closed_loop.py \
      --output-dir /path/to/pi05-tf32-500 --tasks all --episodes 50 --gpu <idle-permitted-index>

The immutable ``paired_run_manifest.json`` is written before the first child starts.  Completed task
rows are merged atomically into per-arm raw JSONL; rerunning the same command skips complete tasks and
reruns incomplete ones from episode zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "tf32_closed_loop_preregistration.json"
TREATMENT_PACKAGE = ROOT / "examples" / "checkpoint" / "pi05-libero-v044-tf32-h100"
DEFAULT_REPO = "lerobot/pi05_libero_finetuned_v044"
DEFAULT_REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
SUITE = "libero_spatial"
ARMS = ("control_fp32", "treatment_tf32")
SOURCE_FILES = (
    Path(__file__).resolve(),
    PREREG,
    TREATMENT_PACKAGE / "config.json",
    TREATMENT_PACKAGE / "instinctflash.json",
    HERE / "pi05_iwm" / "adapter.py",
    HERE / "pi05_iwm" / "passes.py",
    HERE / "pi05_iwm" / "precision.py",
    HERE / "pi05_iwm" / "static_capture.py",
    HERE / "pi05_iwm" / "surface.py",
    HERE / "emit_tf32_closed_loop_outcomes.py",
    HERE / "certify_tf32_closed_loop.py",
    ROOT / "instinctflash" / "runtime" / "facade.py",
    ROOT / "instinctflash" / "verify" / "certify.py",
)
SEED_STRIDE = 50


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_text(path, text)


def _parse_tasks(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(10))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(x) for x in part.split("-", 1))
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(part))
    tasks = sorted(selected)
    if not tasks or tasks[0] < 0 or tasks[-1] > 9:
        raise argparse.ArgumentTypeError("tasks must be 'all' or ids/ranges within 0..9")
    return tasks


def _task_name(task_id: int) -> str:
    return f"task_{task_id:02d}"


def _source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): _sha256(path) for path in SOURCE_FILES}


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _require_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    if result.stdout.strip():
        raise RuntimeError(
            "paired outcomes require a clean committed worktree; commit the runner, preregistration "
            "and numeric stack before starting"
        )


def _checkpoint_fingerprints(checkpoint: Path) -> dict[str, dict[str, Any]]:
    """Hash every checkpoint file that can affect model or processor semantics."""
    names = sorted(
        path.name
        for path in checkpoint.iterdir()
        if path.is_file() and (path.suffix in {".json", ".safetensors", ".model"})
    )
    if "model.safetensors" not in names:
        raise RuntimeError(f"checkpoint {checkpoint} has no model.safetensors")
    return {
        name: {"bytes": (checkpoint / name).stat().st_size, "sha256": _sha256(checkpoint / name)}
        for name in names
    }


def _checkpoint_sizes(checkpoint: Path) -> dict[str, int]:
    return {
        path.name: path.stat().st_size
        for path in checkpoint.iterdir()
        if path.is_file() and (path.suffix in {".json", ".safetensors", ".model"})
    }


def _checkpoint_revision(path: Path, requested: str) -> str:
    # HF snapshots are .../snapshots/<commit>.  A local copy is permitted only when the caller still
    # supplies its immutable revision; the manifest records both the revision and content location.
    if path.parent.name == "snapshots":
        actual = path.name
        if requested and requested != actual:
            raise RuntimeError(f"checkpoint path is revision {actual}, requested {requested}")
        return actual
    if not requested:
        raise RuntimeError("a non-HF local checkpoint requires --checkpoint-revision")
    return requested


def _resolve_checkpoint(value: str, revision: str) -> tuple[Path, str]:
    path = Path(value).expanduser()
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(value, revision=revision))
    path = path.resolve()
    actual_revision = _checkpoint_revision(path, revision)
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"checkpoint {path} is incomplete; missing {missing}")
    return path, actual_revision


def _identity_document(checkpoint: Path, revision: str, tasks: list[int], episodes: int) -> dict:
    declaration = json.loads(PREREG.read_text())
    if not declaration.get("declared_before_paired_run"):
        raise RuntimeError("preregistration is not marked as preceding the paired run")
    if declaration.get("checkpoint", {}).get("revision") != revision:
        raise RuntimeError(
            "checkpoint revision disagrees with preregistration: "
            f"{revision} != {declaration.get('checkpoint', {}).get('revision')}"
        )
    if int(declaration.get("seed_stride", -1)) != SEED_STRIDE:
        raise RuntimeError("runner SEED_STRIDE disagrees with preregistration")
    sources = _source_hashes()
    checkpoint_files = _checkpoint_fingerprints(checkpoint)
    shared = {
        "checkpoint_repo": DEFAULT_REPO,
        "checkpoint_revision": revision,
        "checkpoint_path": str(checkpoint),
        "checkpoint_files": checkpoint_files,
        "static_kv_source": sources["examples/pi05_vla/pi05_iwm/static_capture.py"],
        "runner_source": sources["examples/pi05_vla/run_tf32_closed_loop.py"],
    }
    control = {
        **shared,
        "precision": {"float32_matmul_precision": "highest", "allow_tf32": False},
        "static_kv_graph": True,
    }
    treatment = {
        **shared,
        "precision": {"float32_matmul_precision": "high", "allow_tf32": True},
        "numeric_package": sources[
            "examples/checkpoint/pi05-libero-v044-tf32-h100/instinctflash.json"
        ],
        "numeric_config": sources[
            "examples/checkpoint/pi05-libero-v044-tf32-h100/config.json"
        ],
        "adapter_source": sources["examples/pi05_vla/pi05_iwm/adapter.py"],
        "passes_source": sources["examples/pi05_vla/pi05_iwm/passes.py"],
        "static_kv_graph": True,
        "tier": "NUMERIC",
    }
    return {
        "manifest_version": 1,
        "preregistration_sha256": _sha256(PREREG),
        "source_git_head": _git_head(),
        "source_files": sources,
        "checkpoint": {
            "repo": DEFAULT_REPO,
            "revision": revision,
            "path": str(checkpoint),
            "files": checkpoint_files,
        },
        "protocol": {
            "suite": SUITE,
            "tasks": tasks,
            "episodes_per_task": episodes,
            "batch_size": 1,
            "seed_base": int(declaration["seed_base"]),
            "seed_stride": SEED_STRIDE,
            "seed_rule": declaration["seed_rule"],
            "init_states": True,
            "hard_reset": True,
            "use_async_envs": False,
            "max_parallel_tasks": 1,
            "observation_resolution": [256, 256],
            "arm_order": declaration["arm_order"],
        },
        "identities": {
            "control_hash": f"sha256:{_canonical_sha256(control)}",
            "treatment_hash": f"sha256:{_canonical_sha256(treatment)}",
            "control": control,
            "treatment": treatment,
        },
        "declared_before_first_outcome": True,
    }


def _prepare_manifest(
    output_dir: Path, checkpoint: Path, revision: str, tasks: list[int], episodes: int
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "paired_run_manifest.json"
    expected = _identity_document(checkpoint, revision, tasks, episodes)
    if path.exists():
        manifest = json.loads(path.read_text())
        comparable = {k: v for k, v in manifest.items() if k not in {"run_id", "declared_utc"}}
        if comparable != expected:
            raise RuntimeError(
                f"{path} does not match this source/checkpoint/protocol; use a new output directory"
            )
        return manifest

    preexisting = [output_dir / f"{arm}.raw.jsonl" for arm in ARMS]
    if any(path.exists() and path.stat().st_size for path in preexisting):
        raise RuntimeError("outcomes exist without a prior paired_run_manifest.json; refusing")
    manifest = {**expected, "run_id": str(uuid.uuid4()), "declared_utc": _utc_now()}
    _atomic_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _task_rows_complete(
    rows: list[dict[str, Any]], *, run_id: str, arm: str, task_id: int, episodes: int,
    seed_base: int, seed_stride: int = SEED_STRIDE,
) -> bool:
    chosen = [row for row in rows if row.get("task_id") == task_id]
    if len(chosen) != episodes:
        return False
    for index, row in enumerate(sorted(chosen, key=lambda item: item.get("episode_index", -1))):
        if (
            row.get("run_id") != run_id
            or row.get("arm") != arm
            or row.get("task") != _task_name(task_id)
            or row.get("episode_index") != index
            or row.get("seed") != seed_base + seed_stride * task_id + index
            or not isinstance(row.get("success"), bool)
        ):
            return False
    return True


def _merge_task_rows(path: Path, fresh: list[dict[str, Any]], task_id: int) -> None:
    rows = [row for row in _read_jsonl(path) if row.get("task_id") != task_id]
    rows.extend(fresh)
    rows.sort(key=lambda row: (int(row["task_id"]), int(row["episode_index"])))
    _write_jsonl(path, rows)


@dataclass
class EpisodeSeedSchedule:
    """Reseed policy noise independently at every batch=1 episode boundary."""

    torch_module: Any
    policy: Any
    seed_base: int

    def __post_init__(self) -> None:
        self.seeds: list[int] = []
        self._episode_index = 0
        self._original_reset = self.policy.reset

        def reset() -> Any:
            seed = self.seed_base + self._episode_index
            self.torch_module.manual_seed(seed)
            cuda = getattr(self.torch_module, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "manual_seed_all", None)):
                cuda.manual_seed_all(seed)
            self.seeds.append(seed)
            self._episode_index += 1
            return self._original_reset()

        self.policy.reset = reset


def _precision_assertion(torch_module: Any, arm: str) -> dict[str, Any]:
    precision = torch_module.get_float32_matmul_precision()
    allow = bool(torch_module.backends.cuda.matmul.allow_tf32)
    expected = ("highest", False) if arm == "control_fp32" else ("high", True)
    if (precision, allow) != expected:
        raise RuntimeError(
            f"{arm} precision contract is {(precision, allow)}, expected {expected}"
        )
    return {"float32_matmul_precision": precision, "allow_tf32": allow}


def _configure_fp32_policy(policy_cfg: Any, checkpoint: Path) -> None:
    policy_cfg.pretrained_path = checkpoint
    policy_cfg.pretrained_revision = None
    policy_cfg.compile_model = False
    policy_cfg.device = "cuda"
    # TF32 is a reduced-mantissa execution mode for FP32 GEMMs.  The published v044 deployment
    # config says bfloat16; leaving it untouched would benchmark BF16 and merely label it TF32.
    policy_cfg.dtype = "float32"


def _worker(args: argparse.Namespace) -> int:
    # EGL selection must happen before importing LIBERO/robosuite.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path[:0] = [str(ROOT), str(HERE)]

    manifest = json.loads(Path(args.run_manifest).read_text())
    if manifest["run_id"] != args.run_id:
        raise RuntimeError("worker run_id does not match immutable manifest")
    protocol = manifest["protocol"]
    seed_base = int(protocol["seed_base"])
    seed_stride = int(protocol["seed_stride"])

    if _source_hashes() != manifest["source_files"]:
        raise RuntimeError("source files changed after paired_run_manifest.json was locked")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("paired worker requires CUDA")
    if tuple(torch.cuda.get_device_capability(0)) != (9, 0):
        raise RuntimeError(f"TF32 operating point requires SM90, got {torch.cuda.get_device_capability(0)}")

    from lerobot.envs import make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.scripts.lerobot_eval import eval_policy_all

    checkpoint = Path(args.worker_checkpoint)
    expected_sizes = {
        name: int(metadata["bytes"])
        for name, metadata in manifest["checkpoint"]["files"].items()
    }
    if _checkpoint_sizes(checkpoint) != expected_sizes:
        raise RuntimeError("checkpoint files changed after paired_run_manifest.json was locked")
    env_cfg = LiberoEnvConfig(
        task=SUITE,
        task_ids=[args.worker_task_id],
        init_states=True,
        hard_reset=True,
        max_parallel_tasks=1,
        observation_height=256,
        observation_width=256,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs[SUITE][args.worker_task_id]
    base_env = vec_env.envs[0]
    if not (
        base_env.init_states
        and base_env.hard_reset
        and base_env.init_state_id == 0
        and base_env._reset_stride == 1
    ):
        raise RuntimeError("LIBERO did not construct the preregistered init-state schedule")

    policy_cfg = PI05Config.from_pretrained(checkpoint)
    _configure_fp32_policy(policy_cfg, checkpoint)
    if policy_cfg.chunk_size != 50 or policy_cfg.n_action_steps != 50:
        raise RuntimeError(
            f"v044 action geometry changed: chunk={policy_cfg.chunk_size}, "
            f"n_action_steps={policy_cfg.n_action_steps}"
        )

    from pi05_iwm.precision import Pi05PrecisionLease

    mode = "fp32" if args.worker_arm == "control_fp32" else "tf32"
    lease = Pi05PrecisionLease(torch, mode)
    try:
        precision = _precision_assertion(torch, args.worker_arm)
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
        policy.eval()
        non_fp32 = [
            (name, str(parameter.dtype))
            for name, parameter in policy.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if non_fp32:
            raise RuntimeError(
                "TF32 operating point requires FP32 floating parameters; found "
                f"{non_fp32[:8]}{' ...' if len(non_fp32) > 8 else ''}"
            )

        if args.worker_arm == "treatment_tf32":
            import instinctflash
            from instinctflash.runtime.facade import plan_declaration
            from pi05_iwm.adapter import Pi05Adapter, _validate_tf32_plan

            if "pi05" not in instinctflash.available_models():
                instinctflash.register("pi05", Pi05Adapter)
            checkpoint_decl, adapter, plan, _device = plan_declaration(
                TREATMENT_PACKAGE, tier_ceiling="numeric"
            )
            _validate_tf32_plan(checkpoint_decl, plan)
            if "plan tier: NUMERIC" not in plan.explain():
                raise RuntimeError("treatment planner did not expose NUMERIC")
            print(plan.explain(), flush=True)
            installed = adapter.install(policy, plan, device="cuda")
            if "pi05_tf32_numeric" not in installed:
                raise RuntimeError(f"formal TF32 adapter was not installed: {installed}")
        else:
            # The control uses the same replay-safe static-KV implementation.  Installing it directly
            # avoids the legacy environment opt-in and makes the arm difference visibly arithmetic-only.
            from pi05_iwm.static_capture import install_static_capture
            from pi05_iwm.surface import Pi05Surface

            Pi05Surface(policy.model).hoist_loop_constants()
            install_static_capture(policy.model, step_tables=True)
            installed = ["loop_constant_hoist", "graph_capture_static_kv"]

        if getattr(policy.model, "_ifl_static_denoiser", None) is None:
            raise RuntimeError(f"{args.worker_arm} is missing the shared static-KV denoiser")
        precision = _precision_assertion(torch, args.worker_arm)

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_cfg
        )

        task_seed_base = seed_base + seed_stride * args.worker_task_id
        schedule = EpisodeSeedSchedule(torch, policy, task_seed_base)
        action_trace = hashlib.sha256()
        original_select_action = policy.select_action

        def traced_select_action(*select_args, **select_kwargs):
            action = original_select_action(*select_args, **select_kwargs)
            tensor = action.detach().to(device="cpu", dtype=torch.float32).contiguous()
            action_trace.update(str(tuple(tensor.shape)).encode())
            action_trace.update(tensor.numpy().tobytes())
            return action

        policy.select_action = traced_select_action
        with torch.no_grad():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.worker_episodes,
                max_episodes_rendered=0,
                return_episode_data=False,
                start_seed=task_seed_base,
                max_parallel_tasks=1,
            )
        expected_seeds = [task_seed_base + index for index in range(args.worker_episodes)]
        if schedule.seeds != expected_seeds:
            raise RuntimeError(f"policy episode seeds drifted: {schedule.seeds} != {expected_seeds}")
        if base_env.init_state_id != args.worker_episodes:
            raise RuntimeError(
                f"LIBERO init-state schedule drifted: next id {base_env.init_state_id}, "
                f"expected {args.worker_episodes}"
            )
        entries = info.get("per_task", [])
        if len(entries) != 1 or int(entries[0]["task_id"]) != args.worker_task_id:
            raise RuntimeError(f"unexpected evaluator task result: {entries}")
        successes = entries[0]["metrics"]["successes"]
        if len(successes) != args.worker_episodes:
            raise RuntimeError(
                f"evaluator returned {len(successes)} episodes, expected {args.worker_episodes}"
            )

        rows = [
            {
                "run_id": args.run_id,
                "arm": args.worker_arm,
                "suite": SUITE,
                "task_id": args.worker_task_id,
                "task": _task_name(args.worker_task_id),
                "episode_index": index,
                "seed": task_seed_base + index,
                "init_state_id": index,
                "policy_seed": task_seed_base + index,
                "success": bool(success),
                "checkpoint_revision": manifest["checkpoint"]["revision"],
            }
            for index, success in enumerate(successes)
        ]
        static = policy.model._ifl_static_denoiser
        result = {
            "rows": rows,
            "precision": precision,
            "installed": installed,
            "static_kv_replays": int(static.replays),
            "action_trace_sha256": action_trace.hexdigest(),
            "plan_tier": "NUMERIC" if args.worker_arm == "treatment_tf32" else "BITEXACT",
        }
        _atomic_text(Path(args.worker_result), json.dumps(result, indent=2, sort_keys=True) + "\n")
    finally:
        lease.close()
    return 0


def _worker_command(
    args: argparse.Namespace,
    *,
    arm: str,
    task_id: int,
    checkpoint: Path,
    manifest: Path,
    run_id: str,
    result: Path,
    episodes: int | None = None,
) -> list[str]:
    return [
        args.python,
        str(Path(__file__).resolve()),
        "--worker-arm",
        arm,
        "--worker-task-id",
        str(task_id),
        "--worker-episodes",
        str(args.episodes if episodes is None else episodes),
        "--worker-checkpoint",
        str(checkpoint),
        "--worker-result",
        str(result),
        "--run-manifest",
        str(manifest),
        "--run-id",
        run_id,
    ]


def _null_projection(payload: dict[str, Any]) -> dict[str, Any]:
    """Fields that must repeat exactly in a same-arm closed-loop null control."""
    return {
        "rows": [
            {
                key: row[key]
                for key in (
                    "arm",
                    "suite",
                    "task_id",
                    "task",
                    "episode_index",
                    "seed",
                    "init_state_id",
                    "policy_seed",
                    "success",
                    "checkpoint_revision",
                )
            }
            for row in payload["rows"]
        ],
        "precision": payload["precision"],
        "installed": payload["installed"],
        "action_trace_sha256": payload["action_trace_sha256"],
        "plan_tier": payload["plan_tier"],
    }


def _null_summary_valid(summary: dict[str, Any], manifest: dict[str, Any]) -> bool:
    if summary.get("run_id") != manifest.get("run_id") or summary.get("passed") is not True:
        return False
    expected_precision = {
        "control_fp32": {"float32_matmul_precision": "highest", "allow_tf32": False},
        "treatment_tf32": {"float32_matmul_precision": "high", "allow_tf32": True},
    }
    expected_installed = {
        "control_fp32": ["loop_constant_hoist", "graph_capture_static_kv"],
        "treatment_tf32": [
            "loop_constant_hoist", "graph_capture_static_kv", "pi05_tf32_numeric"
        ],
    }
    for arm in ARMS:
        result = summary.get("results", {}).get(arm, {})
        left, right = result.get("repeat_0"), result.get("repeat_1")
        if result.get("passed") is not True or not isinstance(left, dict) or left != right:
            return False
        if left.get("precision") != expected_precision[arm]:
            return False
        if left.get("installed") != expected_installed[arm]:
            return False
        if left.get("plan_tier") != ("NUMERIC" if arm == "treatment_tf32" else "BITEXACT"):
            return False
        trace = left.get("action_trace_sha256")
        if not isinstance(trace, str) or len(trace) != 64:
            return False
        try:
            int(trace, 16)
        except ValueError:
            return False
        rows = left.get("rows")
        if not isinstance(rows, list) or len(rows) != 1:
            return False
        row = rows[0]
        if (
            row.get("arm") != arm
            or row.get("suite") != SUITE
            or row.get("task_id") != 0
            or row.get("task") != "task_00"
            or row.get("episode_index") != 0
            or row.get("seed") != int(manifest["protocol"]["seed_base"])
            or row.get("init_state_id") != 0
            or row.get("policy_seed") != int(manifest["protocol"]["seed_base"])
            or row.get("checkpoint_revision") != manifest["checkpoint"]["revision"]
            or not isinstance(row.get("success"), bool)
        ):
            return False
    return True


def _ensure_null_controls(
    args: argparse.Namespace, output_dir: Path, checkpoint: Path, manifest: dict[str, Any]
) -> None:
    summary_path = output_dir / "null_controls.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if not _null_summary_valid(summary, manifest):
            raise RuntimeError("existing null_controls.json is not a passing control for this run")
        return
    if any((output_dir / f"{arm}.raw.jsonl").exists() for arm in ARMS):
        raise RuntimeError("paired outcomes exist before required null controls; refusing")

    null_root = output_dir / "null_controls"
    results: dict[str, Any] = {}
    for arm in ARMS:
        repeats = []
        for repeat in range(2):
            repeat_dir = null_root / arm / f"repeat_{repeat}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            result_path = repeat_dir / "worker_result.json"
            command = _worker_command(
                args,
                arm=arm,
                task_id=0,
                checkpoint=checkpoint,
                manifest=output_dir / "paired_run_manifest.json",
                run_id=manifest["run_id"],
                result=result_path,
                episodes=1,
            )
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": os.pathsep.join(
                    [str(ROOT), str(HERE), os.environ.get("PYTHONPATH", "")]
                ),
            }
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            _atomic_text(repeat_dir / "worker.log", completed.stdout + completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{arm} null repeat {repeat} failed with rc={completed.returncode}; "
                    f"see {repeat_dir / 'worker.log'}"
                )
            repeats.append(json.loads(result_path.read_text()))
        left, right = (_null_projection(payload) for payload in repeats)
        if left != right:
            raise RuntimeError(
                f"{arm} same-seed null repeats disagree; closed-loop harness is not reproducible"
            )
        results[arm] = {"repeat_0": left, "repeat_1": right, "passed": True}
    summary = {
        "run_id": manifest["run_id"],
        "completed_utc": _utc_now(),
        "protocol": "task 0 episode 0 repeated in two fresh processes per arm",
        "results": results,
        "passed": True,
    }
    if not _null_summary_valid(summary, manifest):
        raise RuntimeError("internal error: generated null-control summary failed schema validation")
    _atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _run_parent(args: argparse.Namespace) -> int:
    _require_clean_worktree()
    tasks = _parse_tasks(args.tasks)
    if args.episodes <= 0 or args.episodes > 50:
        raise RuntimeError("--episodes must be within 1..50")
    checkpoint, revision = _resolve_checkpoint(args.checkpoint, args.checkpoint_revision)
    output_dir = args.output_dir.resolve()
    manifest = _prepare_manifest(output_dir, checkpoint, revision, tasks, args.episodes)
    manifest_path = output_dir / "paired_run_manifest.json"
    seed_base = int(manifest["protocol"]["seed_base"])
    seed_stride = int(manifest["protocol"]["seed_stride"])
    _ensure_null_controls(args, output_dir, checkpoint, manifest)

    schedule = []
    for task_id in tasks:
        selected_arms = ARMS if args.arms == "both" else (args.arms,)
        if args.arms == "both" and task_id % 2:
            selected_arms = tuple(reversed(selected_arms))
        schedule.extend((arm, task_id) for arm in selected_arms)
    for arm, task_id in schedule:
        raw_path = output_dir / f"{arm}.raw.jsonl"
        current = _read_jsonl(raw_path)
        if _task_rows_complete(
            current,
            run_id=manifest["run_id"],
            arm=arm,
            task_id=task_id,
            episodes=args.episodes,
            seed_base=seed_base,
            seed_stride=seed_stride,
        ):
            print(f"[{arm}] task {task_id}: complete, resume skips it", flush=True)
            continue

        task_dir = output_dir / arm / f"task_{task_id:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        result_path = task_dir / "worker_result.json"
        command = _worker_command(
            args,
            arm=arm,
            task_id=task_id,
            checkpoint=checkpoint,
            manifest=manifest_path,
            run_id=manifest["run_id"],
            result=result_path,
        )
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), str(HERE), os.environ.get("PYTHONPATH", "")]
            ),
        }
        print(f"[{arm}] task {task_id}: launching isolated worker", flush=True)
        completed = subprocess.run(command, env=env, capture_output=True, text=True)
        _atomic_text(task_dir / "worker.log", completed.stdout + completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{arm} task {task_id} failed with rc={completed.returncode}; "
                f"see {task_dir / 'worker.log'}"
            )
        payload = json.loads(result_path.read_text())
        rows = payload["rows"]
        if not _task_rows_complete(
            rows,
            run_id=manifest["run_id"],
            arm=arm,
            task_id=task_id,
            episodes=args.episodes,
            seed_base=seed_base,
            seed_stride=seed_stride,
        ):
            raise RuntimeError(f"{arm} task {task_id} worker emitted an invalid schedule")
        _merge_task_rows(raw_path, rows, task_id)
        print(
            f"[{arm}] task {task_id}: {sum(row['success'] for row in rows)}/{len(rows)}; "
            f"precision={payload['precision']}; static replays={payload['static_kv_replays']}",
            flush=True,
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tasks", default="all", help="all, comma-separated ids, or ranges (e.g. 0,2-4)")
    parser.add_argument("--episodes", type=int, default=50, help="episodes per selected task (1..50)")
    parser.add_argument("--arms", choices=("both", *ARMS), default="both")
    parser.add_argument("--checkpoint", default=DEFAULT_REPO)
    parser.add_argument("--checkpoint-revision", default=DEFAULT_REVISION)
    # REQUIRED, no default: '0' silently targeted a training-reserved H100 on this fleet. The
    # device is a per-run decision the operator must make (never GPUs 0-6 here; see the fleet
    # rules), not something a benchmark script may assume.
    parser.add_argument("--gpu", required=True,
                        help="CUDA device index for both arms (required; no default on purpose)")
    parser.add_argument("--python", default=sys.executable, help="Python executable for isolated workers")

    # Private worker protocol; intentionally not a user-facing alternate operating-point selector.
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-task-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-episodes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--run-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    random.seed(0)  # orchestration-only names/order; policy RNG is reseeded per episode in the worker.
    if args.worker_arm:
        required = (
            args.worker_task_id,
            args.worker_episodes,
            args.worker_checkpoint,
            args.worker_result,
            args.run_manifest,
            args.run_id,
        )
        if any(value is None for value in required):
            raise RuntimeError("incomplete private worker invocation")
        return _worker(args)
    if args.output_dir is None:
        raise RuntimeError("--output-dir is required")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
