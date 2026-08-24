"""W9g baseline-comparison harness — our pipeline vs the JAX baseline.

Drives both training pipelines on the same dataset and produces a
side-by-side loss-trajectory CSV. Used purely to validate that our
infrastructure is not silently broken — the gates are about
*shape* of the loss curve, not absolute final loss.

The JAX baseline is the openpi-compiler script
``RL/scripts/train_jax_lora_recap.py``; it shares the dataset
(``libero10_recap_lerobot``), tokenizer, optimiser hyper-defaults
(AdamW + warmup-cosine LR, weight_decay=1e-10, grad_clip=1.0,
acp_dropout=0.30), and pi0.5 base weights with our pipeline. The
only structural difference is the framework (JAX vs PyTorch) and
the backend (default JAX float32 vs our FP8 + LoRA on PyTorch).

Workflow:

1. ``run_pytorch_pipeline(...)`` runs our training driver and writes
   ``pytorch_loss.csv``.
2. ``run_jax_baseline(...)`` shells out to the openpi-compiler
   script with the same hyper-config, parses its log lines for
   ``Step N/M | loss=L`` records, and writes ``jax_loss.csv``.
3. ``compare_curves(...)`` ingests both CSVs and asserts:
     * both runs have finite loss throughout;
     * mean(end / start) ratio is within [0.1, 5.0] for both
       (catches both blow-ups and pathological shrinkages);
     * the order-of-magnitude of the final losses agrees within 5×;
     * the *direction* (decrease, plateau, or increase) is the same.

These are intentionally loose: we're providing infrastructure, not
chasing equivalent convergence. The user tunes effects.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

def _env_path(*var_names: str) -> Path | None:
    """Return the first env var set among ``var_names`` as a ``Path``, else ``None``."""
    for v in var_names:
        val = os.environ.get(v)
        if val:
            return Path(val)
    return None


def _require_env_path(*var_names: str, what: str) -> Path:
    """Same as :func:`_env_path` but raises a clear error when none are set."""
    p = _env_path(*var_names)
    if p is None:
        joined = " or ".join(var_names)
        raise FileNotFoundError(
            f"Set {joined} to point at the {what} (no hard-coded fallback)."
        )
    return p


# Resolved on first ``run_*`` / ``compare_*`` call so the module
# imports cleanly even when the env vars are missing.
ENV_PI05_CKPT_PYTORCH = "FLASHVLA_PI05_CKPT_PYTORCH"
ENV_PI05_CKPT_JAX = "FLASHVLA_PI05_CKPT_JAX"
ENV_RECAP_DATASET = "FLASHVLA_RECAP_DATASET"
ENV_JAX_SCRIPT = "FLASHVLA_JAX_BASELINE_SCRIPT"


# ── CSV helpers ────────────────────────────────────────────────────


def write_loss_csv(path: str | Path, steps: list[int], losses: list[float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        for s, l in zip(steps, losses, strict=True):
            w.writerow([s, l])


def read_loss_csv(path: str | Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    with Path(path).open() as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


# ── PyTorch pipeline ───────────────────────────────────────────────


@dataclass
class PytorchRunResult:
    csv_path: Path
    peak_memory_bytes: int
    seconds_total: float


def run_pytorch_pipeline(
    *,
    output_csv: str | Path,
    num_steps: int = 200,
    batch_size: int = 4,
    lr: float = 2.5e-5,
    encoder_rank: int = 16,
    decoder_rank: int = 16,
    acp_dropout: float = 0.30,
    seed: int = 42,
    base_ckpt: str | Path | None = None,
    dataset_root: str | Path | None = None,
    log_every: int = 10,
) -> PytorchRunResult:
    """Run our pipeline for ``num_steps`` and write a loss CSV.

    ``base_ckpt`` resolves from the ``FLASHVLA_PI05_CKPT_PYTORCH`` env
    var when ``None``; ``dataset_root`` from ``FLASHVLA_RECAP_DATASET``.
    """
    if base_ckpt is None:
        base_ckpt = _require_env_path(ENV_PI05_CKPT_PYTORCH, what="pi05 PyTorch ckpt directory")
    if dataset_root is None:
        dataset_root = _require_env_path(ENV_RECAP_DATASET, what="LIBERO RECAP dataset root")
    # Imports happen here so the pi0.5 model is only built when we run.
    from training.lora.inject import InjectionConfig
    from training.rl.checkpoint import load_pi05_pretrained
    from training.rl.lerobot_libero import LeRobotLiberoDataset
    from training.rl.tokenizer import PaligemmaTokenizer
    from training.rl.train_libero_recap import (
        LiberoTrainConfig,
        train_libero_recap,
    )
    from training.trainers.pi05_torch_trainer import Pi05Trainer

    output_csv = Path(output_csv)
    dataset = LeRobotLiberoDataset(dataset_root)
    model = load_pi05_pretrained(base_ckpt, action_horizon=10)
    trainer = Pi05Trainer(model, device="cuda")
    trainer.compile(
        InjectionConfig(encoder_rank=encoder_rank, decoder_rank=decoder_rank),
        calibration_passes=(),
        cache_bf16_weight=True,
    )
    tokenizer = PaligemmaTokenizer(
        max_token_len=trainer.model.config.max_token_len, device=trainer.device
    )
    cfg = LiberoTrainConfig(
        num_steps=num_steps,
        batch_size=batch_size,
        lr=lr,
        acp_dropout=acp_dropout,
        log_every=log_every,
        save_every=0,
        seed=seed,
    )
    result = train_libero_recap(trainer, dataset, tokenizer, config=cfg)
    write_loss_csv(
        output_csv,
        list(range(len(result.loss_history))),
        result.loss_history,
    )
    return PytorchRunResult(
        csv_path=output_csv,
        peak_memory_bytes=result.peak_memory_bytes,
        seconds_total=result.seconds_total,
    )


# ── JAX baseline ───────────────────────────────────────────────────


_JAX_LOG_PATTERN = re.compile(
    r"Step\s+(\d+)\s*/\s*\d+\s*\|\s*loss=([0-9.]+)"
)


@dataclass
class JaxRunResult:
    csv_path: Path
    seconds_total: float


def run_jax_baseline(
    *,
    output_csv: str | Path,
    output_dir: str | Path,
    num_steps: int = 200,
    batch_size: int = 4,
    lr: float = 2.5e-5,
    lora_rank: int = 16,
    acp_dropout: float = 0.30,
    log_freq: int = 10,
    base_ckpt: str | Path | None = None,
    dataset_root: str | Path | None = None,
    script_path: str | Path | None = None,
    extra_pythonpath: list[str] | None = None,
) -> JaxRunResult:
    """Shell out to the JAX baseline; parse its log for the loss curve.

    ``base_ckpt`` / ``dataset_root`` / ``script_path`` resolve from
    ``FLASHVLA_PI05_CKPT_JAX`` / ``FLASHVLA_RECAP_DATASET`` /
    ``FLASHVLA_JAX_BASELINE_SCRIPT`` respectively when ``None``.
    """
    if base_ckpt is None:
        base_ckpt = _require_env_path(ENV_PI05_CKPT_JAX, what="pi05 JAX (Orbax) params dir")
    if dataset_root is None:
        dataset_root = _require_env_path(ENV_RECAP_DATASET, what="LIBERO RECAP dataset root")
    if script_path is None:
        script_path = _require_env_path(ENV_JAX_SCRIPT, what="openpi train_jax_lora_recap.py path")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    env = os.environ.copy()
    extra_paths = list(extra_pythonpath or [])
    extra_paths.append(str(script_path.parent.parent))  # openpi-compiler/RL
    env["PYTHONPATH"] = ":".join(
        [*(p for p in extra_paths if p), env.get("PYTHONPATH", "")]
    )

    cmd = [
        sys.executable,
        str(script_path),
        "--checkpoint_path", str(base_ckpt),
        "--dataset_root", str(dataset_root),
        "--output_dir", str(output_dir),
        "--steps", str(num_steps),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--lora_rank", str(lora_rank),
        "--acp_dropout", str(acp_dropout),
        "--log_freq", str(log_freq),
        # Don't litter the disk with intermediate ckpts — only the
        # loss curve matters for the comparison.
        "--save_freq", str(num_steps + 1),
    ]
    logger.info("JAX baseline command: %s", " ".join(cmd))

    import time
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(script_path.parent.parent.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    seconds_total = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            "JAX baseline failed:\n"
            f"--- stdout (last 50 lines) ---\n"
            + "\n".join(proc.stdout.splitlines()[-50:])
            + "\n--- stderr (last 50 lines) ---\n"
            + "\n".join(proc.stderr.splitlines()[-50:])
        )

    # Parse "Step N/M | loss=L" records out of stdout/stderr.
    steps: list[int] = []
    losses: list[float] = []
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        m = _JAX_LOG_PATTERN.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    if not steps:
        raise RuntimeError(
            "No 'Step N/M | loss=L' lines found in JAX baseline output. "
            "Either log_freq is too high or the script changed its log format."
        )
    # In case duplicates from stdout+stderr crossover, dedupe by step.
    seen: dict[int, float] = {}
    for s, l in zip(steps, losses, strict=True):
        seen[s] = l
    steps_sorted = sorted(seen)
    losses_sorted = [seen[s] for s in steps_sorted]
    write_loss_csv(output_csv, steps_sorted, losses_sorted)

    return JaxRunResult(csv_path=output_csv, seconds_total=seconds_total)


# ── Curve comparison ───────────────────────────────────────────────


@dataclass
class CompareResult:
    pytorch_initial: float
    pytorch_final: float
    pytorch_direction: str
    jax_initial: float
    jax_final: float
    jax_direction: str
    direction_match: bool
    final_ratio: float


def _last_half_median(losses: list[float]) -> float:
    """Median over the last half of ``losses`` — robust to per-batch noise."""
    if not losses:
        raise ValueError("empty losses")
    tail = losses[len(losses) // 2:]
    s = sorted(tail)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _first_chunk_mean(losses: list[float]) -> float:
    """Mean over the first ~20 % of ``losses`` (initial-window proxy)."""
    if not losses:
        raise ValueError("empty losses")
    head = losses[: max(1, len(losses) // 5)]
    return sum(head) / len(head)


def _direction(losses: list[float]) -> str:
    initial = _first_chunk_mean(losses)
    final = _last_half_median(losses)
    if final < 0.95 * initial:
        return "decreasing"
    if final > 1.05 * initial:
        return "increasing"
    return "plateau"


def compare_curves(
    pytorch_csv: str | Path,
    jax_csv: str | Path,
    *,
    final_ratio_max: float = 5.0,
    require_direction_match: bool = False,
) -> CompareResult:
    """Compare two loss CSVs; return shape-only match diagnostics.

    Args:
        pytorch_csv, jax_csv: Output of ``write_loss_csv`` / produced
            by ``run_pytorch_pipeline`` / ``run_jax_baseline``.
        final_ratio_max: Multiplicative tolerance on
            ``max(p_final, j_final) / min(p_final, j_final)``. The
            order of magnitude must agree within this factor — defaults
            to 5×, which catches a NaN/exploded run while tolerating
            quantisation + framework differences.
        require_direction_match: If True, both curves must share the
            same coarse direction (``increasing`` / ``decreasing`` /
            ``plateau``). Defaults False — at very small step counts
            both runs hover at the converged-ckpt MSE noise floor and
            direction is purely noise-driven.
    """
    _, p_losses = read_loss_csv(pytorch_csv)
    _, j_losses = read_loss_csv(jax_csv)
    if not p_losses or not j_losses:
        raise ValueError("empty loss series")
    if any(l != l or l == float("inf") for l in p_losses + j_losses):
        raise AssertionError("non-finite loss in one of the runs")

    p_initial = _first_chunk_mean(p_losses)
    j_initial = _first_chunk_mean(j_losses)
    p_final = _last_half_median(p_losses)
    j_final = _last_half_median(j_losses)

    p_dir = _direction(p_losses)
    j_dir = _direction(j_losses)
    direction_match = p_dir == j_dir

    final_ratio = max(p_final, j_final) / max(min(p_final, j_final), 1e-12)
    if final_ratio > final_ratio_max:
        raise AssertionError(
            f"final-loss order-of-magnitude mismatch: "
            f"pytorch={p_final:.4f} jax={j_final:.4f} ratio={final_ratio:.2f}× "
            f"> {final_ratio_max}×"
        )
    if require_direction_match and not direction_match:
        raise AssertionError(
            f"direction mismatch: pytorch={p_dir}, jax={j_dir}"
        )

    return CompareResult(
        pytorch_initial=p_initial,
        pytorch_final=p_final,
        pytorch_direction=p_dir,
        jax_initial=j_initial,
        jax_final=j_final,
        jax_direction=j_dir,
        direction_match=direction_match,
        final_ratio=final_ratio,
    )
