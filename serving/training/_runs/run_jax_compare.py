"""Run the W9g comparison at NUM_STEPS and persist both loss CSVs.

Output: training/_runs/<NUM>_step_archive/{pytorch_loss.csv, jax_loss.csv,
                                            summary.json}
"""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("PJRT_DEVICE", "CPU")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from training.rl.jax_baseline_compare import (
    compare_curves, run_jax_baseline, run_pytorch_pipeline,
)

NUM_STEPS = 500
ARCHIVE = Path(__file__).parent / f"{NUM_STEPS}_step_archive"
ARCHIVE.mkdir(parents=True, exist_ok=True)
common = dict(num_steps=NUM_STEPS, batch_size=4, lr=2.5e-5, acp_dropout=0.30)

t0 = time.perf_counter()
jax_res = run_jax_baseline(
    output_csv=ARCHIVE / "jax_loss.csv",
    output_dir=ARCHIVE / "jax_run",
    lora_rank=16, log_freq=10, **common,
)
print(f"[archive] JAX baseline done in {jax_res.seconds_total:.0f}s")

py_res = run_pytorch_pipeline(
    output_csv=ARCHIVE / "pytorch_loss.csv",
    encoder_rank=16, decoder_rank=16, seed=42, log_every=10, **common,
)
print(f"[archive] PyTorch run done in {py_res.seconds_total:.0f}s, peak={py_res.peak_memory_bytes/1e9:.2f}GB")

cmp = compare_curves(
    ARCHIVE / "pytorch_loss.csv", ARCHIVE / "jax_loss.csv",
    final_ratio_max=10.0, require_direction_match=False,
)

summary = {
    "num_steps": NUM_STEPS,
    "batch_size": common["batch_size"],
    "lr": common["lr"],
    "acp_dropout": common["acp_dropout"],
    "lora_rank": 16,
    "wall_seconds": {
        "pytorch": py_res.seconds_total,
        "jax": jax_res.seconds_total,
        "total": time.perf_counter() - t0,
    },
    "peak_memory_gb": py_res.peak_memory_bytes / 1e9,
    "pytorch": {
        "initial": cmp.pytorch_initial,
        "final_median_last_half": cmp.pytorch_final,
        "direction": cmp.pytorch_direction,
    },
    "jax": {
        "initial": cmp.jax_initial,
        "final_median_last_half": cmp.jax_final,
        "direction": cmp.jax_direction,
    },
    "final_ratio": cmp.final_ratio,
    "direction_match": cmp.direction_match,
}
(ARCHIVE / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
