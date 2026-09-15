# Reproducing this study

There are two distinct operations: recomputing the report from saved outputs,
and running new Thor measurements. Neither operation runs a simulator.

## Recompute the report

`artifact-index.json` identifies the local evidence archive and its SHA-256.
Verify that digest before unpacking into an empty directory. The archive contains
`archive-files.json`, which gives a digest for each retained source/output file.
With NumPy installed, run:

```sh
python eval/fp8_comparison_2026-09-09/analyze.py \
  --root /path/to/restored/raw \
  --output eval/fp8_comparison_2026-09-09/pi05-summary.json
python eval/fp8_comparison_2026-09-09/render_report.py \
  --root /path/to/restored/raw \
  --reports eval/fp8_comparison_2026-09-09
```

The analyzer validates action-file hashes and matched inputs, noise and output
geometry. It retains each process's latency distribution and fixed-input
repeatability controls. The report requires all four campaign completion files;
a completed campaign may contain failed starts, which remain in the attempt logs.

## Run new GPU measurements

The archived probes target the recorded Thor installation, including compiled
engine binaries. This is an installation-specific study, not a portable
checkpoint benchmark installer. See `environments.json`, `device.txt`,
`power-mode.txt` and `external-artifacts.json` in the archive. External model
weights, recorded observations and repack/calibration artifacts are referenced by
hash and are not all bundled. They must be available to reproduce the measurement.

Create a new campaign directory with an empty `results/`. Copy the archived
`source/`, `native-vla4/`, `native-vla2/` and `vla4-inputs.pt` there. Preserve the
original campaign. Bind the paths in the runners and the two V2 helper modules
to the new directory and restored external artifacts. The archived
`reference-runtime/instinctflash` provides the measured capture implementation;
use its parent in `PYTHONPATH` instead of substituting the latest workspace.
The V2 native configuration and VLA4 Qwen configuration paths also need to point
to their recorded versions. Retain a diff of any path-only changes.

On an otherwise idle Thor in the recorded power mode, execute sequentially:

```sh
~/frt_env/bin/python /new/campaign/source/run_pi05.py /new/campaign
~/frt_env/bin/python /new/campaign/source/run_vla4.py /new/campaign
~/frt_env/bin/python /new/campaign/source/run_remaining.py /new/campaign
```

`run_remaining.py` applies the native working-directory correction documented in
[SETUP_AMENDMENT.md](SETUP_AMENDMENT.md), and writes the nine V2 starts and four
constructor-TF32 controls into a new `cwd-repair/` subdirectory. Do not replay the
known-invalid original V2 setup just to generate matching failures.

The runners select the model-specific Python environments for LingBot. They
retain failed starts and refuse to overwrite trial logs. Keep the fixed inputs,
precision flags, calibration split, warmup count and rotated trial order from
[PROTOCOL.md](PROTOCOL.md). A different device, binary build, calibration or
execution profile is a new comparison and needs separately labelled results.

Historical closed-loop evidence is separate from these probes. Its original
source paths and report hashes are recorded in `historical-evidence.json`; the
pi05 paired-count recount is in `pi05-historical-recount.json`.
