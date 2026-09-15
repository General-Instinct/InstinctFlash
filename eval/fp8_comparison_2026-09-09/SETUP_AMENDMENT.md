# V2 working-directory correction

The first V2 stock and capture starts failed before inference because the native
server reads `configs/robot_configs/robotwin.yaml` relative to the process working
directory. The first engine start succeeded. When the same invalid native setup
was reached on the next capture start, that start and the remaining orchestration
were stopped. `interrupted-setup.json`, the original `v2-status.json`, and all
original logs/results remain retained.

Before replacement trials: run the complete nine-start V2 arm set in
`cwd-repair/results/`, with the native source directory as working directory.
The probe, inputs, precision flags, thresholds, calibration, warmup and iteration
counts do not change. The successful pre-repair engine start is also excluded
from the primary table so the table uses the complete replacement arm set.
This is a setup repair, not selection based on latency or capture admission.

The four constructor-TF32 spot checks had not started. They will also run under
the new orchestrator, with the same declared profiles; V2 gets the corrected
working directory. Any subsequent capture rejection remains fallback. No retry
on measured numerical outcomes is permitted.

## Receipt-field correction

The replacement V2 `capture.0` completed its predictions and saved its action
NPZ, then failed while writing metadata: the probe read `den._graph` whereas
V2's `StaticVelocity` exposes `den.graph`. This is a receipt bug, not a capture
admission rejection. Preserve that log, NPZ and the old probe source. Fix only
the attribute used in metadata, pausing orchestration until the running engine
start finishes so its source hash remains correct. Subsequent starts use the
correct receipt field. After the remaining campaign finishes, run one replacement
`capture.0` in `receipt-repair/results/`; retain its result whether capture is
active or fallback. No numerical threshold or model execution code is changed.
