# Thor FP8 comparison, declared before measured trials

Primary question: separate engine execution gains from FP8 arithmetic gains,
without comparing different action horizons or omitting vision in only one arm.

Pi05 v044 revision 8e174154ef5f6c60a8da12ae99c303d8963138c1:

- Six arms: upstream BF16 and native capture at chunk 10; engine FP16 and
  engine FP8 at chunk 10; upstream BF16 and native capture at chunk 50 as a
  separately labelled horizon reference. All use ten denoise steps.
- Three fresh processes per arm, eight warmup and 128 timed predictions.
  Order rotates across repetitions. Keep failed attempts; no selecting fastest
  successful startup. Pilot calls are separate and never pooled.
- Two 224x224 real LIBERO camera frames, 48-token state-containing prompt,
  identical input and actual fp16-representable noise across comparable arms.
  Five existing recorded scenes: indices 0–2 for FP8 calibration, 3–4 for
  evaluation. These are tiny engineering samples, not a representative quality
  benchmark. No claim of unseen deployment distribution.
- Time CPU preprocessed frames and token IDs through CPU normalized action
  output. Every call recomputes vision and language prefix. Excludes original
  camera preprocessing, tokenization, action unnormalization, transport and
  robot control. Engine's static prompt embeddings may be prepared at startup;
  this is disclosed rather than presented as complete Runtime API parity.
- Save each sample, output, repeated fixed-input controls, load/calibration and
  warmup times, numeric flags, source/input hashes and memory observations.
  PyTorch allocator peaks exclude external engine allocations. Report them as
  such, never as total engine memory or a memory-reduction factor.
- Native capture rejection remains a measured fallback with its actual state.
  FP16 engine failure does not authorize replacing it with FP8.
- No speed threshold or task-quality certificate is declared for these probes.
  No modification to default precision, engine or calibration thresholds.

Other families: audit VLA4B/V2/GR00T existing raw protocols and frontend
contracts. Remeasure compatible staged paths where executable. Unsupported or
semantically mismatched comparisons stay explicit. Review historical closed-loop
results separately; none certify new source versions or this new calibration.

Runtime integration audit: verify camera mapping, state/prompt handling,
normalization, chunk geometry and refresh policy before labelling a comparison
as interchangeable robot-facing Runtime execution.

VLA4 additional measured arm set, declared before any VLA4 execution:
upstream, native capture and staged FP8 engine, three fresh processes each,
eight warmup and 128 measured calls. All compute 50x75 normalized actions at ten
steps; controller slicing to 25x14 is outside every arm. Existing synthetic
reference inputs are reduced to input-only records and rounded to common
BF16-representable values. Sample 0 calibrates; samples 1–11 cycle during
measurement. All paths process vision and prefix every call. This supplies
latency and numerical diagnostics only, not real-scene or closed-loop evidence.

V2 additional measured arm set, declared before any V2 execution: upstream,
native NUMERIC capture (vision, prefill and denoise with unchanged admission
guard), and full engine (FP16 vision/prefill, FP8 expert with FP16-source router).
Three fresh processes each, eight warmup and 128 measured calls. Use recorded
observations 0 and 8 from the historical M2 set, which share the same prompt.
All compute normalized 50x55 chunks at ten steps. Common BF16-representable
inputs and actual noise; capture failure is retained as fallback, never retried
until accepted. Timing excludes raw-camera processing and controller decoding.
Engine uses explicitly inventoried historical repack/calibration/vision-table
artifacts. These two frames and that calibration do not establish held-out task
quality. Frozen helper copies only change source-location bindings.

Before any LingBot measured process: source inspection found that both native
constructors set float32 matmul precision to `high`. The probes reassert TF32 off
after loading and before warmup, and record the observed final switches. Both
LingBot arms use cuDNN deterministic=true (the native deployment setting),
benchmark=false. This is a controlled comparison setting, not an as-shipped
vendor-default measurement. The pi05 protocol retains its original switches.

Also declared before any LingBot GPU trial: four additional spot checks, one
fresh stock and one fresh capture process for each LingBot family, retain the
native constructor's matmul TF32 choice. They use the same eight warmup/128
measured calls and common inputs; other controlled settings stay unchanged.
These are named `stock_vendor`/`capture_vendor`, reported as n=1, not pooled
with the three-start TF32-off arms. They are constructor-TF32 controls, not full
vendor-default or complete robot-API benchmarks. No further automatic trials.
