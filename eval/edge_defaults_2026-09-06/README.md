# Native Thor execution: LingBot-VLA-V2 and LingBot-VA

Completed 2026-09-06. Experiment root: `/home/ubuntu/ifl_eval/edge_defaults_20260906`;
Thor root: `/home/guanming/ifl_eval/edge_defaults_20260906`.
FP8 engine closed-loop evaluation is deferred. These experiments preserve the
native step schedule, guidance and precision settings.

The previous planner rejected capture for every Thor model based on a pi05
measurement. Device policy now separates family/operating-point evidence. V2
capture is NUMERIC because its startup check accepts a nonzero envelope; the
BITEXACT ceiling declines it, including on H100. Explicit NUMERIC permission
allows native capture on the measured Thor operating point. A direct installer
also refuses a capture plan mislabeled BITEXACT. A BITEXACT planning ceiling
constrains transformations; it does not certify a nondeterministic original model.
Fresh fine-tunes inherit an
adapter, not a quality certificate.

V2 experiments use the pinned native checkpoint, three real inputs and actual
recorded initial noise from the previous campaign. Each arm has a fresh process,
eight warmup calls and 64 measured calls. The order is capture, original, previous
Runtime, previous Runtime, original, capture. The experimental device-gate bypass
is isolated in the frozen execution snapshot; it does not modify the native
model or the production device profile. Full action arrays accompany timings. A second frozen source snapshot validates
the final production Runtime with explicit NUMERIC and BITEXACT ceilings, without
the experimental planner bypass.
The capture self-check's H100 envelope is retained as an implementation check,
not treated as a Thor quality certificate.

LingBot-VA uses its native 25-video/50-action schedule. The baseline contains
FSDP elision, allocator/debug-copy elision, conditioning prefill and ring KV.
Convolutions retain their native layout/backend: P007 is NUMERIC and excluded. The first candidate adds action terminal
forward elision. A third arm adds selective CUDA graphs for the 50 action-denoise
forwards only, using the existing executor that keeps ring bookkeeping on the host;
video and KV-commit forwards stay eager. This is an experimental candidate, not
a default change. Its two-graph cache bounds memory and drops at episode reset. Both replay the same real frames cyclically for 48 cycles and
reset for a second episode. This crosses ring saturation and tests reset
repeatability. It is a systems stress replay, not a new RoboTwin success rate.
The first full episode and all instrumented cycles are excluded from latency
summaries. Instrumented cycles report transformer CUDA-event spans and host
submission times; event spans include idle time and are not kernel-busy totals.

`probe_va.py` reproduces the VA experiment in the existing native model environment.
`analyze.py --root EXPERIMENT_ROOT --output summary.json` regenerates statistics
and byte comparisons from raw evidence. Committed raw timings and actions are in
[`raw/`](raw/); [`summary.json`](summary.json) includes all sample statistics,
comparisons and consumed-file hashes. [`evidence-index.json`](evidence-index.json)
records raw-file hashes and external source archives/logs; [`validation.json`](validation.json)
separates passing checks from known unrelated test limitations.

```bash
python eval/edge_defaults_2026-09-06/analyze.py \
  --root eval/edge_defaults_2026-09-06/raw --output /tmp/edge-summary.json
```

The V2 production snapshot predates the legacy-plugin fallback and explicit
precision dispatch correction; those later changes have CPU and installed-wheel
validation. The fallback was also compared against the frozen planner for the
measured V2 declaration on Thor/H100 and produced identical pass results.
No timing is attributed to a source snapshot that was not executed.

## LingBot-VLA-V2 results

All values below are measured milliseconds per 50-action chunk, native NFE=10.
Each row contains 64 timed calls after eight warmups, on three recorded inputs.

| Path | Run | p50 | p99 |
| --- | --- | ---: | ---: |
| Original | 1 | 748.26 | 875.70 |
| Original | 2 | 701.47 | 859.71 |
| Previous native Runtime | 1 | 713.17 | 739.45 |
| Previous native Runtime | 2 | 698.67 | 734.54 |
| Experimental native capture | 1 | 404.62 | 408.34 |
| Experimental native capture | 2 | 401.10 | 407.03 |
| Final Runtime, NUMERIC | validation | 398.91 | 407.32 |
| Final Runtime, BITEXACT ceiling | validation | 716.64 | 749.36 |

The experimental paired p50 ratios are 1.85x and 1.75x. Final Runtime validation
uses the production planner and adapter, without the experimental bypass. Its
NUMERIC path captures; the BITEXACT ceiling declines capture. Both keep native
matmul TF32=true, cuDNN TF32=true, cuDNN benchmark=false. Explicit in-process
placement prevented engine execution in the measured snapshot. The subsequent
precision-policy correction also removes that unused engine from native plans:
`precision="native"` is now the default even with a NUMERIC ceiling. This dispatch
change has separate CPU regression coverage; the timings above belong to the
frozen GPU-tested snapshot.

Original repeats match in only 22/64 corresponding action chunks (maximum
absolute difference 0.0114672). First original/capture comparison also matches
22/64 (maximum 0.0102424). Final strict/numeric validation matches 20/64
(maximum 0.0112289). These differences are native action values, not accuracy-loss
percentages. The original's variability does not establish that capture adds no
error. Capture remains NUMERIC and no new closed-loop certificate is claimed.
The startup denoise self-check happens to be exact on its six staged inputs;
that does not establish full-policy exactness on the recorded observations.

To select the measured native capture path explicitly:

```python
runtime = Runtime.from_pretrained(
    "robbyant/lingbot-vla-v2-6b-robotwin",
    revision="0451855729ec904f970600e0aec8b84661423afe",
    precision="native",
    tier_ceiling="numeric",
)
```

The ordinary BITEXACT planning ceiling keeps the native eager executor. This is
an intentional correction to the previous H100 default: a tolerance-gated
capture must not silently spend a BITEXACT ceiling. The historical H100
reproduction script now requests its NUMERIC capture arm explicitly.

## Precision boundary

The public Runtime and serving CLI now default to `precision="native"`. A NUMERIC
ceiling can permit native capture but never authorizes FP8. Explicit FP8 defaults
to a NUMERIC ceiling, rejects BITEXACT, and refuses unsupported deployments rather
than falling back to a different precision. Preflight and execution share the
model/device/plan gate. The old placement path that bypassed a ceiling-demoted
engine pass has been removed. Step-count overrides remain separate.
See [precision and execution controls](../../INSTALL.rst#load-and-predict) for support limits.
This batch does not add FP8 closed-loop evaluation or new FP8 model integrations.

## LingBot-VA results

The experiment did not pin clocks or change board power settings. Raw tegrastats
are retained with the external artifacts; this is one serialized pair per candidate.

Native 25V/50A, cycle latency includes inference and the real KV-commit message.
Run 0 is discarded; instrumented cycles are excluded. Early means one-indexed
cycles 2–8 (6 uninstrumented samples), saturated means cycles 37–48 (9 samples).
The small samples describe this replay; they do not establish a production p99.

| Path | Early p50, ms | Saturated p50, ms | Actions matching base |
| --- | ---: | ---: | ---: |
| Native optimized baseline | 5885.25 | 8500.33 | reference |
| + action terminal forward elision (P010) | 5905.80 | 8456.55 | 96/96 |
| + P010 and selective action CUDA graphs | 6029.20 | 8694.24 | 96/96 |

P010 is exact across both 48-cycle episodes and saturation: max absolute action
delta 0, both same-arm repeats 48/48 exact. Its controller elided/reserved 96
terminal writes, with no materialization or self-disable. At this 50-action-step
schedule, removing one forward yields no convincing overall speedup: early is
0.35% slower and saturated is 0.52% faster in this one pair. Existing P010 behavior
is retained; this result is not a new claim of a large Thor speedup.

In the instrumented baseline early cycle, 25 video denoise forwards span about
2662 ms and 50 action denoise forwards about 2567 ms; the elidable action terminal
forward spans only about 51 ms. At saturation, those spans grow to about 3595,
4095 and 81 ms. They locate the expensive regions but include stream idle time,
so they are not GPU kernel-busy totals.

Selective action capture also matches all 96 baseline outputs byte-for-byte,
including saturation, with 48/48 exact reset repeats. It performed 96 captures
and 4800 replays, with no graph failure; its P010 controller never materialized
or disabled. Nevertheless it is 2.45% slower early and 2.28% slower saturated
than the baseline (2.09% and 2.81% slower than P010 alone). Growing/rotating ring
signatures require a new capture every cycle, and removing submission overhead
does not repay that cost in this measured native schedule. This candidate remains
experimental and **is not installed as a new default**. These results are scoped
to the replay, checkpoint, schedule and board; they do not certify other policies
or establish a simulator success rate.

P0 is delivered as model-specific native capture eligibility plus an explicit
precision boundary. P2 now has a completed saturation profile and two measured
candidates, including a negative result. Further VA speed work should target the
large video/action denoise regions and be gated by paired output and latency
measurements. FP8 closed-loop evaluation (P1) remains deferred.
