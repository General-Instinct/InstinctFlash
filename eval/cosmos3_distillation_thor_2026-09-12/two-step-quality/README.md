# Original-weight two-step quality diagnostic

Separate follow-up to the frozen one-step/control studies. This uses original
Edge weights with complete SDE `[1, 0.5, 0]`, CFG1, BF16 native attention and
zero action padding. It is not a trained two-step student.

The frozen plan reuses the exact 16 historical observations and eight explicit
request seeds per observation. Earlier student and control outcomes were already
seen; this is development diagnosis, not fresh confirmation. All 128 requests,
full endpoints, noise, conditioning, targets and external Wan VAE identity are
retained. Capture timings are not latency benchmarks. The separate two-step
latency screen is in `../two-step-budget`.

The capture helper was copied from the frozen original controls and extended to
validate complete one-, two- and four-step grids. Twelve CPU tests passed: actual
native endpoint parity for CFG1/CFG4 at all three lengths and UniPC4, plus five
malformed-grid rejections. Earlier control sources and receipts are unchanged.

On the frozen Thor environment, run:

```
python run_control.py original_sde2_cfg1 FRESH_OUTPUT
python validate_control.py FRESH_OUTPUT --source SOURCE_DIRECTORY --output VALIDATION_JSON
```

Check `validate_control.py --help` for argument syntax. GPU capture holds the
shared Thor lock and rejects contention. No H100 training resources are used.

The first capture was launched from remote `two-step-quality-v1` to
`control-original-sde2-cfg1-v1`, with its sibling `.log`. Completion, full input
pairing and scoring must be verified before interpreting any quality change.
The native result does not qualify the action-changing cuDNN variant.

`export_control.py` validates the complete bank and compares every request's
reference, initial noise, mask, targets and normalizer to both CFG1 training
seeds. No padding-coordinate exclusion applies to these matched SDE inputs.
`score.py COMPACT_DIRECTORY FRESH_REPORT_JSON` checks compact archive and source
identities, then reuses the frozen scoring and separately implemented audit
formulas. It retains all 96 metric vectors and 15 comparison groups: three
original controls, eight deployed one-step variants, and four within-episode
seed means. Confidence intervals are exploratory and do not establish a
noninferiority margin or closed-loop task success.

## Completed native result

All 128 captures and full input pairing to both CFG1 seeds passed. Independent
formulas reproduced all 96 metric vectors and 1,440 bootstrap comparisons
(15 groups). See `score_receipt.json` for all primary comparisons and intervals;
the full report and compact arrays remain at the hash-bound local paths.

Primary energy-score changes below are candidate minus baseline, divided by
baseline; lower is better. These are not success-rate percentages.

| Baseline | Joint h1 | Gripper h1 | Joint h32 | Gripper h32 |
|---|---:|---:|---:|---:|
| Original UniPC4 | +13.08% | +13.82% | +3.03% | -7.33% |
| Original SDE1 CFG1 | -0.76% | -20.63% | +4.16% | -22.84% |
| Trained SDE1 CFG1 native, two-seed mean | -2.49% | +10.89% | +3.66% | -11.28% |

Against UniPC4, the short joint and short gripper uncorrected 95% intervals
are strictly positive. Against original SDE1 CFG1, both gripper intervals are
strictly negative. All four intervals against the trained CFG1 native mean
cross zero; this does not establish equivalence. The result motivates testing
actual two-step training, not promoting unchanged weights or selecting a winner.

This bank used **native BF16 attention**. Its corresponding latency screen was
719.32 ms p50; the separately measured 442.98 ms cuDNN variant still needs its own
full quality capture. All native capture/postprocess/scoring processes finished
successfully; the shared Thor GPU lock was released.
