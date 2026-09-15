# Edge V17 two-step student qualification packages

This separate preparer admits only the two formal Edge SDE2 `[1,.5,0]` CFG1
final-online64 seeds. It binds the producer completion, independent formal-training
audit, exact native source/cold gates, student snapshot and full merged inventory.
All native files are copied and hashed before/after preparation; producer outputs
are unchanged. The existing V16 one-step preparer remains unchanged.

Use the same CPU environment as `../realtime-deployment`, including that directory
on PYTHONPATH for `realtime_adapter`. Run:

```sh
python prepare.py EXPORT_ROUNDTRIP FRESH_DESTINATION --model-id MODEL_ID \
  --completion /home/ubuntu/InstinctCompress/results/cosmos3_droid/twostep_v17/training_execution_completion.json
```

The declaration keeps 1000 training timesteps and actual callbacks at 1000/500,
zero action padding, DROID future32, and `servable=false`. Native source/cold
parity does not qualify Thor numerics or task quality. Each seed requires both
native and actual deployed numerical-variant measurements.

Sixteen CPU regression cases cover both seed declarations, their actual clock
conversion, and rejection of incompatible steps, CFG, updates, roles and gates.
An initial broad string edit accidentally changed the declaration's training
clock to 2000. Review caught it before any transfer or inference. Preserve V1
source and exclusions in `failed-v1`; only corrected destinations with `-v3`
suffix are eligible for subsequent qualification.

Both corrected packages completed CPU preparation: all 35 original native files
were verified before and after copying. Exact declarations and manifest hashes
are in `receipts/`. `run_all.py` pins those two manifests and serializes each
seed's native/cuDNN pair under the shared Thor lock. It records child PIDs,
terminal exit codes and invokes the full paired comparator. Results are written
to a fresh directory; run status alone does not establish quality or realtime.

V2 then failed the actual Runtime prefix check before model inference. V3 keeps
prefix1/action2 and adds actual package parsing plus adapter admission to preparation
and regression tests. V1/V2 sources and V2 receipts remain in `failed-v1`/`failed-v2`.

## Completed Thor V3 trained diagnostic

| Final64 seed | Native p50 / p95 | cuDNN p50 / p95 | Attention speedup | Max action delta |
|---|---:|---:|---:|---:|
| 12031 | 721.79 / 724.84 ms | 442.23 / 442.74 ms | 1.6322× | 0.0161133 |
| 12032 | 721.18 / 722.70 ms | 444.27 / 444.66 ms | 1.6233× | 0.0166016 |

All four workers completed successfully. Each pair passed the full comparator
both on Thor and independently after transfer: 16 finite 32×8 actions, complete
1000/500 callbacks, CFG1, zero padding, actual external Wan load, package/source
identities and explicit precision policies. Six warmups and ten measurements
per arm; no tail-reliability or task-quality certificate.

These are trained-student diagnostic pairs, not admission. Producer historical
H100 quality did not recover UniPC4 short-horizon action quality; those scores
do not qualify either Thor numerical variant. The full native/cuDNN arrays and
reports are in `thor-receipts/`. Original source exports and failed V1/V2 evidence
remain intact.
