# Current four-step deployment controls

Compress acknowledged the cross-agent handoff and selected the first declared
seed (12031) of V14 fixed-last/ratio5 and V15 random-exit for matched deployment
qualification. This is deterministic runtime-control selection, not a new
quality ranking. Both original exports stay unchanged.

`prepare.py` verifies the successful export/cold-reload report and full original
inventory, copies the native package outside the frozen export, adds the exact
four-step SDE/CFG4/zero-padding Flash declaration and projection sidecar, and
hashes the complete new package. It then rechecks all original source files.
Both prepared packages are `servable=false` pending fresh Thor qualification.
The [preparation receipts](receipts/) show 35 original files preserved byte for
byte in each package. Native weights/configuration/sampler are not rewritten.

Local packages: `/home/ubuntu/ifl_cosmos_deployment/v14-seed12031` and
`/home/ubuntu/ifl_cosmos_deployment/v15-seed12031`. They are not published models.
Use `benchmark_candidate.py --allow-unqualified` only for this explicitly
recorded offline qualification, or `run_pair.py` to serialize native/cuDNN arms.
In addition to the existing full-action and sampler checks, the candidate
benchmark requires the actual eight padding-projected CFG branches per request
and restored projection hooks. No task-quality certificate is inferred.

## Completed Thor screen

| Control (seed 12031) | Native p50 ms | cuDNN p50 ms | cuDNN p95 ms | Max action delta |
|:--|--:|--:|--:|--:|
| V14 | 2554.6 | 1470.5 | 1474.4 | 0.346191 |
| V15 | 2556.7 | 1469.4 | 1470.3 | 0.377258 |

Each arm completed 16 requests, 64 sampler callbacks and 128 padding-projected
CFG branches. The first request checked the complete native clock sequence
1000/750/500/250 (each conditional and unconditional), allowing the documented
one FP32 ULP. Complete finite actions, manifest/source/fixture agreement and
actual cuDNN execution passed the paired screen. Original sampler and padding
contracts are retained. All four GPU processes exited successfully.

These receipts establish runtime execution and numerical differences. They do
not certify task quality or reclassify either training method as a quality win.
Packages remain unqualified `servable=false`; no publication occurred.
