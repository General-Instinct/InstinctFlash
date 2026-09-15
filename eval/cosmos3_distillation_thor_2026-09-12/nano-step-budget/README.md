# Original Nano few-step cost budget on Thor

This measures released original Nano weights with complete SDE1 `[1,0]` and
SDE2 `[1,.5,0]`, each CFG1, using native and cuDNN BF16 attention. These are not
DMD-trained Nano students or quality-qualified operating points.

Fresh inventories cover all43 native files on both H100 and Thor. The file sets
and every model/config/tokenizer byte agree; only README differs. A private
Thor anchor substitutes the H100 README and rechecks all43 identities. Source
weights, training exports and mechanical Nano qualification weights are untouched.
The separate external Wan VAE is observed and hash-checked during actual loading.

`prepare.py SOURCE INVENTORY FRESH_PACKAGE --steps 1|2` creates package-local
hardlinks with a private native config and explicit complete sampler declaration.
It preserves Nano's `format_prompt_as_json=false`, trained action heads, DROID
history1/future32 and zero padding. The original Nano family model ID permits
existing native Nano action-only residency; each artifact is still explicitly
`servable=false` with an untrained-cost-only manifest and original revision.

The benchmark reuses the established six warmup/ten measured request fixture,
real clocks, full32x8 actions and matched native/cuDNN comparator. All existing
source/precision guards remain in force. Python compilation passed; actual
model execution and comparison receipts are required before reporting speed.

`run_all.py` stages the original anchor, prepares both packages, and serializes
the four arms under the shared Thor GPU lock. It expects the H100 README staged
as `source_README.md` beside the script. Source on Thor: `nano-step-budget-v2`;
status: `nano-step-budget-live-v2.json`, both under the existing study root.
H1004–7 remain Compress-owned. No quality or controller realtime claim follows
from this cost screen.

V1 loaded the native Nano model but stopped before any measured request because
its prompt check incorrectly accessed `service.cfg.format_prompt_as_json`.
Failed sources/report are preserved in `failed-v1`. V2 checks the actual native
`service._transform.prompt_json_formatter` and all three metadata augmentors.
Five CPU tests using the real ActionTransformPipeline pass, including rejection
of JSON prompts and missing metadata augmentors. No model computation changed.

The V2 continuation reuses the completed SDE1 package only after checking its
exact original manifest hash; each worker then revalidates every package file.
It creates the SDE2 package fresh. The original anchor is reused, and all V2
measurement outputs use fresh directories. The V1 staging script is retained
for reproducing the original anchor from the two inventories and H100 README.

## Completed SDE1 cost pair

| Attention | p50 | p95 |
|---|---:|---:|
| Native BF16 | 1120.70 ms | 1170.09 ms |
| cuDNN BF16 | 790.04 ms | 791.86 ms |

The matched attention speedup is1.4185×. All16 finite32x8 action outputs,
actual complete `[1000]` clocks, zero-padding branch checks, source/package
identities and loaded external VAE checks passed on Thor and after transfer.
Native/cuDNN action maxabs is0.03125, meanabs0.00230498. The native Nano
prompt pipeline and existing action-only residency were retained.

This is six warmups and ten measurements, not a tail-latency reliability study.
At a hypothetical500ms deadline this one-step cost still exceeds the budget;
step-count distillation alone does not establish such a target. Actual controller
contract, trained Nano outputs and quality admission remain pending. The SDE2 pair is also complete; see below.

## Completed SDE2 cost pair

| Attention | p50 | p95 |
|---|---:|---:|
| Native BF16 | 2248.63 ms | 2253.44 ms |
| cuDNN BF16 | 1545.66 ms | 1548.06 ms |

The matched attention speedup is1.4548×. All16 finite32x8 outputs and complete
`[1000,500]` two-branch clocks passed the same full comparator on Thor and
locally. Native/cuDNN action maxabs is0.1306152, meanabs0.01186829. Neither
this action delta nor the latency establishes policy quality. Both original
Nano packages have45 manifest entries; original43-file inventory and external
VAE identities remain separately recorded.

All four cost arms are now terminal success and the shared Thor lock is released.
For this unchanged architecture the cuDNN two-step cost is1.956× the one-step
cost. The next latency investigation must profile per-step work and validate
any reused text-conditioning computation on changed images, states and prompts;
training alone does not remove that repeated runtime cost.
