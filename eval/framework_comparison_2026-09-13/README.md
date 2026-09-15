# LeRobot · vLLM-Omni · InstinctFlash — Jetson Thor

[Eight models × three frameworks: measured results](measured_candidates.md).

The table selects the lowest measured p50 per framework and model from the completed candidates, including compiled execution, NUMERIC, FP8 and fewer steps. These are **speed candidates**, not a complete quality-admitted ranking. Existing checkpoint-specific acceptance rules remain binding; [quality admission](quality_admission_inventory.json) records passed gates and missing evidence. Faster distilled Cosmos diagnostics are not promoted because their historical UniPC4 quality recovery failed.

Ten cells have no matching native implementation in the pinned framework registries. Pi0 does not substitute for pi05, and vendor inference does not substitute for LeRobot. See [support evidence](framework_support_audit.json).

Stateless policies use 10 warmups and 30 measured calls. History policies use 11 three-call episodes; the final table compares the 20 continuation calls after the first warmup episode. Reset and prompt setup are separate. All original calls remain archived. These are early-episode measurements, not saturated-cache or tail-latency guarantees. See [timing alignment](HISTORY_TIMING.md).

Different frameworks may select different schedules and numerical paths; ratios are not same-computation speedups. Omni's compiled Cosmos paths currently beat the measured Flash four-step paths. DreamZero's Omni step cache differs from Flash's fixed mask. Thor software versions and the isolated startup compatibility changes are disclosed in [setup notes](COMPATIBILITY.md).

## Evidence and reproduction

- [Matrix and exact sample indices](measured_candidates.json), [raw receipt inventory](receipt_inventory.json), [excluded attempts](excluded_receipts.json).
- [Protocol](PROTOCOL.md), [quality inventory](quality_admission_inventory.json), [status](STATUS.md).
- Framework pins: LeRobot `b6ec0060779550c0a157ae34feb89e0cf86012a8`; vLLM-Omni `f7d9deb45ab56e6a2ccc1690279bd9e6bdefbfe3`; Flash source `c07925f` with the separately recorded Thor kernel recovery.

Regenerate the table from the included small raw receipts:

```bash
python eval/framework_comparison_2026-09-13/build_matrix.py \
  --receipts eval/framework_comparison_2026-09-13/receipts \
  --output eval/framework_comparison_2026-09-13/measured_candidates.json
```

The benchmark drivers and launch scripts are retained in this directory. Their original absolute environment/checkpoint paths must be mapped to a new host; the raw receipts preserve the measured environment. Large action archives and logs remain under `/home/ubuntu/ifl_eval/framework_comparison_20260913` and the corresponding Thor run directory. No H100 result enters this table.

Implementation follow-up: [Omni optimization audit and Thor transfer ablations](../omni_transfer_2026-09-13/README.md).

Latest Edge follow-up: [matched observations and corrected seed binding, three
fresh runs per framework](../edge_matched_frameworks_2026-09-13/README.md).
The integrated Flash BF16 path averages 1099.57 ms versus Omni 1083.96 ms across
run p50s (Flash 1.44% slower). This new cohort has its own common protocol; it
does not overwrite the historical table or establish cross-framework task quality.
