# Nano memory and graph eligibility on Thor

Original Nano SDE1 CFG1 with cuDNN BF16 attention, six warmups and ten measured
requests. Diagnostic memory queries are included in this run; its timing is not
a replacement benchmark. All 16 finite 32×8 outputs match the saved uninstrumented
cuDNN baseline byte for byte. Sources and raw report hashes are bound in
`baseline-receipt.json`.

At request 6 before velocity evaluation, PyTorch allocated 28.60 GiB and reserved
29.50 GiB. CUDA reported only 3.95 GiB free of 122.80 GiB. Linux reported
85.38 GiB available and 81.13 GiB cached. All 576 layer calls bypassed the existing
16 GiB CUDA-free-memory guard; no captures, replays or numerical rejections.

This supports investigating clean checkpoint page-cache reclamation on unified
memory, rather than assuming the model uses almost all physical memory. It does
not prove capture safety or profitability. Releasing less than 1 GiB of inactive
PyTorch reservation alone cannot reach the 16 GiB threshold.

`benchmark_cache_advice.py --release-checkpoint-cache` is a separate experimental
arm: after native loading, issue POSIX_FADV_DONTNEED for the verified package's
safetensors files only. It does not change file contents or lower the graph guard.
Kernel page reclamation is advisory and may retain mapped pages. Actual memory,
existing graph byte checks, full actions and end-to-end latency must be evaluated.
This candidate is not installed in Runtime.

## Scoped cache-advice result

Advising 30.65 GiB of verified safetensors file pages increased CUDA free memory
from 5.53 to 37.31 GiB; Linux Cached fell from 81.14 to 50.49 GiB. At measured
request 6 CUDA free was 32.39 GiB with 32.51 GiB PyTorch reservation. Existing
graphs completed 36 captures, 72 internal checks and 504 replays, without a
rejection. All 16 full actions matched the memory diagnostic baseline byte for
byte after transfer.

This restores graph eligibility but does not demonstrate a speedup: instrumented
p50 was 788.41 ms. These diagnostic runs are not a clean production profitability
qualification. No memory guard, cache policy or Runtime default is changed.
The next runtime investigation must target the dominant generation computation;
neither this graph activation nor the approximately 1% SwiGLU pair establishes
realtime readiness.
