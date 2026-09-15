# History timing alignment

The final comparison uses continuation calls (cycles 1 and 2) for LingBot-VA and DreamZero. The first three-call episode is warmup; the remaining ten episodes supply 20 continuation samples. Stateless policies retain 10 warmup + 30 measured calls.

This applies identically to all three frameworks. It resolves two API differences: Flash can perform episode reset/prompt setup before `predict`, while Omni handles reset inside `generate`; LeRobot VA omits the conditioning-frame actions from its first returned chunk. Continuation calls produce the complete action chunk and avoid treating these different reset boundaries as a framework speed difference.

All 33 original calls, their timings and first-cycle outputs remain in the receipts. The matrix retains both the original all-cycle p50 and the derived continuation p50, with exact selected call indices. No calls are selected by their measured speed or action values. These are early-episode samples, not saturated-KV or sustained-tail measurements.

This alignment is an explicit analysis amendment after API inspection, before completion of the remaining VA measurements. It changes neither quality thresholds nor inference. The separate prompt-boundary mismatch in DreamZero eager v5 is excluded through `excluded_receipts.json`; corrected runs retain one prompt throughout each episode.
