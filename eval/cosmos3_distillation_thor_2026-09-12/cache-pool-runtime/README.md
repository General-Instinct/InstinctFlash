# Shared Cosmos graph-pool lifecycle regression

The shared `ConditioningCache.wrap_layer` now allocates a graph pool per slot. Eviction or transaction cleanup can release the last graph while output storage remains live; reusing a handle from that retired pool triggered the allocator assertion.

[Before](before.json), the exact source at commit c9ece6b completed the first cycle, then recorded four allocator rejections and fell back. [After](after.json), the source hash bound in the receipt completed all three cycles with6 captures,12 checks,12 replays and zero rejections. This probe exercises the real shared wrapper and CUDA LayerGraph with two small modules, retained output storage and changed inputs. It is not a model latency benchmark. The separate full Nano experiment preserves its action-level evidence in `../nano-persistent-text-v2/`.

17 CPU admission/lifecycle/regression tests also pass. The first remote probe launcher omitted the Flash root from PYTHONPATH and failed during import before GPU capture; its log is preserved remotely. The corrected launcher ran both original and patched source in separate processes under the Thor lock. Neither probe changes runtime precision, schedule, or cache admission defaults.
