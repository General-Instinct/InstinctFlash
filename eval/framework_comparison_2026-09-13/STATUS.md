# Thor comparison status

The [generated eight-model, three-framework table](measured_candidates.md) is the authoritative speed summary. Raw receipts and selected sample indices are in [the matrix JSON](measured_candidates.json). This is a measured-candidate comparison; checkpoint-specific quality admission is recorded [separately](quality_admission_inventory.json).

InstinctFlash covers all eight models. The pinned LeRobot registry supports pi05, GR00T N1.7 and LingBot-VA; the pinned vLLM-Omni registry supports Cosmos Edge, Cosmos Nano and DreamZero actions. The other ten cells are unsupported, not unmeasured substitutes. All 14 runnable cells completed successfully, including LeRobot VA default and 2V/4A. There are no pending speed-coverage cells.

Compiled Omni Edge and Nano beat the currently measured InstinctFlash four-step candidates on Thor. DreamZero's Omni step cache and Flash fixed schedule differ, so their latency comparison is not a same-computation speedup. All losing candidates and failed attempts remain archived.

Faster distilled Cosmos diagnostics remain unqualified: historical UniPC4 quality recovery did not pass. VLA-4B's current FP8 route passed its existing aggregate T2 numerical rule; GR00T native retains its exact startup gate. Neither establishes a new closed-loop quality certificate for other recipes.
