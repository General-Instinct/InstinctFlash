# Measured source versions

The final comparison uses `source-v5` for **both arms** of VLA-4B, VLA-V2 and GR00T, and `source-v1` for both arms of the other six configurations. No pair mixes snapshots. Earlier VLA/GR00T measurements are archived with `.before-source-v5` names and excluded from the published comparison. Snapshots v2–v4 were staged during development; their queued tests were replaced before acquiring the GPU lock.

V5 contains the final vision/text graph fixes, graph lifetime cleanup and additional post-timing graph statistics. Its changes relevant to the remaining six configurations are a packing-module docstring, an execution-description string, and SM110 guards around the already-measured Cosmos/DreamZero extensions. Those guards keep the older H100 Q/K/V recipes; the Thor path retains the same projections and arithmetic. VA and pi05 numerical implementations are unchanged.

The final verification checks all five immutable source manifests and the four compiled libraries in each snapshot. Each timing receipt is bound to its actual source version. The current runtime and benchmark were also checked against all 425 files in `source-v5` after publication. The [regression receipt](regression.json) records 88 passing tests of this final code.
