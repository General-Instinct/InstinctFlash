# VA native/FP8 saturation stress

LIBERO-long completed 48 cycles × two episodes in each public Runtime precision
arm on Thor, with the original 20V/50A schedule. All 192 action chunks are finite
and have native shape 7×4×4. Each precision repeats all 48 corresponding actions
byte for byte after reset. Boot, source, schedule and normalized cache occupancy
checks pass. RoboTwin's corresponding 25V/50A pair also completed: all 192 chunks have
native shape 16×2×16, are finite, and all 48 reset pairs match byte for byte
within each precision. Normalized cache occupancy matches across precisions.
All 368 source/config/input hashes and the boot identity were rechecked unchanged
at campaign completion.

| RoboTwin saturation interval (cycles 36–47, zero-based) | Native median | FP8 median | Native / FP8 |
| --- | ---: | ---: | ---: |
| First exposure | 8496.87 ms | 8150.24 ms | 1.04× |
| Repeat with graph shapes established | 8491.30 ms | 5614.70 ms | 1.51× |

The first FP8 cycle costs 8270.98 ms versus 5866.24 ms native. These results
have the same synthetic-history and single-episode-per-phase limitations as
LIBERO below. They do not establish task quality or real-time reliability.
[RoboTwin verified comparison](va-saturated-robotwin-comparison.json) /
[native receipt](va-saturated-stress-robotwin-native.json) /
[FP8 receipt](va-saturated-stress-robotwin-fp8.json).

| LIBERO-long saturation interval | Native median | FP8 median | Native / FP8 |
| --- | ---: | ---: | ---: |
| First exposure to these history positions | 4688.51 ms | 4707.49 ms | 1.00× |
| Repeat with the CUDA Graph shapes already established | 4694.00 ms | 2596.81 ms | 1.81× |

The first FP8 cycle costs 6594.85 ms versus 4473.28 ms native, separate from
setup (21.97 s FP8 / 25.52 s native). FP8 graphs are keyed by stream, spans and
history positions; the warm result does not describe first exposure. These are
one process per arm, one episode per phase, not sustained-tail or thermal
qualification. All ratios compare whole runtimes, not isolated FP8 arithmetic.

The stress sequence uses recorded initial/first-commit windows, then repeats
the last full recorded observation window and executed-action chunk. It is
deliberately **not a physically continuous robot episode**. Decoded native/FP8
action MAE is 0.010084 and maximum absolute difference is 1.523438 over all
retained chunks. Those mixed-unit deltas are not a task-success loss estimate.
No closed-loop quality certificate is established here.

The conservative saturation interval is zero-based cycles 15–47: no-eviction
growth exceeds capacity 2160 and the post-transient occupancy reaches its
plateau. Native terminal elision defers provisional action reservation until
the next cache clear; FP8 reserves those 16 slots immediately. Thus native
post-call live count is 2144 and FP8 is 2160. Subtracting FP8's provisional
action slots gives matching occupancy, with the FP8 slab head advancing. This
checks count/position semantics, **not token-identity or numerical equivalence**.

[Verified comparison](va-saturated-libero-comparison.json),
[native receipt](va-saturated-stress-libero-native.json),
[FP8 receipt](va-saturated-stress-libero-fp8.json),
[frozen protocol](va-saturated-stress-protocol.json).
All 368 source/config/input hashes were rechecked unchanged during the pair.
Raw JSON/NPZ artifacts live under
`/home/ubuntu/ifl_eval/thor_precision_completion_20260909/` and the corresponding
`/home/guanming/ifl_eval/thor_precision_completion_20260909/` on Thor.

Reproduce the calls with [the probe](probe_va_saturated_stress.py) in the native
VA environment, using the protocol's pinned source/config/input artifacts.
Validate complete arms with:

```bash
python eval/thor_precision_completion_2026-09-09/compare_va_saturated_stress.py \
  --native /path/to/va-saturated-stress-libero-native.json \
  --fp8 /path/to/va-saturated-stress-libero-fp8.json \
  --family libero --output /path/to/new-comparison.json
```

Each JSON requires its adjacent `.npz`; the verifier rejects incomplete,
hash-mismatched or semantically inconsistent artifacts and refuses output overwrite.
