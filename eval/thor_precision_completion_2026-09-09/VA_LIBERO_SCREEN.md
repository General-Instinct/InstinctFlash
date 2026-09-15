# VA Runtime LIBERO paired screen on Thor

The complete fixed LIBERO-10 campaign measured **native 20/20 successes** and
**FP8 18/20**, an observed **−10 percentage-point** difference. Both discordant
scenes succeeded only with native; no scene succeeded only with FP8.

| Scene | Native | FP8 |
| --- | --- | --- |
| Task 2, seed 20001 | Success, 242 actions | Unsuccessful, 796 actions |
| Task 7, seed 70001 | Success, 280 actions | Unsuccessful, 796 actions |

All other 18 pairs succeeded with both precision modes. Both unsuccessful
FP8 episodes reached the original horizon normally and were retained without
retry, parameter tuning or seed substitution.

Both arms ran the public Runtime on Thor with the pinned
`robbyant/lingbot-va-posttrain-libero-long` checkpoint, revision
`0e89d1e753019988aba484e8da2dc0810e264d9f`, original 20 video / 50 action steps,
BF16 conditioning and identical numerical switches. The simulator used five
zero-action settling steps, the native first-12/subsequent-16 action windows
and actual observed history. Its horizon is checked between complete action
chunks, so unsuccessful episodes contain 796 executed actions plus five
settling actions. The model had no access to future observations.

All 40 episode wire traces and executed action arrays passed independent
reconstruction and hash verification. All 20 paired initial camera inputs were
byte-identical. The complete ordered jobs, endpoint identities, checkpoint
inventory and bytes, frozen source/config files, four compiled libraries and
unchanged Thor boot identity passed final verification for both arms.

This is a small paired **SCREEN**, not an estimate of population loss or proof
of non-inferiority. The simulator pauses while waiting for inference; these
results do not certify real-time deadlines. Startup replay and synthetic-history
latency/saturation measurements remain separate evidence.

- [Verified comparison](va-runtime-libero-paired-comparison.json)
- [Native completion](va-runtime-libero-native-complete.json) /
  [FP8 completion](va-runtime-libero-fp8-complete.json)
- [Native final source check](va-runtime-libero-native-source-final.json) /
  [FP8 final source check](va-runtime-libero-fp8-source-final.json)
- [Protocol and preparation history](VA_RUNTIME_STARTUP.md)

Raw reproducible scripts, frozen source-v4, simulator manifest, traces and arrays:
`/home/ubuntu/ifl_eval/thor_precision_completion_20260909/va-runtime-campaign/`.
`compare_arms.py` requires both complete arms and their final source receipts
before producing the report. Earlier incomplete progress snapshots are historical.
