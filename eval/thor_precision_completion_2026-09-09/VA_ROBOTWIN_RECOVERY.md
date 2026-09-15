# LingBot-VA RoboTwin FP8 recovery

The original FP8 controller stopped after 13 verified trials (11 successes and two task failures). The next fixed scene, clean `dump_bin_bigbin/110101`, was rejected by the simulator expert gate before any policy call. It is an operational interruption, not an additional task failure.

A separate expert-only diagnostic reproduced the frozen initial camera/state digest and passed the expert check. Success and object positions were identical before and after `close_env`. This does not establish the underlying source of expert nondeterminism; it provides no justification for changing simulator teardown or success criteria.

`recover_robotwin_fp8.py` continues in a separate `robotwin-fp8-recovery-v1` directory. It verifies and references the original 13 trials without rerunning them. The original failed campaign remains intact. Each remaining scene uses the original frozen job, simulator, model identity, schedule, and pinned seed. The existing driver checks the initial state against the frozen scene before the expert and policy runs.

There are at most three new attempts per remaining scene. A retry requires the specific pinned-seed expert rejection, a complete trace with **zero policy calls**, matching model identity, and no result file. Any model call, model task failure, unexpected error, timeout, or evidence mismatch prevents automatic retry. Each attempt retains its log and trace with hashes. Task failures count normally and advance the campaign.

This is an explicit recovery amendment to the original no-retry execution policy: pre-policy expert attempts may repeat. It is not a claim of deterministic physics, non-inferiority, or real-time performance. The successful native arm did not need this recovery. The final paired report carries the recovery history.

`verify_robotwin_recovery.py` checks provenance, symlink targets, attempt limits, rejected-attempt traces, all completed policy traces, and controller actions. `compare_robotwin_recovery.py` additionally requires 40 completed trials, the final Thor source/weight receipt, and byte-identical paired initial inputs. The monitor now watches the recovery directory and runs those checks and the final comparison after completion. It does not restart arbitrary failed processes.

The scripts run alongside the original frozen campaign evidence at `/home/ubuntu/ifl_eval/thor_precision_completion_20260909/va-runtime-campaign`; they are not standalone model downloads. Completion and scores must be read from the live receipts, not inferred from this document.
