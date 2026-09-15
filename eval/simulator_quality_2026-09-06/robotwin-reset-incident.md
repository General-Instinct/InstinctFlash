# RoboTwin reset reproducibility repair

The randomized `dump_bin_bigbin` scene at requested/resolved seed 120100 failed
initial-observation validation before policy inference in both LingBot-VLA campaigns.
A second V2 attempt reproduced the same failure. Both initial runs stopped at
64 of 80 completed episodes. These runs remain under
`/home/ubuntu/ifl_eval/screen10_20260906/joint/` as diagnostic evidence; they are
excluded from the corrected primary campaign.

The native `Base_Task.get_cluttered_table` adds `table_xy_bias` to mutable default
bounds. This task supplies an X offset of 0.3. Preparing a scene runs the expert and
then resets again, while an evaluation process starts cold. Repeated calls therefore
sample different distractors despite identical seeds. Robot state matches, but
camera images differ substantially.

| Reset path | Initial observation SHA-256 |
| --- | --- |
| Original preparation after expert | `7dc28a09313938b1c9514259afcc62ffd95a638af1b82843bd6adae4fa35fc4a` |
| Original cold process | `8f2f2c7b725f28e4b17aa2588a9bccee36556fc769185005b641e03f218d2eff` |
| Fixed preparation after expert | `8f2f2c7b725f28e4b17aa2588a9bccee36556fc769185005b641e03f218d2eff` |
| Fixed cold process | `8f2f2c7b725f28e4b17aa2588a9bccee36556fc769185005b641e03f218d2eff` |

The InstinctFlash driver copies the sampler bounds per call in both paths and
records `reset_policy: fresh-clutter-bounds-v1` in new scene manifests. It refuses
old manifests. The native simulator checkout remains unchanged. Regression tests
cover repeated resets and explicit caller-supplied bounds. Live probe observations,
logs and a visual comparison are retained under the experiment root's `reset-probes/`.

Corrected campaigns use fresh scene manifests and new run directories under
`joint-stable/`, with scene preparation snapshot
`72ed8605ae3a7a196fd26d5081e99f57d9e81d95` and paired-runner execution snapshot
`229eec2da7cc86f85bbfe70f313f6d5145f96aff`. No earlier result is relabeled or reused.
An infrastructure reset failure is never counted as policy failure, and the
initial-observation equality check remains mandatory.

The old V2 clean subset was complete before the randomized reset failure:
18/20 original successes versus 17/20 candidate successes. Its `beat_block_hammer`
seed 60100 pair succeeded in the original and failed in the candidate, first
separating at zero-based controller step 50. The corrected campaign reuses no
results. 19/20 clean scene records (initial observation, prompt and seed), including the
discordant hammer scene, match between old and fresh manifests. This observation remains
relevant evidence of variation across runs. The corrected campaign also changes
the execution schedule from serial to paired concurrency; these runs do not
isolate the cause of that variation. A stock-versus-stock repeatability study is
needed before attributing a particular discordance solely to a numerical pass.
The remaining clean scene is `dump_bin_bigbin` requested seed 110101: the old
expert gate resolved to 110113, while the new gate retained 110101. Both actual
seeds and corresponding observations/prompts are preserved.
The [pilot trace diagnostics](vla2-pilot-trace-diagnostics.json) are explicitly
incomplete and are not pooled into the primary success estimates.
