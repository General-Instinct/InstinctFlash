"""Publish measured results and illustrative budget choices after both campaigns finish."""
import argparse
from pathlib import Path
from benchmarks.vla.configuration_select import select
from benchmarks.vla.execution_evidence import verify_record
from benchmarks.vla.realtime import percentile
from benchmarks.vla.util import load_json, write_json_atomic, sha256_file


def summarize(root, output):
    if not load_json(root/'postprocess.complete.json')['complete']:
        raise RuntimeError('paired experiments and bound measurements must finish first')
    analysis=load_json(root/'analysis.json')
    records=[load_json(root/(name+'.execution.json')) for name in ('ab_a','ab_b')]
    for record in records:
        verify_record(record)
    plan=load_json(root/'repeat1/evidence/plan.json')
    tasks={}
    for job in plan['jobs']:
        request=job['request']
        tasks.setdefault(request['suite_id'],set()).add(request['task'])
    policy={'baseline_profile_id':records[0]['profile']['profile_id'],
            'allowed_precisions':['native'],'tier_ceiling':'numeric','allow_step_change':False,
            'max_quality_loss':0.05,'quality_registry_sha256':plan['registry_sha256'],
            'required_suites':sorted(tasks),'required_tasks':{k:sorted(v) for k,v in tasks.items()}}
    common={'control_hz':20,'executed_actions':50,'scheduling':'pipelined',
            'max_deadline_miss_fraction':0.01,'target_device':'H100'}
    budgets={'pipelined_20hz':common,
             'reaction_500ms':dict(common,max_observation_to_action_ms=500),
             'blocking_50hz':dict(common,control_hz=50,scheduling='blocking'),
             'thor_requires_local_evidence':dict(common,target_device='Thor')}
    output.mkdir(parents=True,exist_ok=True)
    write_json_atomic(output/'selection-policy.json',policy)
    selections={}
    for name,budget in budgets.items():
        result=select(records,budget,policy)
        write_json_atomic(output/(name+'.budget.json'),budget)
        write_json_atomic(output/(name+'.selection.json'),result)
        selections[name]={'selected_profile_id':result['selected_profile_id'],'reason':result['reason'],
                          'expected_cli_exit_code':0 if result['selected_profile_id'] else 3}
    timings={}
    for name,record in zip(('original','native_capture'),records):
        samples=record['latency']['samples_ms']
        timings[name]={'profile_id':record['profile']['profile_id'],'count':len(samples),
                       'p50_ms':percentile(samples,.5),'p95_ms':percentile(samples,.95),
                       'p99_ms':percentile(samples,.99),'max_ms':max(samples),
                       'quality_status':record['quality_status']}
    campaign_seconds={name:(root/name/'complete.json').stat().st_mtime-(root/(name+'.launch.json')).stat().st_mtime
                      for name in ('repeat1','repeat2')}
    summary=dict(analysis,latency=timings,selection_scenarios=selections,
                 closed_loop_noise_scope='Same declared episode seeds and frozen scenes; actual-noise tensors were not recorded or injected. AA/BB measure full-system repeatability, not isolated kernel error.',
                 campaign_wall_seconds=campaign_seconds,
                 campaign_timing_scope='Operational launch-to-completion marker times, including simulator startup and per-episode asset checks; independent campaigns overlap.',
                 scenario_scope='Illustrative budgets, not a user-specified robot deployment contract.',
                 latency_scope='128 endpoint roundtrips per arm after both simulator campaigns; real observations, 8 warmups, seeded reset, no stored-noise injection.')
    write_json_atomic(output/'summary.json',summary)
    paths=[root/'analysis.json',root/'installed-smoke.json',root/'startup-findings.json',
           root/'independent-repeat-preregistration.json',root/'independent-repeat-result.json']
    for name in ('ab_a','ab_b'):
        paths.extend(root/(name+suffix) for suffix in ('.receipt.json','.latency.json','.execution.json'))
    for name in ('repeat2_a','repeat2_b'):
        paths.append(root/(name+'.receipt.json'))
    paths.extend(sorted(root.glob('*.py')))
    paths.extend(sorted((root/'latency-inputs').iterdir()))
    paths.extend(sorted((root/'wheels').glob('*.whl')))
    write_json_atomic(output/'evidence-index.json',{'external_root':str(root),
        'files':[{'path':str(path),'sha256':sha256_file(path)} for path in paths],
        'bundles':analysis['bundles'],
        'restore':'Restore the archived source/bundles/receipts/measurements; rebuild execution records against new absolute paths before selection.'})
    rows=[]
    for comparison in ('ab_repeat1','ab_repeat2','aa','bb'):
        for row in analysis['comparisons'][comparison]:
            lo,hi=row['tango_central95']
            setting={'robotwin50_easy':'clean','robotwin50_hard':'randomized'}[row['suite_id']]
            rows.append(f"| {comparison} | {setting} | {row['left_successes']}/{row['pairs']} → {row['right_successes']}/{row['pairs']} | {100*row['delta']:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] | {row['identical_action_pairs']}/{row['pairs']} |")
    timing_rows=[f"| {name} | {r['p50_ms']:.2f} | {r['p99_ms']:.2f} | {r['max_ms']:.2f} | {r['quality_status']} |" for name,r in timings.items()]
    startup=analysis['startup']
    text='''# Native V2 execution evidence and budget selection

Two independent original/native-capture A/B repeats completed **160 real episodes**:
10 tasks × 2 seeds × clean/randomized × 2 arms × 2 repeats. The checkpoint is
`robbyant/lingbot-vla-v2-6b-robotwin@0451855729ec904f970600e0aec8b84661423afe`.
Both arms retain native precision, NFE 10, 50 × 14 action chunks, TF32 enabled and
cuDNN benchmark disabled. Capture is NUMERIC; no FP8, distillation or reduced schedule was evaluated.

## Closed-loop screening

A/A compares originals across repeats; B/B compares accepted capture endpoints across
repeats. These reuse the same episodes and are not additional independent seeds.
Episodes use the same declared seeds and frozen scenes; actual-noise tensors were
not recorded or injected. A/A and B/B include full simulator/runtime repeatability,
not isolated kernel error.
Intervals are descriptive central 95% paired Tango score intervals in percentage
points; task clustering is not modeled. `identical` compares complete executed
action-stream digests. These 20 pairs per setting remain SCREEN, not a 5-point
noninferiority certificate or a BITEXACT claim.

| Comparison | Setting | Successes | Delta (pp) | 95% interval (pp) | Identical |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+f'''

Capture admission: **{startup['capture_ready']}/{startup['capture_attempts']}** valid eight-warmup
startups passed. One independent startup was rejected at max-absolute delta
{startup['rejected_max_abs']} against the unchanged {startup['unchanged_gate']} gate.
The extra independent startup was limited to one prospectively recorded attempt.
Quality results are conditional on accepted endpoints; three trials do not estimate
a stable startup failure rate. Failed setup and partial pilot campaigns are retained
externally and excluded from the 160 analyzed episodes.

## Separately measured endpoint latency

Each arm: 8 warmups, 128 measured calls using real observations, after both simulation
campaigns. Times include serialization and transport; they are not GPU-only latency.

| Execution | p50 (ms) | p99 (ms) | max (ms) | Quality |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(timing_rows)+'''

These are H100 execution records. They do not certify Thor, another checkpoint or
new source/precision/schedule. Paused simulation and 128 calls do not certify a
sustained robot control loop. Independent accepted endpoints match the recorded
hardware/software profile; physical GPU UUID and device load are not profile keys.

## Budget choices and reproducibility

The four checked-in `*.selection.json` files use illustrative explicit budgets:
20 Hz pipelined chunks (2500 ms), an added 500 ms reaction deadline, 50 Hz blocking
calls (20 ms), and a Thor target requiring Thor evidence. Baseline is retained when
it meets the observed budget. SCREEN cannot authorize selecting the accelerated
candidate when baseline misses. Precision permission, step permission and the exact
quality protocol/task set are separate inputs.

`summary.json` contains all episode pairs, intervals, timing summaries, campaign wall times and decisions.
`evidence-index.json` pins external raw bundles, measured receipts, latency samples,
records, scripts and the installed wheel. Evidence records verify those sources and
recompute gates; editing a summary and rehashing it cannot create a certificate.
The frozen GPU execution predates the added reporting/explain interface; its own
source digest is retained rather than assigning its measurements to newer code.

After restoring the external root, reproduce postprocessing with the installed wheel:

```sh
python eval/precision_evidence_2026-09-06/analyze.py --root "$EVIDENCE_ROOT" --output /tmp/v2-analysis.json
python eval/precision_evidence_2026-09-06/summarize.py --root "$EVIDENCE_ROOT" --output /tmp/v2-summary
```

Replay scripts and exact environment paths are retained in the external root;
rerunning simulation requires the pinned RoboTwin assets and model/environment
inventories in each verified bundle. See [execution workflow](../../docs/rfc/execution-evidence.md)
for rebuilding records after relocation and using the three CLI commands.
'''
    (output/'README.md').write_text(text)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    summarize(args.root,args.output)
