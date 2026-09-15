"""Descriptive paired success statistics, independent of release acceptance gates."""
from collections import defaultdict
from instinctflash.verify.certify import _tango_paired_score_bounds


def paired_success_summary(pairs, jobs, control_id):
    expected = defaultdict(set)
    observed = defaultdict(list)
    for job in jobs:
        request = job['request']
        if request['arm']['id'] == control_id and request['suite']['kind'] == 'closed_loop':
            expected[(request['model_id'], request['suite_id'])].add(request['pair_id'])
    for pair in pairs:
        request = pair[0]['request']
        if request['suite']['kind'] == 'closed_loop':
            observed[(request['model_id'], request['suite_id'])].append(pair)
    summaries = []
    for (model, suite), ids in sorted(expected.items()):
        entries = observed[(model, suite)]
        outcomes = [(p[1]['metrics']['success'], p[3]['metrics']['success']) for p in entries]
        n = len(outcomes)
        per_task = defaultdict(lambda: {'pairs': 0, 'control_successes': 0, 'treatment_successes': 0})
        for p, (control, candidate) in zip(entries, outcomes):
            row = per_task[p[0]['request']['task']]
            row['pairs'] += 1
            row['control_successes'] += int(control)
            row['treatment_successes'] += int(candidate)
        control_wins = sum(a for a, _ in outcomes)
        treatment_wins = sum(b for _, b in outcomes)
        summaries.append({
            'model_id': model, 'suite_id': suite, 'quality_gate': False,
            'scope': 'Descriptive paired episodes on the selected fixed tasks; not a release verdict or generalization across unseen tasks.',
            'pairs': n, 'expected_pairs': len(ids),
            'complete': n == len(ids),
            'missing_pair_ids': sorted(ids - {p[0]['request']['pair_id'] for p in entries}),
            'control_successes': control_wins, 'treatment_successes': treatment_wins,
            'control_success_rate': control_wins / n if n else None,
            'treatment_success_rate': treatment_wins / n if n else None,
            'delta': (treatment_wins - control_wins) / n if n else None,
            'control_only_successes': sum(a and not b for a, b in outcomes),
            'treatment_only_successes': sum(b and not a for a, b in outcomes),
            'tango_central95': list(_tango_paired_score_bounds(outcomes, z=1.959963984540054)) if n else None,
            'interval_assumption': 'Matched Bernoulli episode pairs; task clustering is not modeled. Seeds on a fixed task are not new tasks.',
            'per_task': dict(sorted(per_task.items())),
        })
    return summaries
