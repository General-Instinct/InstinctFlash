"""Derive next-step summaries from completed, verified experiment artifacts."""
import argparse
from pathlib import Path

from benchmarks.vla.bundle import verify_bundle
from benchmarks.vla.fixed_input import compare_actions
from benchmarks.vla.policy_trace import read_trace
from benchmarks.vla.plan import pipeline_digest
from benchmarks.vla.util import ConfigurationError, load_json, sha256_file, write_json_atomic


def summarize(root):
    summaries = {}
    for name in ('aa', 'bb', 'ab', 'new-checkpoint'):
        bundle = root / name / 'evidence'
        verified = verify_bundle(bundle)
        report = load_json(bundle / 'report.json')
        if not report['complete'] or report['issues']:
            raise ConfigurationError(f'{name} is incomplete or invalid')
        comparison = report['comparisons'][0]
        action_rows = {row['suite_id']: row for row in comparison['closed_loop_actions']}
        summaries[name] = {
            'bundle': str(bundle), 'verification': verified,
            'plan_id': report['plan_id'], 'report_sha256': report['report_sha256'],
            'execution_pipeline_sha256': report['execution_pipeline_sha256'],
            'settings': [dict(row, identical_action_pairs=action_rows[row['suite_id']]['matching_pairs'])
                         for row in comparison['paired_success']],
        }
        if name in ('aa', 'bb', 'ab'):
            import numpy as np

            def same(left, right):
                if isinstance(left, dict):
                    return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
                if isinstance(left, np.ndarray):
                    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
                return left == right

            pairs = {}
            for job in load_json(bundle / 'plan.json')['jobs']:
                pairs.setdefault(job['request']['pair_id'], []).append(job)
            first_calls = []
            for pair, jobs in pairs.items():
                calls = [read_trace(bundle / 'results' / ('.' + j['job_id'] + '.pending.trace'))[1] for j in jobs]
                left, right = [next(call for call in trace if not call[0].get('reset')) for trace in calls]
                first_calls.append({'pair_id': pair, 'task': jobs[0]['request']['task'],
                                    'suite': jobs[0]['request']['suite_id'],
                                    'input_identical': same(left[0], right[0]),
                                    'noise_identical': same(left[1]['benchmark_noise'], right[1]['benchmark_noise']),
                                    'action': compare_actions(left[1]['action'], right[1]['action'])})
            summaries[name]['first_calls'] = first_calls
    fixed = {}
    raw = {}
    for mode in ('stock', 'runtime_default'):
        path = root / (mode + '.fixed-input.json')
        data = load_json(path); raw[mode] = data
        if not data['rows'] or not data['all_noise_identical']:
            raise ConfigurationError('fixed-input replay lacks identical actual noise')
        repeated = [r for r in data['rows'] if r['repeat'] > 0]
        fixed[mode] = {
            'path': str(path), 'file_sha256': sha256_file(path), 'repeats': data['repeats'],
            'calls_per_repeat': len([r for r in data['rows'] if r['repeat'] == 0]),
            'all_noise_identical': data['all_noise_identical'],
            'repeat_comparisons': len(repeated),
            'identical_repeat_comparisons': sum(r['against_first_repeat']['identical'] for r in repeated),
            'max_abs_against_first_repeat': max(r['against_first_repeat']['max_abs'] for r in repeated),
            'recorded_comparisons': len(data['rows']),
            'identical_recorded_comparisons': sum(r['against_recorded']['identical'] for r in data['rows']),
            'max_abs_against_recorded': max(r['against_recorded']['max_abs'] for r in data['rows']),
        }
    def key(row):
        return row['trace'], row['call'], row['repeat']
    original = {key(r): r for r in raw['stock']['rows']}
    candidate = {key(r): r for r in raw['runtime_default']['rows']}
    if original.keys() != candidate.keys():
        raise ConfigurationError('replay arms cover different inputs/repeats')
    cross = [dict(trace=k[0], call=k[1], repeat=k[2],
                  **compare_actions(original[k]['action_values'], candidate[k]['action_values']))
             for k in original]
    return {'schema_version': 1, 'synthetic': False,
            'analysis_pipeline_sha256': pipeline_digest(), 'analysis_source_sha256': sha256_file(Path(__file__)),
            'scope': 'Attribution screening and distinct native checkpoint validation; not release certification',
            'closed_loop': summaries, 'fixed_input': fixed,
            'fixed_input_cross_arm': {'comparisons': len(cross),
                                      'identical': sum(r['identical'] for r in cross),
                                      'max_abs': max(r['max_abs'] for r in cross), 'rows': cross}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ConfigurationError('refusing to overwrite analysis')
    write_json_atomic(args.output, summarize(args.root.resolve()))


if __name__ == '__main__':
    main()
