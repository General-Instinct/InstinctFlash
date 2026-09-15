"""Compare verified native Thor timing files under explicit illustrative budgets."""
import argparse
from pathlib import Path

from benchmarks.vla.realtime import assess, decision
from benchmarks.vla.util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic


def summarize(measurements, budgets):
    raw = {mode: load_json(measurements / (mode + '.latency.json')) for mode in ('stock', 'runtime_default')}
    for value in raw.values():
        if value.get('sha256') != sha256_json({k: v for k, v in value.items() if k != 'sha256'}):
            raise ConfigurationError('timing identity mismatch')
    for field in ('model', 'revision', 'checkpoint_sha256', 'upstream_sha256', 'packages',
                  'trace_sha256', 'pipeline_sha256', 'executed_actions', 'input_cases'):
        if raw['stock'][field] != raw['runtime_default'][field]:
            raise ConfigurationError(f'timing arms differ in {field}')
    scenarios = []
    for path in sorted(budgets.glob('*.json')):
        budget = load_json(path)
        rows = [{'mode': mode, 'tier': value['tier'], 'assessment': assess(value, budget)}
                for mode, value in raw.items()]
        scenarios.append({'name': path.stem, 'assessments': rows, 'decision': decision(rows)})
    if not scenarios:
        raise ConfigurationError('explicit budgets required')
    return {'schema_version': 1, 'synthetic': False, 'deployment_certified': False,
            'scope': 'Observed native policy calls under illustrative budgets; no sensor/actuator or sustained controller certification',
            'measurements': {mode: {'path': str(measurements / (mode + '.latency.json')),
                                    'file_sha256': sha256_file(measurements / (mode + '.latency.json'))}
                             for mode in raw}, 'scenarios': scenarios}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--budgets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ConfigurationError('refusing to overwrite timing analysis')
    write_json_atomic(args.output, summarize(args.measurements.resolve(), args.budgets.resolve()))


if __name__ == '__main__':
    main()
