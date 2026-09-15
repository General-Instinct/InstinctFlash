"""Replace cross-device control comparisons with validated same-Thor controls."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from run_control import digest


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def checked(item):
    path = Path(item['path'])
    assert digest(path) == item['sha256'], path
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes']
    return path


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = Path(__file__).parent
    quality = source.parent / 'realtime-quality'
    receipt = read(quality / 'deployed_score_receipt.json')
    deployed = read(checked(receipt['report']))
    assert deployed['status'] == 'success' and deployed['independent_numerical_crosscheck']['status'] == 'success'
    protocol = read(quality / 'protocol.json')
    assert digest(quality / 'protocol.json') == deployed['protocol_sha256']
    for identity in deployed['scoring_sources'] + deployed['independent_review_sources']:
        checked(identity)
    scoring_path = next(Path(i['path']) for i in deployed['scoring_sources']
                        if Path(i['path']).name == 'evaluate_cosmos3_pair_v13.py')
    reviewer_path = next(Path(i['path']) for i in deployed['independent_review_sources']
                         if Path(i['path']).name == 'audit_cosmos3_pair_evaluation_v13.py')
    scoring = load_module(scoring_path, 'same_thor_frozen_scoring')
    reviewer = load_module(reviewer_path, 'same_thor_independent_scoring')
    teacher_id = next(r['bank'] for r in protocol['reuse']['banks'] if r['configuration_id'] == 'teacher_unipc32')
    teacher_report = read(checked(teacher_id))
    fields = ('action', 'model_action', 'measured_action', 'model_measured_action')
    expected_teacher_requests = [r for r in protocol['requests'] if r['configuration_id'] == 'teacher_unipc32']
    assert len(teacher_report['requests']) == len(expected_teacher_requests) == 128
    teacher_rows = []
    for identity, expected in zip(teacher_report['requests'], expected_teacher_requests, strict=True):
        record = read(checked(identity))
        assert record['request'] == expected
        with np.load(checked(record['artifact']), allow_pickle=False) as data:
            teacher_rows.append({key: data[key].copy() for key in fields})
    teacher = {key: np.stack([row[key] for row in teacher_rows]).reshape(16, 8, 32, 8) for key in fields}
    controls, independent, inputs = {}, {}, []
    for config in ('original_unipc4', 'original_sde1_cfg1', 'original_sde1_cfg4'):
        directory = root / f'control-compact-{config}-v2'
        report = read(directory / 'report.json')
        assert report['status'] == 'success' and report['configuration'] == config
        assert report['capture_validation']['capture_validation_complete']
        assert report['capture_validation']['requests'] == 128
        for name, sha in report['sources'].items():
            assert digest(source / name) == sha
        assert report['student_validator_sha256'] == digest(quality / 'validate_capture.py')
        cfgs = (1, 4) if config == 'original_unipc4' else (int(config[-1]),)
        expected_students = {f'edge_sde1_cfg{cfg}_seed{seed}_update0064' for cfg in cfgs for seed in (12031, 12032)}
        assert {row['configuration_id'] for row in report['paired_students'].values()} == expected_students
        assert all(row['validation_complete'] and row['requests'] == 128 for row in report['paired_students'].values())
        assert [r['request'] for r in report['requests']] == [r for r in protocol['requests'] if r['configuration_id'] == config]
        if config == 'original_unipc4':
            assert report['exclusion'] == dict(action_rows=33, columns=[8, 64], reason='Native versus zero action padding')
        else:
            assert report['exclusion'] is None
        path = directory / report['archive']['file']
        assert digest(path) == report['archive']['sha256']
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == set(fields)
            assert all(data[key].shape == (128, 32, 8) and data[key].dtype == np.float32
                       and np.isfinite(data[key]).all() for key in fields)
            bank = {key: data[key].reshape(16, 8, 32, 8).copy() for key in fields}
        controls[config] = scoring.episode_metrics(bank, teacher)
        independent[config] = reviewer.episode_metrics(bank, teacher)
        for key, value in controls[config].items():
            reviewer.close(value, independent[config][key], f'{config}/{key}')
        inputs.extend([dict(path=str(directory / 'report.json'), sha256=digest(directory / 'report.json')),
                       dict(path=str(path), sha256=digest(path))])
    comparisons = []
    counts = reviewer.bootstrap_counts()
    for candidate, summaries in deployed['summaries'].items():
        cfg = 1 if candidate.startswith('edge_sde1_cfg1_') else 4
        for baseline in ('original_unipc4', f'original_sde1_cfg{cfg}'):
            values = {}
            for key, summary in summaries.items():
                left = np.array(summary['episode_values'])
                values[key] = scoring.bootstrap_difference(left, controls[baseline][key])
                reviewer.compare_tree(values[key], reviewer.bootstrap_difference(left, independent[baseline][key], counts))
            comparisons.append(dict(candidate=candidate, baseline=baseline, metrics=values))
    assert len(comparisons) == 24
    result = dict(status='success', quality_certified=False,
        scope='All eight deployed variants plus within-episode two-seed means against same-Thor native original controls; historical development only',
        script_sha256=digest(__file__), deployed_score_report=receipt['report'], inputs=inputs,
        primary_metrics=deployed['primary_metrics'], comparisons=comparisons,
        control_summaries={config: {key: dict(mean=float(value.mean()), episode_values=value.tolist())
            for key, value in bank.items()} for config, bank in controls.items()},
        independent_numerical_crosscheck=dict(status='success', control_metric_vectors=288,
            comparison_metric_bootstraps=2304, tolerance=reviewer.TOLERANCE),
        limitations=['Reused 16 development episodes, not fresh confirmation.',
            'Original UniPC4 native padding intentionally differs from zero-padded SDE; matched SDE controls have identical inputs.',
            'Teacher32 remains the independently sampled H100 reference, not command ground truth.',
            'No selected checkpoint, aggregate winner metric, multiple-testing correction, or noninferiority margin.',
            'Actual controller deadline, fresh-observation age and closed-loop task success remain unqualified.'])
    for identity in inputs:
        checked(identity)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = run(args.root, args.output)
    print(json.dumps(dict(status=result['status'], comparisons=len(result['comparisons']), quality_certified=False)))
