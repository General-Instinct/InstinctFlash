"""Score the untrained original SDE2 diagnostic against frozen historical baselines."""
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
    postprocess = read(source / 'postprocess_plan.json')
    assert postprocess['status'] == 'frozen_before_scoring'
    assert postprocess['capture_plan_sha256'] == digest(source / 'plan.json')
    for name, sha in postprocess['sources'].items():
        assert digest(source / name) == sha
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
    report = read(root / 'report.json')
    protocol_new = read(source / 'protocol.json')
    assert report['status'] == 'success' and report['configuration'] == 'original_sde2_cfg1'
    assert report['capture_validation']['capture_validation_complete']
    assert report['capture_validation']['requests'] == 128 and report['exclusion'] is None
    assert [r['request'] for r in report['requests']] == protocol_new['requests']
    assert report['student_validator_sha256'] == digest(quality / 'validate_capture.py')
    assert {r['configuration_id'] for r in report['paired_students'].values()} == {
        f'edge_sde1_cfg1_seed{seed}_update0064' for seed in (12031, 12032)}
    assert all(r['validation_complete'] and r['requests'] == 128 for r in report['paired_students'].values())
    for name, sha in report['sources'].items():
        assert digest(source / name) == sha
    path = root / report['archive']['file']
    assert digest(path) == report['archive']['sha256']
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == set(fields)
        assert all(data[k].shape == (128, 32, 8) and data[k].dtype == np.float32
                   and np.isfinite(data[k]).all() for k in fields)
        bank = {k: data[k].reshape(16, 8, 32, 8).copy() for k in fields}
    values = scoring.episode_metrics(bank, teacher)
    independent = reviewer.episode_metrics(bank, teacher)
    for key in values:
        reviewer.close(values[key], independent[key], key)
    old_receipt = read(source.parent / 'realtime-controls/same_thor_score_receipt.json')
    old = read(checked(old_receipt['report']))
    assert old['status'] == 'success' and old['independent_numerical_crosscheck']['status'] == 'success'
    assert old['deployed_score_report'] == receipt['report']
    for identity in old['inputs']:
        checked(identity)
    baselines = dict(old['control_summaries'], **deployed['summaries'])
    assert len(baselines) == 15
    comparisons = []
    counts = reviewer.bootstrap_counts()
    for baseline, summaries in baselines.items():
        assert set(summaries) == set(values)
        metrics = {}
        for key in values:
            right = np.asarray(summaries[key]['episode_values'])
            metrics[key] = scoring.bootstrap_difference(values[key], right)
            reviewer.compare_tree(metrics[key], reviewer.bootstrap_difference(independent[key], right, counts))
        comparisons.append(dict(candidate='original_sde2_cfg1/native', baseline=baseline, metrics=metrics))
    inputs = [dict(path=str(root/'report.json'), sha256=digest(root/'report.json')),
              dict(path=str(path), sha256=digest(path)), old_receipt['report'], receipt['report']]
    result = dict(status='success', quality_certified=False, trained_student=False,
        scope='Original SDE2 CFG1 native historical diagnostic; no two-step training or fresh confirmation',
        script_sha256=digest(__file__), protocol_sha256=digest(source/'protocol.json'), inputs=inputs,
        primary_metrics=deployed['primary_metrics'], comparisons=comparisons,
        summaries={key: dict(mean=float(v.mean()), episode_values=v.tolist()) for key,v in values.items()},
        independent_numerical_crosscheck=dict(status='success', metric_vectors=len(values),
            comparison_metric_bootstraps=len(values)*len(comparisons), tolerance=reviewer.TOLERANCE),
        limitations=old['limitations'] + ['Native attention only; cuDNN requires its own paired quality capture.',
            'Both training seeds and both native/cuDNN one-step baselines retained; no winner selection.',
            'Original SDE2 outcomes were not read before freezing the request plan; prior development outcomes were seen.'])
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
