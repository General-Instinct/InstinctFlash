"""Score all eight deployed banks with frozen producer math; never certify quality."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from validate_capture import digest


FIELDS = ('action', 'model_action', 'measured_action', 'model_measured_action')


def read(path):
    return json.loads(Path(path).read_text())


def checked(item):
    path = Path(item['path'])
    assert digest(path) == item['sha256'], path
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes'], path
    return path


def run(pair_root, replay, source, output):
    pair_root, replay, source, output = map(Path, (pair_root, replay, source, output))
    if output.exists():
        raise FileExistsError(output)
    protocol = read(source / 'protocol.json')
    plan = read(source / 'plan.json')
    assert digest(source / 'protocol.json') == plan['protocol_sha256']
    audit = read(replay / 'independent_audit.json')
    assert audit['status'] == 'success' and audit['valid'] is True
    assert audit['evaluation_protocol']['sha256'] == plan['protocol_sha256']
    analysis = read(checked(audit['analysis_report']))
    assert analysis['status'] == 'success'
    assert audit['coverage']['raw_requests'] == 1280
    assert audit['coverage']['comparison_metric_bootstraps'] == 2112
    scoring_sources = [item for item in protocol['sources'] if Path(item['path']).name in
                       ('evaluate_cosmos3_pair_v13.py', 'analyze_cosmos3_energy_score_v9.py')]
    assert len(scoring_sources) == 2
    for item in scoring_sources:
        checked(item)
    scoring_path = next(Path(item['path']) for item in scoring_sources
                        if Path(item['path']).name == 'evaluate_cosmos3_pair_v13.py')
    spec = importlib.util.spec_from_file_location('frozen_deployed_scores', scoring_path)
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    assert scoring.K == 8 and scoring.HORIZONS == (1, 32)
    admission = read(checked(audit['admission']))
    review_sources = admission['sources']
    for item in review_sources:
        checked(item)
    review_path = next(Path(item['path']) for item in review_sources
                       if Path(item['path']).name == 'audit_cosmos3_pair_evaluation_v13.py')
    spec = importlib.util.spec_from_file_location('independent_deployed_scores', review_path)
    reviewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reviewer)
    teacher_id = next(row['bank'] for row in protocol['reuse']['banks']
                      if row['configuration_id'] == 'teacher_unipc32')
    teacher_report = read(checked(teacher_id))
    teacher_requests = [r for r in protocol['requests'] if r['configuration_id'] == 'teacher_unipc32']
    assert len(teacher_report['requests']) == len(teacher_requests) == 128
    teacher_rows = []
    for identity, request in zip(teacher_report['requests'], teacher_requests, strict=True):
        record = read(checked(identity))
        assert record['status'] == 'success' and record['request'] == request
        with np.load(checked(record['artifact']), allow_pickle=False) as archive:
            teacher_rows.append({key: archive[key].copy() for key in FIELDS})
    teacher = {key: np.stack([row[key] for row in teacher_rows]).reshape(16, 8, 32, 8)
               for key in FIELDS}
    metrics, independent, inputs = {}, {}, []
    for cfg in (1, 4):
        for seed in (12031, 12032):
            config = f'edge_sde1_cfg{cfg}_seed{seed}_update0064'
            directory = pair_root / f'quality-pair-cfg{cfg}-seed{seed}-v1'
            report = read(directory / 'report.json')
            assert report['status'] == 'success' and report['configuration_id'] == config
            assert report['protocol_sha256'] == plan['protocol_sha256']
            assert report['plan_sha256'] == digest(source / 'plan.json')
            assert report['script_sha256'] == digest(source / 'export_pair.py')
            assert report['validator_sha256'] == digest(source / 'validate_capture.py')
            assert [r['request'] for r in report['requests']] == [
                r for r in protocol['requests'] if r['configuration_id'] == config]
            assert len(report['requests']) == 128
            inputs.append(dict(path=str(directory / 'report.json'), sha256=digest(directory / 'report.json')))
            for attention in ('native', 'cudnn'):
                archive_id = report['archives'][attention]
                path = directory / archive_id['file']
                assert digest(path) == archive_id['sha256']
                with np.load(path, allow_pickle=False) as archive:
                    assert set(archive.files) == set(FIELDS)
                    bank = {}
                    for key in FIELDS:
                        value = archive[key]
                        assert value.shape == (128, 32, 8) and value.dtype == np.float32
                        assert np.isfinite(value).all()
                        bank[key] = value.reshape(16, 8, 32, 8).copy()
                name = f'{config}/{attention}'
                metrics[name] = scoring.episode_metrics(bank, teacher)
                independent[name] = reviewer.episode_metrics(bank, teacher)
                assert set(metrics[name]) == set(independent[name])
                for key, value in metrics[name].items():
                    reviewer.close(value, independent[name][key], f'{name}/{key}')
                inputs.append(dict(path=str(path), sha256=archive_id['sha256']))
    comparisons = []
    for cfg in (1, 4):
        for attention in ('native', 'cudnn'):
            key = f'edge_sde1_cfg{cfg}_paired_seed_mean_update0064/{attention}'
            left, right = [metrics[f'edge_sde1_cfg{cfg}_seed{seed}_update0064/{attention}']
                           for seed in (12031, 12032)]
            metrics[key] = {name: .5 * (left[name] + right[name]) for name in left}
            first, second = [independent[f'edge_sde1_cfg{cfg}_seed{seed}_update0064/{attention}']
                             for seed in (12031, 12032)]
            independent[key] = {name: np.mean(np.stack((first[name], second[name])), axis=0)
                                for name in first}
        for seed in (12031, 12032, 'paired_seed_mean'):
            prefix = f'edge_sde1_cfg{cfg}_' + (f'seed{seed}' if isinstance(seed, int) else seed) + '_update0064'
            candidate, baseline = prefix + '/cudnn', prefix + '/native'
            comparisons.append(dict(candidate=candidate, baseline=baseline, scope='same Thor inputs',
                metrics={name: scoring.bootstrap_difference(value, metrics[baseline][name])
                         for name, value in metrics[candidate].items()}))
        for attention in ('native', 'cudnn'):
            candidate = f'edge_sde1_cfg{cfg}_paired_seed_mean_update0064/{attention}'
            for baseline in ('original_unipc4', f'original_sde1_cfg{cfg}'):
                comparisons.append(dict(candidate=candidate, baseline=baseline,
                    scope='Thor deployed versus H100 historical control; not cross-device input bitexact',
                    metrics={name: scoring.bootstrap_difference(value,
                        np.array(analysis['summaries'][baseline][name]['episode_values']))
                        for name, value in metrics[candidate].items()}))
    counts = reviewer.bootstrap_counts()
    for comparison in comparisons:
        left = independent[comparison['candidate']]
        baseline = comparison['baseline']
        right = independent[baseline] if baseline in independent else {
            key: np.array(value['episode_values']) for key, value in analysis['summaries'][baseline].items()}
        for key, value in comparison['metrics'].items():
            reviewer.compare_tree(value, reviewer.bootstrap_difference(left[key], right[key], counts),
                                  f"{comparison['candidate']} versus {baseline}/{key}")
    result = dict(status='success', quality_certified=False,
        scope='Historical development diagnostics only; no noninferiority, closed-loop, or realtime admission',
        protocol_sha256=plan['protocol_sha256'], script_sha256=digest(__file__),
        producer_audit=dict(path=str(replay / 'independent_audit.json'), sha256=digest(replay / 'independent_audit.json')),
        scoring_sources=scoring_sources, independent_review_sources=review_sources,
        independent_numerical_crosscheck=dict(status='success', tolerance=reviewer.TOLERANCE,
            raw_variant_metric_vectors=8 * 96, comparison_metric_bootstraps=len(comparisons) * 96),
        inputs=inputs, primary_metrics=list(scoring.PRIMARY_METRICS),
        summaries={key: {name: dict(mean=float(value.mean()), episode_values=value.tolist())
                         for name, value in bank.items()} for key, bank in metrics.items()},
        comparisons=comparisons,
        limits=['All four students and both deployed variants included; no checkpoint selection.',
                'Two training seeds averaged within 16 episodes, never treated as 32 independent episodes.',
                'No best-of-K, metric aggregation, multiple-testing correction, or noninferiority margin.',
                'Numerical crosscheck uses separately implemented formulas, not an independent capture audit.',
                'Producer final completion inventory and closed-loop deployment review remain separate gates.'])
    for item in [*inputs, *scoring_sources, *review_sources]:
        checked(item)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair_root', type=Path)
    parser.add_argument('replay', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    result = run(args.pair_root, args.replay, args.source, args.output)
    print(json.dumps(dict(status=result['status'], variants=len(result['summaries']),
                         comparisons=len(result['comparisons']), quality_certified=False)))
