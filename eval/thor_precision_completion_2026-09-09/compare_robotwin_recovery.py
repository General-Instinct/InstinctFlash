"""Compare complete, independently verified Runtime RoboTwin arms only."""
from pathlib import Path
import numpy as np
from verify_robotwin_episode_v2 import verify
from benchmarks.vla.policy_trace import read_trace
from benchmarks.vla.util import load_json, sha256_file, sha256_json, write_json_atomic

root = Path(__file__).resolve().parent


def match_identities(native, fp8):
    assert native['precision'] == 'native' and fp8['precision'] == 'fp8'
    for key in ('model_id', 'model_revision', 'checkpoint_sha256', 'runtime_source_sha256',
                'engine_source_sha256', 'upstream_source_sha256', 'pipeline_sha256',
                'packages_sha256', 'checkpoint_manifest_sha256', 'package_declaration_sha256',
                'startup_observation_sha256', 'startup_prompt', 'hardware', 'numeric_environment',
                'protocol', 'seed_mode'):
        assert native[key] == fp8[key], key
    for key in ('nfe', 'guidance', 'grid_shifts', 'geometry', 'action_semantics_sha256', 'dtype'):
        assert native['execution'][key] == fp8['execution'][key], key


def collect(precision):
    folder = root / ('robotwin-fp8-recovery-v1' if precision == 'fp8' else 'robotwin-native-run')
    if precision == 'fp8':
        from verify_robotwin_recovery import collect as audit_recovery
        audit_recovery()
    complete = load_json(folder / 'complete.json')
    assert complete == load_json(folder / 'progress.json')
    assert complete['status'] == 'complete' and complete['precision'] == precision
    index = load_json(root / 'robotwin-jobs/index.json')
    assert complete['expected_jobs'] == index['jobs'] and len(complete['jobs']) == 40
    for field, name in [('runner_sha256', 'recover_robotwin_fp8.py' if precision == 'fp8' else 'run_robotwin_arm.py'),
                        ('verifier_sha256', 'verify_robotwin_episode.py'),
                        ('source_manifest_sha256', 'source-robotwin-v2.json'),
                        ('scene_sha256', 'robotwin-scenes.json')]:
        assert complete[field] == sha256_file(root / name), field
    identity_path = root / f'robotwin-{precision}-serve-v1.json'
    identity = load_json(identity_path)
    assert complete['identity'] == identity
    final = load_json(root / f'robotwin-{precision}-source-final.json')
    assert final['ok'] is True and final['identity_sha256'] == sha256_file(identity_path)
    assert final['source_files'] == 366 and final['compiled_libraries'] == 4
    scenes = load_json(root / 'robotwin-scenes.json')
    rows = {}
    for row, expected in zip(complete['jobs'], index['jobs']):
        assert row['verified'] is True and row['exit_code'] == 0
        amendment = load_json(root / 'robotwin-verifier-amendment.json')
        assert complete['verifier_sha256'] == amendment['original_sha256']
        assert complete.get('original_runner_sha256', complete['runner_sha256']) == amendment['runner_sha256']
        assert sha256_file(root / 'verify_robotwin_episode_v2.py') == amendment['corrected_sha256']
        assert row['scene_key'] == amendment['affected_returned_scene_key']
        assert row['result'] == expected['job']
        job_path = root / 'robotwin-jobs' / expected['job']
        assert job_path.parent == root / 'robotwin-jobs' and sha256_file(job_path) == expected['sha256']
        path = folder / row['result']
        assert path.parent == folder
        result = load_json(path)
        assert result['source_request_sha256'] == sha256_json(load_json(job_path)['request'])
        assert result['scene_manifest_sha256'] == complete['scene_sha256']
        verified = verify(path, identity, scenes)
        assert verified['scene_key'] == expected['scene_key']
        assert all(row[key] == value for key, value in verified.items() if key != 'scene_key')
        work = path.with_suffix('.episode')
        _, calls = read_trace(work / 'trace')
        origin = load_json(work / 'controller_origin.json')
        rows[expected['scene_key']] = (verified, calls[1][0]['obs'][0], origin)
    assert len(rows) == 40
    return identity, rows, sha256_file(folder / 'complete.json')


def same_initial_input(first, second):
    assert set(first) == set(second), 'initial input keys'
    for key in first:
        a, b = first[key], second[key]
        if isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape, key
            assert a.tobytes() == b.tobytes(), key
        else:
            assert a == b, key


def summarize(rows):
    count = len(rows)
    assert count > 0
    n = sum(row['native']['success'] for row in rows)
    f = sum(row['fp8']['success'] for row in rows)
    return {'pairs': count, 'native_success': n, 'fp8_success': f,
            'observed_delta_percentage_points': 100 * (f - n) / count,
            'native_only': sum(row['native']['success'] and not row['fp8']['success'] for row in rows),
            'fp8_only': sum(row['fp8']['success'] and not row['native']['success'] for row in rows)}


def compare():
    native, n, nh = collect('native')
    fp8, f, fh = collect('fp8')
    match_identities(native, fp8)
    assert set(n) == set(f)
    rows = []
    for key in n:
        nr, no, origin_n = n[key]
        fr, fo, origin_f = f[key]
        same_initial_input(no, fo)
        assert origin_n == origin_f, (key, 'controller origin')
        rows.append({'scene_key': key, 'native': nr, 'fp8': fr})
    groups = {}
    for suite in ('robotwin50_easy', 'robotwin50_hard'):
        subset = [r for r in rows if r['scene_key'].split('/')[0] == suite]
        assert len(subset) == 20
        groups[suite] = summarize(subset)
    return {'ok': True, 'pairs': 40, 'groups': groups, 'rows': rows,
            'recovery': load_json(root / 'robotwin-fp8-recovery-v1/complete.json')['recovery'],
            'verifier_amendment': load_json(root / 'robotwin-verifier-amendment.json'),
            'native_complete_sha256': nh, 'fp8_complete_sha256': fh,
            'scope': '10 tasks x 2 seeds x clean/randomized; paused paired screen only, not non-inferiority or real-time qualification'}


if __name__ == '__main__':
    output = root / 'robotwin-recovery-paired-comparison.json'
    assert not output.exists(), 'preserve prior comparison'
    report = compare()
    write_json_atomic(output, report)
    print({k: v for k, v in report.items() if k != 'rows'})
