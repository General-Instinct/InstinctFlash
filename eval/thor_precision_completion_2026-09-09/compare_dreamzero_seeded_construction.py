"""Compare two completed fresh-process diagnostics of the SAME precision.

Reports observed state/output differences; does not certify task quality or
all-input equivalence. No report is produced for missing or corrupted evidence.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_run(directory, precision):
    directory = Path(directory)
    stem = f'dreamzero-{precision}-seeded_construction'
    receipt_path = directory / f'{stem}.json'
    report = json.loads(receipt_path.read_text())
    require(report['ok'] and report['precision'] == precision, 'Incomplete/wrong precision')
    for suffix, key in (('-state.json', 'state_sha256'), ('-actions.npz', 'actions_sha256')):
        require(sha(directory / f'{stem}{suffix}') == report[key], f'Corrupted {key}')
    require(sha(directory / 'source.json') == report['source_sha256'], 'Corrupted sources')
    sources = json.loads((directory / 'source.json').read_text())
    require(bool(sources), 'Empty source manifest')
    state = json.loads((directory / f'{stem}-state.json').read_text())
    require(state['construction_seed'] == report['construction_seed'], 'Seed mismatch')
    require(bool(state['model_state']), 'Empty persistent state')
    with np.load(directory / f'{stem}-actions.npz', allow_pickle=False) as data:
        actions = data['actions'].copy()
    require(actions.shape == (6, 24, 8) and np.isfinite(actions).all(), 'Invalid actions')
    require(actions[:3].tobytes() == actions[3:].tobytes(), 'Episode reset mismatch')
    require(report['frame_positions'] == [3, 5, 7, 3, 5, 7], 'History mismatch')
    require(report['scheduler_steps'] == 16 and sum(report['dit_step_mask']) == 8, 'Schedule mismatch')
    require(report['load_receipt']['verified_dit_tensors'] == 1317, 'Missing load verification')
    counts = report['executed_fp8_projections']
    require((len(counts) == 120 and min(counts.values()) > 0) if precision == 'fp8' else not counts,
            'Precision execution mismatch')
    return report, state, actions, sha(receipt_path)


def compare(left, right, precision):
    require(Path(left).resolve() != Path(right).resolve(), 'Same directory is not a repeat')
    a, sa, aa, ha = read_run(left, precision)
    b, sb, ab, hb = read_run(right, precision)
    require(a['run_id'] != b['run_id'] and a['pid'] != b['pid'], 'Need distinct process runs')
    for key in ('precision', 'checkpoint', 'construction_seed', 'probe_sha256', 'input_sha256',
                'boot_id', 'torch_version', 'device_name', 'matmul_tf32', 'cudnn_tf32',
                'cudnn_benchmark', 'source_sha256', 'scheduler_steps', 'dit_step_mask'):
        require(a[key] == b[key], f'Unmatched {key}')
    require(sa['construction_rng'] == sb['construction_rng'], 'Unmatched construction RNG')
    require(sa['construction_rng']['cuda'], 'Missing CUDA RNG')
    require(aa.dtype == ab.dtype, 'Action dtype differs')
    names = set(sa['model_state']) | set(sb['model_state'])
    changed = sorted(name for name in names if sa['model_state'].get(name) != sb['model_state'].get(name))
    difference = np.abs(aa.astype(np.float64) - ab.astype(np.float64))
    return dict(precision=precision, receipts=[ha, hb], persistent_state_differences=changed,
                action_byte_equal=aa.tobytes() == ab.tobytes(),
                action_mae=float(difference.mean()), action_max_abs=float(difference.max()),
                scope='Two fresh processes with matched construction RNG and Python sources; '
                      'persistent state and six recorded-input chunks only. Does not cover '
                      'nonpersistent buffers, native binaries, task quality or all-input equivalence.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('precision', choices=('native', 'fp8'))
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('output')
    args = parser.parse_args()
    result = compare(args.left, args.right, args.precision)
    with open(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))
