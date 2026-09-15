"""Export validated controls and compare full inputs to every applicable student."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from run_control import digest
from validate_control import validate


def compare_inputs(control, student, *, native_padding, offset):
    for key in ('reference', 'measured_action', 'model_measured_action'):
        assert control[key].shape == student[key].shape
        assert control[key].tobytes() == student[key].tobytes(), key
    for key in ('initial_noise', 'preserve_mask'):
        assert control[key].shape == student[key].shape
        if native_padding:
            # Only this predeclared intervention differs; never suppress video
            # coordinates, active action channels, or arbitrary large deltas.
            selected = np.ones(control[key].shape, dtype=bool)
            assert offset == selected.shape[1] - 33 * 64
            selected[:, offset:].reshape(1, 33, 64)[:, :, 8:] = False
            assert control[key][selected].tobytes() == student[key][selected].tobytes(), key
        else:
            assert control[key].tobytes() == student[key].tobytes(), key


def export(directory, output, source, student_source, study):
    directory, output, source, student_source, study = map(Path,
        (directory, output, source, student_source, study))
    if output.exists():
        raise FileExistsError(output)
    validation = validate(directory, source)
    config = validation['configuration']
    original = json.loads((directory / 'report.json').read_text())['configuration']
    native_padding = original['action_padding'] == 'native'
    cfgs = (1, 4) if native_padding else (int(original['guidance']),)
    spec = importlib.util.spec_from_file_location('student_capture_validation', student_source / 'validate_capture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    students = {}
    for cfg in cfgs:
        for seed in (12031, 12032):
            bank = study / f'quality-v16-cfg{cfg}-seed{seed}-native-v1'
            students[str(bank)] = module.validate(bank, student_source)
    actions = ('action', 'model_action', 'measured_action', 'model_measured_action')
    compact = {key: [] for key in actions}
    rows = []
    for index in range(128):
        meta = json.loads((directory / f'{index:04d}.json').read_text())
        paired_request = {key: value for key, value in meta['request'].items() if key != 'configuration_id'}
        with np.load(directory / f'{index:04d}.npz', allow_pickle=False) as control:
            for bank in students:
                student_meta = json.loads((Path(bank) / f'{index:04d}.json').read_text())
                assert {k: v for k, v in student_meta['request'].items() if k != 'configuration_id'} == paired_request
                for key in ('normalizer', 'action_offset', 'action_shape'):
                    assert meta['trace'][key] == student_meta['trace'][key], (index, key)
                with np.load(Path(bank) / f'{index:04d}.npz', allow_pickle=False) as student:
                    compare_inputs(control, student, native_padding=native_padding, offset=meta['trace']['action_offset'])
            for key in actions:
                compact[key].append(control[key].copy())
        rows.append(dict(request=meta['request'], source_arrays_sha256=meta['arrays_sha256']))
    assert digest(directory / 'report.json') == validation['report_sha256']
    for bank, record in students.items():
        assert digest(Path(bank) / 'report.json') == record['report_sha256']
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'actions.npz', **{k: np.stack(v) for k, v in compact.items()})
    report = dict(status='success', quality_certified=False, configuration=config, requests=rows,
        capture_validation=validation, paired_students=students,
        input_pairing='All reference/target bytes identical; noise/mask identical except explicit action-padding coordinates'
            if native_padding else 'All reference/noise/mask/target bytes identical',
        exclusion=None if not native_padding else dict(action_rows=33, columns=[8, 64], reason='Native versus zero action padding'),
        archive=dict(file='actions.npz', sha256=digest(output / 'actions.npz'), shape=[128, 32, 8]),
        sources={name: digest(Path(__file__).with_name(name)) for name in ('export_control.py', 'validate_control.py', 'run_control.py')},
        student_validator_sha256=digest(student_source / 'validate_capture.py'),
        scope='Same-Thor historical input pairing and compact original control export; no quality admission')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--student-source', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.directory, args.output, args.source, args.student_source, args.study)
    print(json.dumps(dict(status=result['status'], configuration=result['configuration'],
                         requests=len(result['requests']), paired_students=len(result['paired_students']))))
