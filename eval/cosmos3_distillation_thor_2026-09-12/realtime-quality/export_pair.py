"""Validate paired deployed inputs and export compact action banks for scoring."""
import argparse
import json
from pathlib import Path

import numpy as np

from validate_capture import digest, validate


def export_pair(native, cudnn, output, source):
    native, cudnn, output, source = map(Path, (native, cudnn, output, source))
    if output.exists():
        raise FileExistsError(output)
    validations = [validate(directory, source) for directory in (native, cudnn)]
    assert validations[0]['attention'] == 'native'
    assert validations[1]['attention'] == 'cudnn'
    assert validations[0]['configuration_id'] == validations[1]['configuration_id']
    inputs = ('reference', 'preserve_mask', 'initial_noise',
              'measured_action', 'model_measured_action')
    actions = ('action', 'model_action', 'measured_action', 'model_measured_action')
    banks = [{key: [] for key in actions} for _ in range(2)]
    records = []
    maxima = {key: 0.0 for key in ('action', 'model_action')}
    for index in range(128):
        metadata = [json.loads((directory / f'{index:04d}.json').read_text())
                    for directory in (native, cudnn)]
        assert metadata[0]['request'] == metadata[1]['request']
        for key in ('normalizer', 'effective_sampler_arguments', 'action_offset', 'action_shape'):
            assert metadata[0]['trace'][key] == metadata[1]['trace'][key], (index, key)
        # Compare actual effective arrays, not the earlier preparation-mask trace.
        with np.load(native / f'{index:04d}.npz', allow_pickle=False) as left, \
                np.load(cudnn / f'{index:04d}.npz', allow_pickle=False) as right:
            for key in inputs:
                assert left[key].shape == right[key].shape
                assert left[key].tobytes() == right[key].tobytes(), (index, key)
            for key in maxima:
                maxima[key] = max(maxima[key], float(np.abs(left[key] - right[key]).max()))
            for bank, arrays in zip(banks, (left, right), strict=True):
                for key in actions:
                    bank[key].append(arrays[key].copy())
        records.append(dict(index=index, request=metadata[0]['request'],
                            source_arrays_sha256=[m['arrays_sha256'] for m in metadata]))
    # Recheck immutable report identities after reading their referenced arrays.
    for directory, validation in zip((native, cudnn), validations, strict=True):
        assert digest(directory / 'report.json') == validation['report_sha256']
    output.mkdir(parents=True, exist_ok=False)
    archives = {}
    for variant, bank in zip(('native', 'cudnn'), banks, strict=True):
        path = output / f'{variant}.npz'
        np.savez_compressed(path, **{key: np.stack(value) for key, value in bank.items()})
        archives[variant] = dict(file=path.name, sha256=digest(path), shape=[128, 32, 8])
    report = dict(status='success', quality_certified=False,
                  scope='Paired input identity and compact action export; no quality score or realtime certificate',
                  configuration_id=validations[0]['configuration_id'], requests=records,
                  identical_input_fields=list(inputs), action_maxabs=maxima,
                  source_validations=validations, archives=archives,
                  script_sha256=digest(__file__), validator_sha256=digest(Path(__file__).with_name('validate_capture.py')),
                  protocol_sha256=digest(source / 'protocol.json'), plan_sha256=digest(source / 'plan.json'))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path)
    parser.add_argument('cudnn', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    result = export_pair(args.native, args.cudnn, args.output, args.source)
    print(json.dumps({key: value for key, value in result.items() if key != 'requests'}, indent=2))
