"""Versioned model/simulator contracts. Loading this catalog imports no model or simulator."""
from __future__ import annotations

import copy
from pathlib import Path

from .util import ConfigurationError, load_json, require_keys, require_revision, sha256_json

CATALOG = Path(__file__).with_name('config') / 'adapters.json'


def load_adapters(path=CATALOG):
    raw = load_json(Path(path))
    if raw.get('schema_version') != 1 or not isinstance(raw.get('adapters'), list):
        raise ConfigurationError('unsupported adapter catalog')
    ids, protocols = set(), set()
    for entry in raw['adapters']:
        require_keys(entry, ('id', 'protocol', 'backbone', 'driver_module', 'simulator',
                            'suite_ids', 'checkpoints', 'observations', 'actions', 'history',
                            'evaluation_mode'), 'adapter')
        for key in ('id', 'protocol', 'backbone', 'driver_module', 'simulator'):
            if not isinstance(entry[key], str) or not entry[key]:
                raise ConfigurationError(f'adapter {key} must be nonempty text')
        if entry['id'] in ids or entry['protocol'] in protocols:
            raise ConfigurationError('duplicate adapter id or protocol')
        ids.add(entry['id']); protocols.add(entry['protocol'])
        if not entry['checkpoints'] or not entry['suite_ids']:
            raise ConfigurationError('adapter needs explicit checkpoints and suites')
        for checkpoint in entry['checkpoints']:
            require_keys(checkpoint, ('id', 'revision'), 'adapter checkpoint')
            require_revision(checkpoint['revision'], 'adapter checkpoint')
        require_keys(entry['observations'], ('camera_keys', 'image_transform', 'instruction'), 'adapter observations')
        require_keys(entry['actions'], ('wire_shape', 'controller_dim', 'semantics', 'digest_encoding'), 'adapter actions')
        dims = entry['actions']['wire_shape']
        if not isinstance(dims, list) or not dims or any(type(n) is not int or n <= 0 for n in dims):
            raise ConfigurationError('adapter wire shape must contain positive integers')
        if type(entry['actions']['controller_dim']) is not int or entry['actions']['controller_dim'] <= 0:
            raise ConfigurationError('adapter controller dimension must be positive')
    return raw['adapters']


def resolve_adapter(request):
    """Return a supported contract, refusing mismatched checkpoints even within one backbone.

    Legacy suites without a bridge remain on their existing driver validation path.
    An explicitly named bridge must be registered; no automatic shape conversion.
    """
    suite = request['suite']
    protocol = suite.get('protocol', {}).get('bridge')
    if suite['kind'] != 'closed_loop' or protocol is None:
        return None
    contract = next((a for a in load_adapters() if a['protocol'] == protocol), None)
    if contract is None:
        raise ConfigurationError(f'no adapter registered for bridge {protocol!r}')
    model = request['model']
    ckpt = {k: model['checkpoint'][k] for k in ('id', 'revision')}
    if model['backbone'] != contract['backbone'] or ckpt not in contract['checkpoints']:
        raise ConfigurationError(f"adapter {contract['id']} does not support checkpoint {ckpt['id']}@{ckpt['revision']}")
    if contract.get('checkpoint_variants'):
        subdir=model['checkpoint'].get('subdir','libero_10')
        if contract['checkpoint_variants'].get(subdir)!=suite['id']:
            raise ConfigurationError('checkpoint subdirectory is incompatible with the selected task suite')
    if suite['id'] not in contract['suite_ids']:
        raise ConfigurationError(f"adapter {contract['id']} does not support suite {suite['id']}")
    if suite['protocol'].get('evaluation_mode') != contract['evaluation_mode']:
        raise ConfigurationError('adapter evaluation mode mismatch')
    return copy.deepcopy(contract)


def bind_adapter(request, driver):
    if request['suite']['kind'] == 'closed_loop' and driver.get('adapter_id'):
        selected = next((a for a in load_adapters() if a['id'] == driver['adapter_id']), None)
        if selected is None:
            raise ConfigurationError('unknown driver adapter_id')
        protocol = request['suite'].setdefault('protocol', {})
        for key, value in [('bridge', selected['protocol']), ('evaluation_mode', selected['evaluation_mode'])]:
            if key in protocol and protocol[key] != value:
                raise ConfigurationError('driver adapter conflicts with suite protocol')
            protocol[key] = value
    contract = resolve_adapter(request)
    if contract is None:
        return
    # Accept the documented module or exact local script entrypoint, not an opaque wrapper.
    command = driver.get('command', [])
    script = Path(__file__).resolve().parents[2].joinpath(*contract['driver_module'].split('.')).with_suffix('.py')
    module_entry = command[1:3] == ['-m', contract['driver_module']]
    script_entry = len(command) >= 2 and Path(command[1]).is_absolute() and Path(command[1]).resolve() == script
    if not (module_entry or script_entry):
        raise ConfigurationError(f"adapter {contract['id']} requires driver module {contract['driver_module']} or its local script")
    request['adapter'] = {'contract': contract, 'sha256': sha256_json(contract)}


def validate_bound_adapter(request):
    contract = resolve_adapter(request)
    if contract is None or request.get('adapter') != {'contract': contract, 'sha256': sha256_json(contract)}:
        raise ConfigurationError('missing or changed adapter contract; rebuild the plan')
    return contract
