"""Prepare original-weight Nano SDE1/SDE2 cost artifacts without modifying the source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def prepare(source, inventory, output, steps):
    assert type(steps) is int and steps in (1, 2)
    source, inventory, output = map(Path, (source, inventory, output))
    expected = json.loads(inventory.read_text())
    if output.exists():
        raise FileExistsError(output)
    assert len(expected['files']) == 43
    for item in expected['files']:
        relative = Path(item['relative_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        original = source / relative
        assert original.stat().st_size == item['bytes'] and digest(original) == item['sha256'], original
    config = json.loads((source / 'config.json').read_text())
    assert config['model']['config'].get('fixed_step_sampler_config') is None
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in expected['files']:
        relative = item['relative_path']
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == 'config.json':
            shutil.copyfile(source / relative, destination)
        else:
            try:
                os.link((source / relative).resolve(), destination)
            except OSError:
                shutil.copyfile(source / relative, destination)
        rows.append(dict(item))
    sampler = dict(_type='fixed_step_sampler_config', sample_type='sde', t_list=[1. - i / steps for i in range(steps)])
    config['model']['config']['fixed_step_sampler_config'] = sampler
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for item in rows:
        if item['relative_path'] == 'config.json':
            item.update(bytes=(output / 'config.json').stat().st_size, sha256=digest(output / 'config.json'))
    for filename, template in (('instinctflash.json', 'declaration.json'),
                               ('instinctcompress_action_padding.json', 'padding.json')):
        destination = output / filename
        if filename == 'instinctflash.json':
            declaration = json.loads(Path(__file__).with_name(template).read_text())
            declaration['execution']['nfe']['action'] = steps
            declaration['execution']['sampling']['sigmas'] = [1. - i / steps for i in range(steps + 1)]
            destination.write_text(json.dumps(declaration, indent=2) + '\n')
        else:
            shutil.copyfile(Path(__file__).with_name(template), destination)
        rows.append(dict(relative_path=filename, bytes=destination.stat().st_size, sha256=digest(destination)))
    manifest = dict(kind='untrained_original_nano_cost_only', steps=steps, files=rows,
        original_revision='2b9f9517efcfbf26e222945b386ae9b65c0930ac', trained_student=False, quality_certified=False,
        native_config_delta={'model.config.fixed_step_sampler_config': dict(before=None, after=sampler)})
    (output / 'budget_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for item in expected['files']:
        assert digest(source / item['relative_path']) == item['sha256']
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.inventory, args.output, args.steps)
    print(json.dumps(dict(files=len(result['files']), trained_student=False, quality_certified=False)))
