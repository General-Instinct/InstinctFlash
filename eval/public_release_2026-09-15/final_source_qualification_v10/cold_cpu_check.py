from __future__ import annotations
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

stage = Path(__file__).resolve().parent
manifest_path = stage / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
blocked = []
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'diffusers', 'safetensors'}:
            blocked.append(fullname)
            raise RuntimeError('model import prohibited in CPU metadata qualification: ' + fullname)
sys.meta_path.insert(0, NoModels())
def guard(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.posix_spawnp'}:
        blocked.append(event)
        raise RuntimeError('network/process action prohibited in CPU metadata qualification')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = str(args[0]).lower()
        if name.endswith(('.safetensors', '.ckpt', '.pth', '.pt', '.bin')):
            blocked.append(name)
            raise RuntimeError('weight access prohibited in CPU metadata qualification')
sys.addaudithook(guard)
assert sys.flags.isolated and sys.dont_write_bytecode
site_root = Path(sys.prefix).resolve()
assert not any(Path(p).resolve() == Path(manifest['repository']).resolve() for p in sys.path if p)
packages = {}
for distribution, (directory, module) in manifest['packages'].items():
    info = importlib.metadata.distribution(distribution)
    imported = importlib.import_module(module)
    origin = Path(imported.__file__).resolve()
    assert origin.is_relative_to(site_root), origin
    assert not json.loads(info.read_text('direct_url.json') or '{}').get('dir_info', {}).get('editable')
    packages[distribution] = {'version': info.version, 'origin': str(origin)}
from instinctflash.runtime.loader import available_models, discover_plugins
assert discover_plugins() == []
expected_backbones = {'wan_va', 'pi05', 'lingbot_vla', 'lingbot_vla_v2', 'groot_n17', 'cosmos3_policy', 'dreamzero'}
registered = available_models()
assert expected_backbones <= set(registered), registered
points = []
for group in ('instinctflash.adapters', 'instinctflash.passes'):
    for point in importlib.metadata.entry_points(group=group):
        point.load()
        points.append({'group': group, 'name': point.name, 'value': point.value})
assert len([point for point in points if point['group'] == 'instinctflash.adapters']) == 6
installed_files = {}
for distribution, (directory, module) in manifest['packages'].items():
    info = importlib.metadata.distribution(distribution)
    for relative, entry in manifest['files'].items():
        if directory == '.':
            selected = relative.startswith(('instinctflash/', 'benchmarks/'))
            inside = relative
        else:
            modules = (module, 'instinct_compress') if distribution == 'instinctflash-cosmos3-sde1' else (module,)
            selected = any(relative.startswith(directory + '/' + owner + '/') for owner in modules)
            inside = relative[len(directory) + 1:]
        if selected and entry.get('wheel_required', True):
            installed = Path(info.locate_file(inside))
            assert installed.is_file() and not installed.is_symlink(), inside
            assert digest(installed) == entry['sha256'], inside
            installed_files[relative] = entry['sha256']
core = importlib.metadata.distribution('instinctflash')
for included in ('instinctflash/train', 'instinctflash/distill'):
    assert Path(core.locate_file(included)).is_dir(), included
for held in ('serving', 'eval'):
    assert not Path(core.locate_file(held)).exists(), held
assert not any(Path(core.locate_file('instinctflash/native')).glob('*.so'))
assert 'torch' not in sys.modules and importlib.machinery.PathFinder.find_spec('torch') is None
from benchmarks.regression import reproduce, serve_smoke
assert reproduce.profiles_path().resolve().is_relative_to(site_root)
assert digest(reproduce.profiles_path()) == manifest['files']['release/deployment_profiles.json']['sha256']
assert digest(reproduce.fixture_path()) == reproduce.FIXTURE_SHA
plans = []
for profile in json.loads(reproduce.profiles_path().read_text())['models']:
    for mode in profile['execution_modes']:
        plan = reproduce.make_plan(profile['id'], mode)
        assert plan['publication_ready'] is False and plan['task_quality_validated'] is False
        assert serve_smoke.selected_cell(plan)['arm'] in {'runtime_selected','operating_point'}
        plans.append([profile['id'],mode,len(plan['matrix']['cells'])])
assert len(plans) == 20
extra = reproduce.make_plan('va', 'fp8', extra_modes=('2v4a-fp8',))
assert len(extra['matrix']['cells']) == 4
from benchmarks.regression import framework_compare, screen_replay
framework_plans = [framework_compare.plan(cell['id']) for cell in framework_compare.catalog()['cells']]
assert len(framework_plans) == 6 and all(p['task_quality_certified'] is False for p in framework_plans)
assert sum(screen_replay.CELLS.values()) == 316
from cosmos3_sde1 import __main__ as experimental
experimental_names = list(json.loads((experimental.DATA / 'recipes.json').read_text())['recipes'])
assert len(experimental_names) == 3
assert all(experimental.recipe(name)['task_quality_certified'] is False for name in experimental_names)
import instinct_compress
assert Path(instinct_compress.__file__).resolve().is_relative_to(site_root)
assert len(list(Path(instinct_compress.__file__).parent.rglob('*.py'))) == 8
attribution = json.loads(Path(core.locate_file('benchmarks/regression/fixtures/recorded_inputs_v1_attribution.json')).read_text())
assert attribution['fixture']['sha256'] == reproduce.FIXTURE_SHA
assert digest(Path(core.locate_file('benchmarks/regression/fixtures/LICENSE.RoboTwin.txt'))) == attribution['notice']['sha256']
assert not blocked, blocked
result = {'schema': 'instinctflash.public_release_cold_cpu.v1', 'ok': True,
          'scope': 'Nine wheel installs, imports, adapter/pass entrypoints, 20 installed reproduction plans, VA four-cell extra-mode plan, six framework plans, three experimental recipes, generic SCREEN replay import, dataset notice and exact selected file bytes; no Torch/vendor libraries or inference',
          'manifest_sha256': digest(manifest_path), 'script_sha256': digest(Path(__file__)),
          'python': sys.version, 'executable': sys.executable, 'isolated': True,
          'installed_distributions': packages, 'registered_backbones_and_aliases': registered,
          'entrypoints': points, 'installed_reproduction_plans': plans, 'framework_plan_ids': [p['cell']['id'] for p in framework_plans], 'experimental_recipes': experimental_names, 'installed_selected_files': installed_files,
          'blocked_operations': blocked, 'torch_installed': False, 'torch_imported': False,
          'publication_ready': False, 'pipeline_validated': False,
          'limitations': ['Only NumPy and PyYAML were added from the offline cache for FlashRT import/config checks; no Torch or upstream vendor stack was installed.',
                          'No checkpoint download, model loading, GPU probing, inference or full reproduction pipeline was attempted.']}
with (stage / 'cold_cpu_validation.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'ok': True, 'packages': len(packages), 'entrypoints': len(points), 'selected_files': len(installed_files), 'torch_imported': False}))
