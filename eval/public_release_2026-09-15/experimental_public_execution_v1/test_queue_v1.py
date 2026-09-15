"""CPU-only synthetic contract and owned-process lifecycle tests; no model import."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('sde1_queue_tested', HERE / 'run_queue_v1.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
RECIPES = queue.read(Path(__file__).parents[3] / 'examples/cosmos3_sde1/cosmos3_sde1/data/recipes.json')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def base_config():
    return {
        'schema': 'instinctflash.experimental_public_execution.v1', 'python': queue.PYTHON,
        'recipes': list(queue.IDS), 'reclaim_owned_edge_preparations': True,
        'minimum_tmpfs_free_bytes': 12 * queue.GIB, 'minimum_disk_free_bytes': queue.GIB,
        'base': '/home/guanming/ifl_public_qualification_20260915_v1',
        'run_root': '/home/guanming/ifl_public_qualification_20260915_v1/experimental_public_execution_v1',
        'edge_tmpfs_root': '/dev/shm/ifl_public_sde1_execution_20260915_v1',
        'nano_base': '/home/guanming/ifl_eval/cosmos_distill_20260912/nano-original-native-v1',
        'overlay_root': '/dev/shm/ifl_public_sde1_assets_v1', 'overlays': copy.deepcopy(queue.OVERLAYS),
        'edge_preparation_bytes': 9171985492,
        'nvidia_smi': '/usr/sbin/nvidia-smi',
        'activations': {'edge': [{}, {}, {}], 'nano': [{}, {}, {}]},
        'timeouts': {'base_view': 1800, 'prepare': 1800, 'run': 3900},
        'base_view_helper': {'path': '/remote_control/prepare_experimental_thor_base_view_v1.py'},
        'installed_sources': {x: {'placeholder.py': '0' * 64} for x in
                              ['instinctflash', 'cosmos3-iwm', 'instinctflash-cosmos3-sde1', 'cosmos-framework']},
        'template_only': True,
    }


def fake_run(tmp_path, name='edge-seed12031'):
    run, prepared = tmp_path / 'run', tmp_path / 'prepared'
    run.mkdir()
    prepared.mkdir()
    family = RECIPES['recipes'][name]['family']
    warm, measured = (6, 10) if family == 'edge' else (12, 24)
    count = warm + measured
    source_root = tmp_path / 'site'
    source_map = {}
    for relative in ['cosmos3_sde1/__main__.py', 'cosmos3_sde1/benchmark_' + family + '.py',
                     'benchmarks/regression/fixtures/recorded_inputs_v1.npz']:
        p = source_root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'CPU fixture only')
        source_map[str(p)] = queue.sha(p)
    fixture = str(source_root / 'benchmarks/regression/fixtures/recorded_inputs_v1.npz')
    command = [queue.PYTHON, '-I', '-B', '-m', 'cosmos3_sde1.benchmark_' + family,
               str(prepared), str(run / 'receipt.json'), '--attention', 'cudnn', '--allow-unqualified',
               '--fixture', fixture]
    if family == 'nano':
        command += ['--cache-mode', 'reuse', '--swiglu']
    invocation = {'command': command, 'plan': {'recipe': name},
                  'source_sha256': source_map[str(source_root / 'cosmos3_sde1/__main__.py')]}
    np.savez_compressed(run / 'receipt.npz', actions=np.zeros((count, 32, 8), dtype=np.float32))
    calls = [{'i': i, 'phase': 'warmup' if i < warm else 'measured', 'ms': 100.,
              'graph_counters': {'captures': 36, 'checks': 72, 'replays': 36 * i}}
             for i in range(count)]
    receipt = {
        'ok': True, 'category': 'SCREEN', 'task_quality_certified': False,
        'checkpoint': str(prepared), 'attention': 'cudnn', 'qualification_only': True,
        'fixture_sha256': source_map[fixture],
        'declared_execution': {'sigmas': [1., 0.], 'steps': 1, 'guidance': 1.,
                               'branches_per_callback': 1, 'action_padding': 'zero'},
        'first_request_native_branch_clocks': [1000.], 'calls': calls,
        'actions_sha256': queue.sha(run / 'receipt.npz'),
        'backend_stats': {'velocity_evaluations': count, 'sampler_calls': count,
                          'action_padding': 'zero', 'padding_projection_calls': count,
                          'padding_projected_velocity_branches': count, 'padding_hooks_restored': True,
                          'numeric_attention': {'eligible_python_calls': 1}},
        'external_runtime_assets': [RECIPES['recipes'][name]['auxiliary']],
        'sources': {str(source_root / 'cosmos3_sde1/__main__.py'): source_map[str(source_root / 'cosmos3_sde1/__main__.py')]},
        'benchmark_sha256': source_map[str(source_root / ('cosmos3_sde1/benchmark_' + family + '.py'))],
        'p50_ms': 100., 'trained_student': False, 'swiglu_requested': True, 'cache_mode': 'reuse',
        'format_prompt_as_json': False,
    }
    completion = {'status': 'passed', 'returncode': 0, 'category': 'SCREEN',
                  'task_quality_certified': False, 'p50_ms': 100.}
    for filename, data in [('completion.json', completion), ('receipt.json', receipt), ('invocation.json', invocation)]:
        put(run / filename, data)
    return run, prepared, source_map


def test_exact_sequential_plan_and_public_cli(tmp_path):
    config = base_config()
    put(tmp_path / 'config.json', config)
    actual = queue.commands(queue.load_config(tmp_path / 'config.json'))
    assert [x['id'] for x in actual] == ['base-view', 'edge-seed12031-prepare', 'edge-seed12031-run',
                                        'edge-seed12032-prepare', 'edge-seed12032-run',
                                        'nano-original-prepare', 'nano-original-run']
    assert all(x['argv'][:3] == [queue.PYTHON, '-I', '-B'] for x in actual)
    assert '--link-unmodified' not in actual[1]['argv'] and '--link-unmodified' not in actual[3]['argv']
    assert actual[5]['argv'][-1] == '--link-unmodified'
    assert all('fetch' not in x['argv'] for x in actual)
    with pytest.raises(ValueError, match='template'):
        queue.load_config(tmp_path / 'config.json', execute=True)


@pytest.mark.parametrize('name', queue.IDS)
def test_valid_public_screen_field_contract(tmp_path, name):
    run, prepared, source_map = fake_run(tmp_path, name)
    result = queue.validate_run(name, run, prepared, RECIPES, source_map)
    assert result['requests'] == (36 if name == 'nano-original' else 16)
    assert result['category'] == 'SCREEN' and result['task_quality_certified'] is False


@pytest.mark.parametrize('mutation', ['grid', 'padding', 'clock', 'count', 'median', 'source', 'process',
                                     'command', 'fixture', 'promotion'])
def test_rejects_false_public_success(tmp_path, mutation):
    run, prepared, source_map = fake_run(tmp_path)
    receipt = queue.read(run / 'receipt.json')
    if mutation == 'grid':
        receipt['declared_execution']['sigmas'] = [1., .5, 0.]
    elif mutation == 'padding':
        receipt['backend_stats']['padding_projection_calls'] -= 1
    elif mutation == 'clock':
        receipt['first_request_native_branch_clocks'] = [999.]
    elif mutation == 'count':
        receipt['calls'].pop()
    elif mutation == 'median':
        receipt['p50_ms'] = 99.
    elif mutation == 'source':
        receipt['sources']['/unbound/private.py'] = 'a' * 64
    elif mutation == 'process':
        completion = queue.read(run / 'completion.json')
        completion['returncode'] = -15
        put(run / 'completion.json', completion)
    elif mutation == 'command':
        invocation = queue.read(run / 'invocation.json')
        invocation['command'].remove('-I')
        put(run / 'invocation.json', invocation)
    elif mutation == 'fixture':
        receipt['fixture_sha256'] = '0' * 64
    elif mutation == 'promotion':
        receipt['task_quality_certified'] = True
    put(run / 'receipt.json', receipt)
    with pytest.raises(ValueError):
        queue.validate_run(queue.IDS[0], run, prepared, RECIPES, source_map)


def test_nonfinite_actions_rejected_even_with_matching_archive_hash(tmp_path):
    run, prepared, source_map = fake_run(tmp_path)
    values = np.zeros((16, 32, 8), dtype=np.float32)
    values[5, 7, 0] = np.nan
    np.savez_compressed(run / 'receipt.npz', actions=values)
    receipt = queue.read(run / 'receipt.json')
    receipt['actions_sha256'] = queue.sha(run / 'receipt.npz')
    put(run / 'receipt.json', receipt)
    with pytest.raises(ValueError, match='finite full-action'):
        queue.validate_run(queue.IDS[0], run, prepared, RECIPES, source_map)


def test_nano_measured_graph_rebuild_rejected(tmp_path):
    run, prepared, source_map = fake_run(tmp_path, 'nano-original')
    receipt = queue.read(run / 'receipt.json')
    receipt['calls'][12]['graph_counters']['captures'] += 1
    put(run / 'receipt.json', receipt)
    with pytest.raises(ValueError, match='Nano timed'):
        queue.validate_run('nano-original', run, prepared, RECIPES, source_map)


def test_real_timeout_cleans_only_owned_process_group(tmp_path):
    stage = {'argv': [sys.executable, '-c', 'import time;time.sleep(60)'], 'timeout': .15}
    with pytest.raises(subprocess.TimeoutExpired):
        queue.run_process(stage, dict(__import__('os').environ), str(tmp_path), tmp_path / 'worker.log')
    record = queue.read(tmp_path / 'worker.process.json')
    assert record['failed'] is True and record['returncode'] != 0
    assert queue.group_members(record['process_group']) == []


def test_real_parent_success_with_live_descendant_is_rejected_and_cleaned(tmp_path):
    script = 'import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])'
    stage = {'argv': [sys.executable, '-c', script], 'timeout': 5}
    with pytest.raises(ValueError, match='live descendant'):
        queue.run_process(stage, dict(__import__('os').environ), str(tmp_path), tmp_path / 'worker.log')
    record = queue.read(tmp_path / 'worker.process.json')
    assert record['failed'] is True and queue.group_members(record['process_group']) == []


def owned_prepared(tmp_path):
    tmp = tmp_path / 'owned'
    tmp.mkdir(mode=0o700)
    prepared = tmp / queue.IDS[0]
    prepared.mkdir()
    (prepared / 'weight.safetensors').write_bytes(b'NEW synthetic weight')
    (prepared / 'config.json').write_text('{}')
    rows = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': queue.sha(p)} for p in prepared.iterdir()]
    declaration = {'files': rows, 'recipe': queue.IDS[0], 'category': 'SCREEN', 'task_quality_certified': False,
                   'link_unmodified': False, 'recipe_sha256': hashlib.sha256(json.dumps(
                       RECIPES['recipes'][queue.IDS[0]], sort_keys=True).encode()).hexdigest()}
    put(prepared / 'public_preparation.json', declaration)
    put(prepared / 'instinctcompress_manifest.json', {})
    run = tmp_path / 'run'
    put(run / 'completion.json', {'status': 'passed'})
    put(run / 'receipt.json', {'synthetic': True})
    archive = tmp_path / 'archive'
    queue.archive_small(prepared, run, archive)
    put(archive / 'validation.json', {'status': 'passed', 'recipe': queue.IDS[0]})
    ownership = {'config_sha256': '1' * 64}
    put(tmp / 'ownership.json', ownership)
    config = {'edge_tmpfs_root': str(tmp), '_config_sha256': '1' * 64, 'reclaim_owned_edge_preparations': True}
    return config, prepared, archive, ownership


def test_real_archive_then_only_owned_copy_is_reclaimed(tmp_path, monkeypatch):
    config, prepared, archive, ownership = owned_prepared(tmp_path)
    original = tmp_path / 'original.safetensors'
    original.write_bytes(b'ORIGINAL MUST REMAIN')
    # Local tests do not claim handle visibility for unrelated CI processes.
    monkeypatch.setattr(queue, 'same_uid_handles', lambda _: {'test_scope': 'mocked_no_handles'})
    assert queue.validate_preparation(queue.IDS[0], prepared, RECIPES)['recipe'] == queue.IDS[0]
    result = queue.reclaim_edge(config, queue.IDS[0], prepared, archive, ownership)
    assert result['status'] == 'owned_preparation_reclaimed'
    assert original.read_bytes() == b'ORIGINAL MUST REMAIN' and not prepared.exists()
    assert (archive / 'preparation/public_preparation.json').is_file()
    assert not (archive / 'preparation/weight.safetensors').exists()


@pytest.mark.parametrize('failure', ['archive_mutated', 'source_metadata_mutated', 'linked_weight',
                                    'extra_file', 'wrong_name', 'missing_validation'])
def test_cleanup_rejects_uncertain_staging(tmp_path, monkeypatch, failure):
    config, prepared, archive, ownership = owned_prepared(tmp_path)
    monkeypatch.setattr(queue, 'same_uid_handles', lambda _: {})
    name = queue.IDS[0]
    if failure == 'archive_mutated':
        (archive / 'run/receipt.json').write_text('corrupt')
    elif failure == 'source_metadata_mutated':
        (prepared / 'config.json').write_text('{"new":true}')
    elif failure == 'linked_weight':
        __import__('os').link(prepared / 'weight.safetensors', tmp_path / 'original.safetensors')
    elif failure == 'extra_file':
        (prepared / 'unexpected').touch()
    elif failure == 'wrong_name':
        name = 'nano-original'
    elif failure == 'missing_validation':
        (archive / 'validation.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        queue.reclaim_edge(config, name, prepared, archive, ownership)
    assert prepared.is_dir()


def test_capacity_reserves_space_for_completed_edge_preparation(monkeypatch):
    config = base_config()
    config['nvidia_smi'] = 'never_executed'
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **k: '')
    class Stat:
        f_bavail = 21 * queue.GIB
        f_frsize = 1
    monkeypatch.setattr(queue.os, 'statvfs', lambda _: Stat())
    stage = queue.commands(config)[1]
    record = queue.idle_and_capacity(config, stage)
    assert record['required_tmpfs_bytes'] == 12 * queue.GIB + 9171985492
    Stat.f_bavail = 20 * queue.GIB
    with pytest.raises(ValueError, match='insufficient free'):
        queue.idle_and_capacity(config, stage)


def test_activation_is_parsed_and_private_overrides_removed(tmp_path, monkeypatch):
    config = base_config()
    native_python = tmp_path / 'python'
    native_python.touch()
    files = [tmp_path / 'vendor.env', tmp_path / 'tools.env', tmp_path / 'assets.env']
    files[0].write_text('export IFL_BF16_KERNEL_LIBRARY=/old/private.so\nexport PYTHONPATH=/old/vendor\n')
    files[1].write_text(f'export UV_PYTHON={native_python}\nexport UV_OFFLINE=1\nexport UV_PYTHON_DOWNLOADS=never\nexport PATH=/public/bin:"$PATH"\n')
    files[2].write_text('export HF_HOME=/qualified/assets\n')
    config['activations']['edge'] = [queue.ref(p) for p in files]
    config['ptxas'] = {'path': '/usr/local/cuda-13.2/bin/ptxas'}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    env = queue.environment(config, 'edge')
    assert env['UV_PYTHON'] == str(native_python)
    assert env['VIRTUAL_ENV'] == queue.ENV_ROOT and env['HF_HOME'] == '/qualified/assets'
    assert env['TRITON_PTXAS_PATH'] == env['TRITON_PTXAS_BLACKWELL_PATH'] == config['ptxas']['path']
    assert not any(k.startswith('IFL_') for k in env) and 'CUDA_VISIBLE_DEVICES' not in env and 'PYTHONPATH' not in env


def test_actual_same_uid_open_handle_blocks_reclamation_scope(tmp_path, monkeypatch):
    import os
    import types
    real_path = Path
    owned = tmp_path / 'owned'
    owned.mkdir()
    file = owned / 'open.bin'
    file.write_bytes(b'protected')
    own_proc = real_path('/proc') / str(os.getpid())
    # Exercise real /proc descriptors for our process, without requiring global
    # permissions for unrelated test-runner processes.
    monkeypatch.setattr(queue, 'Path', lambda *args: types.SimpleNamespace(iterdir=lambda: [own_proc])
                        if args == ('/proc',) else real_path(*args))
    with file.open('rb'):
        with pytest.raises(ValueError, match='live same-UID handle'):
            queue.same_uid_handles(owned)
    result = queue.same_uid_handles(owned)
    assert result['checked_pids'] == [os.getpid()] and result['global_handle_absence_claimed'] is False
