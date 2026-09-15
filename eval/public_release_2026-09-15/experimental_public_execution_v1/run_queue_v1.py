"""Finite on-host operational queue for three public SDE1 SCREEN recipes.

Planning is CPU/local. Root invokes execution only after the predecessor passes,
under its exclusive /tmp/thor_gpu.lock. No SSH, downloads, retries or package edits.
Only new owned Edge tmpfs preparations are optionally reclaimed after durable
small-evidence archival. Original weights, overlays, base view, Nano and runs stay.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time

PYTHON = '/dev/shm/ifl_public_envs_20260915_v1/cosmos_pytorch_triton_v1/bin/python'
ENV_ROOT = str(Path(PYTHON).parent.parent)
IDS = ('edge-seed12031', 'edge-seed12032', 'nano-original')
OVERLAYS = {
    'edge-seed12031': {'filename': 'cosmos3-edge-sde1-cfg1-seed12031.safetensors',
                     'bytes': 419568688, 'sha256': '0d6e90101673cac8a1c4c621e31a2c1349a51515d01003903d39d3ad8ec563a2'},
    'edge-seed12032': {'filename': 'cosmos3-edge-sde1-cfg1-seed12032.safetensors',
                     'bytes': 419568688, 'sha256': 'e337921e220cca7aa4929ad34be771b74734e3e2dcd388138afed011108cb648'},
}
GIB = 1 << 30
_STOPPING = False


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(rows):
        result = {}
        for k, v in rows:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(Path(path).parent)


def ref(path):
    p = Path(path).absolute()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def checked(item):
    p = Path(item['path'])
    require(p.is_absolute() and p.is_file() and sha(p) == item['sha256'], 'bound input changed')
    require('bytes' not in item or p.stat().st_size == item['bytes'], 'bound input size changed')
    return p


def path_guard(value):
    p = Path(value)
    require(p.is_absolute() and p == p.resolve() and not any(x.isspace() for x in str(p)),
            'operational paths must be absolute, resolved and contain no whitespace')
    return p


def load_config(path, execute=False):
    c = read(path)
    require(c['schema'] == 'instinctflash.experimental_public_execution.v1' and c['python'] == PYTHON,
            'unexpected queue/interpreter')
    require(c['recipes'] == list(IDS) and type(c['reclaim_owned_edge_preparations']) is bool,
            'fixed three-recipe sequence/reclamation permission changed')
    require(c['overlays'] == OVERLAYS and c['edge_preparation_bytes'] == 9171985492,
            'exact released overlays/preparation size changed')
    require(c['nvidia_smi'] == '/usr/sbin/nvidia-smi', 'match the public benchmark GPU-contention executable')
    require(set(c['installed_sources']) == {'instinctflash', 'cosmos3-iwm', 'instinctflash-cosmos3-sde1',
                                            'cosmos-framework'} and all(c['installed_sources'].values()),
            'complete public implementation source maps are required')
    require(c['minimum_tmpfs_free_bytes'] >= 12 * GIB and c['minimum_disk_free_bytes'] >= GIB,
            'headroom floors cannot be reduced')
    root, tmp = path_guard(c['run_root']), path_guard(c['edge_tmpfs_root'])
    require(tmp.parent == Path('/dev/shm') and tmp.name.startswith('ifl_public_sde1_execution_'),
            'Edge staging must be a new explicitly owned tmpfs root')
    require(root.parent == Path(c['base']) and root.name == 'experimental_public_execution_v1',
            'new run root must be inside the approved qualification base')
    for key in ('nano_base', 'overlay_root'):
        p = path_guard(c[key])
        require(not p.is_relative_to(root) and not p.is_relative_to(tmp)
                and not root.is_relative_to(p) and not tmp.is_relative_to(p), 'original/overlay roots overlap outputs')
    require(set(c['activations']) == {'edge', 'nano'} and all(len(x) == 3 for x in c['activations'].values()),
            'vendor/native-tool/assets activation triple required')
    require(1 <= c['timeouts']['base_view'] <= 3600 and 1 <= c['timeouts']['prepare'] <= 3600
            and 1 <= c['timeouts']['run'] <= 3900, 'unbounded operation lifetime')
    if execute:
        require(c['template_only'] is False, 'template is not execution admission')
        require(c['queue_source_sha256'] == sha(__file__), 'operational queue source changed')
        checked(c['preparation_plan'])
        checked(c['base_view_helper'])
        completed = read(checked(c['predecessor_completion']))
        require(completed['status'] == 'passed' and len(completed['jobs']) == 2
                and all(x['returncode'] == 0 and x['receipt_status'] == 'passed' for x in completed['jobs']),
                'main DreamZero paired/serving queue has not passed')
        for items in c['activations'].values():
            for item in items:
                checked(item)
        checked(c['ptxas'])
        for name in IDS[:2]:
            overlay = Path(c['overlay_root']) / c['overlays'][name]['filename']
            require(overlay.is_file() and overlay.stat().st_size == c['overlays'][name]['bytes'],
                    'staged overlay missing or wrong size; public prepare will verify the full hash')
    return c


def commands(c):
    root, tmp = Path(c['run_root']), Path(c['edge_tmpfs_root'])
    result = [{'id': 'base-view', 'family': 'edge', 'argv': [PYTHON, '-I', '-B', c['base_view_helper']['path'],
               '--output', str(root / 'edge_base')], 'output': str(root / 'edge_base'), 'timeout': c['timeouts']['base_view']}]
    for name in IDS:
        family = 'edge' if name.startswith('edge-') else 'nano'
        prepared = tmp / name if family == 'edge' else root / 'prepared' / name
        argv = [PYTHON, '-I', '-B', '-m', 'cosmos3_sde1', 'prepare', name, '--base',
                str(root / 'edge_base') if family == 'edge' else c['nano_base'], '--output', str(prepared)]
        if family == 'edge':
            argv += ['--overlay', str(Path(c['overlay_root']) / c['overlays'][name]['filename'])]
        else:
            argv += ['--link-unmodified']
        result.append({'id': name + '-prepare', 'family': family, 'recipe': name, 'argv': argv,
                       'output': str(prepared), 'timeout': c['timeouts']['prepare']})
        result.append({'id': name + '-run', 'family': family, 'recipe': name,
                       'argv': [PYTHON, '-I', '-B', '-m', 'cosmos3_sde1', 'run', name, '--checkpoint', str(prepared),
                                '--output', str(root / 'runs' / name)],
                       'output': str(root / 'runs' / name), 'prepared': str(prepared),
                       'timeout': c['timeouts']['run']})
    return result


def environment(c, family):
    env = dict(os.environ)
    for item in c['activations'][family]:
        for line in checked(item).read_text().splitlines():
            parts = shlex.split(line)
            if not parts or parts[0].startswith('#'):
                continue
            if parts[0] == 'unset':
                for key in parts[1:]:
                    env.pop(key, None)
            else:
                require(parts[0] == 'export' and len(parts) == 2 and '=' in parts[1], 'unsupported activation syntax')
                key, value = parts[1].split('=', 1)
                if key == 'PATH':
                    value = value.replace('${PATH}', env.get('PATH', '')).replace('$PATH', env.get('PATH', ''))
                require('$' not in value and '`' not in value, 'unsupported activation substitution')
                env[key] = value
    for key in list(env):
        if key.startswith('IFL_') or key in {'PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE',
                                            'CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_MODULE_LOADING'}:
            del env[key]
    env['VIRTUAL_ENV'] = ENV_ROOT
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    env.update(TRITON_PTXAS_PATH=c['ptxas']['path'], TRITON_PTXAS_BLACKWELL_PATH=c['ptxas']['path'],
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    require(env.get('UV_OFFLINE') == '1' and env.get('UV_PYTHON_DOWNLOADS') == 'never'
            and Path(env['UV_PYTHON']).is_file(), 'retained offline native-HF-tool activation is incomplete')
    return env


def sources(c):
    result = {}
    for distribution, files in c['installed_sources'].items():
        dist = importlib.metadata.distribution(distribution)
        for relative, expected in files.items():
            p = Path(dist.locate_file(relative)).resolve()
            require(p.is_relative_to(Path(ENV_ROOT)) and sha(p) == expected, 'installed public source differs')
            result[str(p)] = expected
    require(str(Path(sys.prefix)) == ENV_ROOT and sys.flags.isolated and sys.flags.optimize == 0,
            'execute with corrected interpreter -I -B; assertions must remain enabled')
    return result


def idle_and_capacity(c, stage):
    apps = subprocess.check_output([c['nvidia_smi'], '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                   text=True, timeout=15).strip()
    require(not apps, 'GPU is not idle; root exclusively owns scheduling')
    def free(path):
        p = Path(path)
        while not p.exists():
            p = p.parent
        info = os.statvfs(p)
        return info.f_bavail * info.f_frsize
    tmp_free, disk_free = free(c['edge_tmpfs_root']), free(c['run_root'])
    reserve = c['minimum_tmpfs_free_bytes']
    if stage['family'] == 'edge' and stage['id'].endswith('-prepare'):
        reserve += c['edge_preparation_bytes']
    require(tmp_free >= reserve and disk_free >= c['minimum_disk_free_bytes'], 'insufficient free space before operation')
    return {'tmpfs_free_bytes': tmp_free, 'required_tmpfs_bytes': reserve, 'disk_free_bytes': disk_free,
            'GPU_compute_processes': [], 'created_unix': time.time()}


def group_members(group):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / 'stat').read_text()
            rest = text[text.rfind(')') + 2:].split()
            if int(rest[2]) == group and rest[0] != 'Z':
                members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            try:
                same_uid = path.stat().st_uid == os.getuid()
            except FileNotFoundError:
                continue
            if same_uid:
                raise
    return members


def stop_group(process):
    for sig, grace in [(signal.SIGTERM, 30), (signal.SIGKILL, 10)]:
        if not group_members(process.pid):
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        end = time.monotonic() + grace
        while group_members(process.pid) and time.monotonic() < end:
            time.sleep(.1)
    if process.poll() is None:
        process.wait(timeout=5)
    require(not group_members(process.pid), 'owned process group did not exit')


def run_process(stage, env, cwd, log_path):
    result = {'command': stage['argv'], 'started_unix': time.time(), 'timeout_seconds': stage['timeout']}
    with Path(log_path).open('x') as log:
        process = subprocess.Popen(stage['argv'], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        result['pid'] = result['process_group'] = process.pid
        try:
            result['returncode'] = process.wait(timeout=stage['timeout'])
            require(result['returncode'] == 0, 'public command failed')
            require(not group_members(process.pid), 'public command left an owned live descendant')
        except BaseException:
            result['failed'] = True
            stop_group(process)
            raise
        finally:
            result['completed_unix'] = time.time()
            result['returncode'] = process.returncode
            log.flush()
            os.fsync(log.fileno())
            write(Path(log_path).with_suffix('.process.json'), result)
    return result


def validate_preparation(name, prepared, recipes):
    result = read(prepared / 'public_preparation.json')
    spec = recipes['recipes'][name]
    require(result['recipe'] == name and result['category'] == 'SCREEN' and result['task_quality_certified'] is False
            and result['recipe_sha256'] == hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            'public preparation recipe differs')
    require(result['link_unmodified'] is (name == 'nano-original'), 'wrong preparation copy/link policy')
    require(len({x['path'] for x in result['files']}) == len(result['files']), 'duplicate prepared path')
    extra = 'budget_manifest.json' if name == 'nano-original' else 'instinctcompress_manifest.json'
    expected = {x['path'] for x in result['files']} | {'public_preparation.json', extra}
    require({str(p.relative_to(prepared)) for p in prepared.rglob('*') if p.is_file()} == expected,
            'prepared file coverage differs')
    for item in result['files']:
        p = prepared / item['path']
        require(p.resolve().is_relative_to(prepared.resolve()) and not p.is_symlink()
                and p.stat().st_size == item['bytes'], 'prepared path or file size differs')
    return result


def validate_run(name, directory, prepared, recipes, source_map):
    import numpy as np
    specification = recipes['recipes'][name]
    completion, receipt, invocation = (read(directory / f) for f in ['completion.json', 'receipt.json', 'invocation.json'])
    require(completion['status'] == 'passed' and completion['returncode'] == 0 and receipt['ok'] is True,
            'public child completion failed')
    for r in (completion, receipt):
        require(r['category'] == 'SCREEN' and r['task_quality_certified'] is False, 'quality promotion is forbidden')
    require(receipt['checkpoint'] == str(prepared) and receipt['attention'] == 'cudnn'
            and invocation['plan']['recipe'] == name, 'run recipe/checkpoint differs')
    main_path = next(p for p in source_map if p.endswith('/cosmos3_sde1/__main__.py'))
    fixture_path = next(p for p in source_map if p.endswith('/benchmarks/regression/fixtures/recorded_inputs_v1.npz'))
    benchmark_module = 'cosmos3_sde1.benchmark_' + specification['family']
    expected_command = [PYTHON, '-I', '-B', '-m', benchmark_module, str(prepared),
                        str(directory / 'receipt.json'), '--attention', 'cudnn', '--allow-unqualified',
                        '--fixture', fixture_path]
    if name == 'nano-original':
        expected_command += ['--cache-mode', 'reuse', '--swiglu']
    require(invocation['command'] == expected_command and invocation['source_sha256'] == source_map[main_path]
            and receipt['fixture_sha256'] == source_map[fixture_path] and receipt['qualification_only'] is True,
            'actual public child command, fixture or source differs')
    declaration = receipt['declared_execution']
    require(declaration == {'sigmas': [1., 0.], 'steps': 1, 'guidance': 1.,
                             'branches_per_callback': 1, 'action_padding': 'zero'}, 'complete SDE1/CFG1 contract differs')
    clock = receipt['first_request_native_branch_clocks']
    require(len(clock) == 1 and abs(clock[0] - 1000.) <= float(np.spacing(np.float32(1000.))),
            'native complete action clock differs')
    warm, measured = specification['warmup_requests'], specification['measured_requests']
    calls = receipt['calls']
    require(len(calls) == warm + measured and [x['i'] for x in calls] == list(range(warm + measured)),
            'request count/order differs')
    require(all(x['phase'] == ('warmup' if i < warm else 'measured') and math.isfinite(x['ms']) and x['ms'] > 0
                for i, x in enumerate(calls)), 'phase/finite timing differs')
    archive = directory / 'receipt.npz'
    require(sha(archive) == receipt['actions_sha256'], 'action archive differs')
    with np.load(archive, allow_pickle=False) as arrays:
        action = arrays['actions']
        require(set(arrays.files) == {'actions'} and action.shape == (warm + measured, 32, 8)
                and np.isfinite(action).all(), 'finite full-action arrays differ')
        action_dtype = str(action.dtype)
    stats = receipt['backend_stats']
    require(stats['velocity_evaluations'] == stats['sampler_calls'] == warm + measured
            and stats['action_padding'] == 'zero' and stats['padding_projection_calls'] == warm + measured
            and stats['padding_projected_velocity_branches'] == warm + measured and stats['padding_hooks_restored'] is True
            and stats['numeric_attention']['eligible_python_calls'] > 0, 'observed sampler/padding/numeric gates differ')
    require(len(receipt['external_runtime_assets']) == 1
            and receipt['external_runtime_assets'][0]['sha256'] == specification['auxiliary']['sha256']
            and receipt['external_runtime_assets'][0]['bytes'] == specification['auxiliary']['bytes'], 'native VAE load differs')
    require(receipt['sources'] and all(p in source_map and source_map[p] == h
                                      for p, h in receipt['sources'].items()),
            'executed source differs from pinned public source')
    expected_bench = next(h for p, h in source_map.items() if p.endswith('/cosmos3_sde1/benchmark_' + specification['family'] + '.py'))
    require(receipt['benchmark_sha256'] == expected_bench, 'public benchmark source differs')
    median = statistics.median(x['ms'] for x in calls[warm:])
    require(median == receipt['p50_ms'] == completion['p50_ms'], 'reported median differs')
    if name == 'nano-original':
        require(receipt['trained_student'] is False and receipt['swiglu_requested'] is True
                and receipt['cache_mode'] == 'reuse' and receipt['format_prompt_as_json'] is False,
                'Nano original-role/cache/plain-prompt contract differs')
        for i in range(warm, warm + measured):
            before, after = calls[i - 1]['graph_counters'], calls[i]['graph_counters']
            require(after['captures'] == before['captures'] and after['checks'] == before['checks']
                    and after['replays'] - before['replays'] == 36, 'Nano timed request prepared graphs or missed a layer')
    return {'status': 'passed', 'recipe': name, 'p50_ms': median, 'requests': len(calls),
            'action_shape': [len(calls), 32, 8], 'action_dtype': action_dtype,
            'category': 'SCREEN', 'task_quality_certified': False,
            'completion': ref(directory / 'completion.json'), 'receipt': ref(directory / 'receipt.json'), 'actions': ref(archive)}


def archive_small(prepared, run, destination):
    destination.mkdir()
    inventory = []
    total = 0
    for label, root in [('preparation', prepared), ('run', run)]:
        for source in sorted(root.rglob('*')):
            require(not source.is_symlink(), 'evidence symlink is forbidden')
            if not source.is_file() or (label == 'preparation' and source.suffix == '.safetensors'):
                continue
            total += source.stat().st_size
            require(source.stat().st_size <= 64 << 20 and total <= 160 << 20, 'small evidence archive exceeds bound')
            before = ref(source)
            target = destination / label / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out, 1 << 20)
                out.flush()
                os.fsync(out.fileno())
            require(sha(target) == before['sha256'] and ref(source) == before, 'evidence changed during copy')
            inventory.append({'source': before, 'copy': ref(target)})
    for p in sorted((p for p in destination.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        sync_dir(p)
    sync_dir(destination)
    require(all(checked(x['copy']) for x in inventory), 'durable evidence rehash failed')
    result = {'status': 'archived_small_evidence', 'files': inventory, 'bytes': total,
              'large_weight_bytes_copied': 0, 'original_run_and_sources_retained': True}
    write(destination / 'archive.json', result)
    return result


def same_uid_handles(target):
    """Only private same-UID staging: unreadable live same-UID handles fail closed."""
    checked_pids, excluded_pids = [], []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                excluded_pids.append(int(process.name))
                continue
            state = (process / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
            if state == 'Z':
                continue
            entries = [(process / 'cwd').readlink(), (process / 'root').readlink()]
            for descriptor in (process / 'fd').iterdir():
                try:
                    entries.append(descriptor.readlink())
                except FileNotFoundError:
                    # A descriptor which has closed cannot retain our files.
                    continue
            for line in (process / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith('/'):
                    entries.append(Path(fields[5]))
            for p in entries:
                text = str(p).removesuffix(' (deleted)')
                require(not Path(text).is_relative_to(target), 'live same-UID handle references owned preparation')
            checked_pids.append(int(process.name))
        except FileNotFoundError:
            # A disappearing descriptor alone does not establish process exit.
            if process.exists():
                raise ValueError('same-UID process/descriptor changed during staging handle check') from None
        except PermissionError:
            if process.exists():
                raise ValueError('unreadable live same-UID process; staging retained') from None
    return {'scope': 'same_effective_uid_only', 'checked_pids': checked_pids,
            'other_uid_pids_excluded': excluded_pids, 'global_handle_absence_claimed': False}


def reclaim_edge(c, name, prepared, archive, ownership):
    require(c['reclaim_owned_edge_preparations'] and name in IDS[:2], 'only explicit owned Edge cleanup is permitted')
    tmp = Path(c['edge_tmpfs_root'])
    require(prepared == tmp / name and prepared.resolve() == prepared and not prepared.is_symlink(),
            'reclamation target is not an exact owned Edge preparation')
    require(read(tmp / 'ownership.json') == ownership and ownership['config_sha256'] == c['_config_sha256']
            and not tmp.is_symlink() and tmp.resolve() == tmp
            and tmp.stat().st_uid == os.getuid() and tmp.stat().st_mode & 0o777 == 0o700,
            'private owned preparation root changed')
    require(read(archive / 'run/completion.json')['status'] == 'passed', 'no passed public completion archive')
    archived = read(archive / 'archive.json')
    require(archived['status'] == 'archived_small_evidence' and all(checked(x['copy']) for x in archived['files']),
            'archived evidence is not durably byte verified')
    validated = read(archive / 'validation.json')
    require(validated['status'] == 'passed' and validated['recipe'] == name,
            'exact recipe numerical/lifecycle validation is required before reclamation')
    for item in archived['files']:
        checked(item['source'])
    entries = []
    for p in prepared.rglob('*'):
        require(not p.is_symlink() and p.stat().st_uid == os.getuid(), 'unexpected owner or symlink inside staging')
        if p.is_file():
            require(p.stat().st_nlink == 1, 'Edge default preparation unexpectedly shares an inode')
            entries.append(str(p.relative_to(prepared)))
    declared = read(prepared / 'public_preparation.json')
    require(set(entries) == {x['path'] for x in declared['files']}
            | {'public_preparation.json', 'instinctcompress_manifest.json'}, 'unexpected staging file; preserve it')
    # The prior finite process group has exited. Unrelated/private privileged
    # handles are not claimed visible; this is only our private same-UID staging.
    handles = same_uid_handles(prepared)
    shutil.rmtree(prepared)
    sync_dir(tmp)
    return {'status': 'owned_preparation_reclaimed', 'path': str(prepared), 'recipe': name,
            'archive': ref(archive / 'archive.json'), 'files_unlinked': len(entries),
            'handle_check': handles,
            'scope': 'Only this queue-created independent Edge copy; originals/overlays/base view/Nano/run evidence remain.'}


def execute(c, config_path):
    c['_config_sha256'] = sha(config_path)
    root, tmp = Path(c['run_root']), Path(c['edge_tmpfs_root'])
    require(not root.exists() and not tmp.exists(), 'fresh run and tmpfs roots required; no resume/retry')
    operations = commands(c)
    first_capacity = idle_and_capacity(c, operations[0])
    source_map = sources(c)
    root.mkdir(mode=0o700)
    tmp.mkdir(mode=0o700)
    for name in ('control', 'runs', 'prepared', 'archives', 'compiler_caches'):
        (root / name).mkdir()
    ownership = {'run_root': str(root), 'tmpfs_root': str(tmp), 'owner_uid': os.getuid(),
                 'config_sha256': c['_config_sha256'], 'source_sha256': sha(__file__),
                 'only_reclaimable_children': list(IDS[:2])}
    write(tmp / 'ownership.json', ownership)
    write(root / 'config.json', {k: v for k, v in c.items() if not k.startswith('_')})
    write(root / 'ownership.json', ownership)
    write(root / 'source_guard.json', source_map)
    write(root / 'initial_capacity.json', first_capacity)
    recipe_path = next(Path(p) for p in source_map if p.endswith('/cosmos3_sde1/data/recipes.json'))
    recipes = read(recipe_path)
    record = {'status': 'running', 'started_unix': time.time(), 'operations': [], 'screens': [],
              'category': 'SCREEN', 'task_quality_certified': False, 'automatic_retry': False}
    try:
        for stage in operations:
            row = {'id': stage['id'], 'capacity': idle_and_capacity(c, stage)}
            write(root / 'control' / (stage['id'] + '.capacity.json'), row['capacity'])
            env = environment(c, stage['family'])
            cache = root / 'compiler_caches' / stage['id']
            cache.mkdir()
            env.update(TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'))
            selected = ('VIRTUAL_ENV', 'HF_HOME', 'HF_HUB_CACHE', 'UV_CACHE_DIR', 'UV_TOOL_DIR', 'UV_PYTHON',
                        'UV_OFFLINE', 'TRITON_PTXAS_PATH', 'TRITON_PTXAS_BLACKWELL_PATH',
                        'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')
            row['applied_environment'] = {k: env[k] for k in selected if k in env}
            row['ptxas'] = ref(checked(c['ptxas']))
            write(root / 'control' / (stage['id'] + '.launch.json'), {'stage': stage, **row})
            row['process'] = run_process(stage, env, c['base'], root / 'control' / (stage['id'] + '.log'))
            require(sources(c) == source_map, 'installed source changed during operation')
            if stage['id'] == 'base-view':
                require(read(Path(stage['output']) / 'base_view_manifest.json')['status'] == 'passed', 'base view failed')
            elif stage['id'].endswith('-prepare'):
                row['preparation'] = validate_preparation(stage['recipe'], Path(stage['output']), recipes)
            else:
                name, prepared, run = stage['recipe'], Path(stage['prepared']), Path(stage['output'])
                row['screen'] = validate_run(name, run, prepared, recipes, source_map)
                validate_preparation(name, prepared, recipes)
                archived = root / 'archives' / name
                archive_small(prepared, run, archived)
                write(archived / 'validation.json', row['screen'])
                record['screens'].append(row['screen'])
                if name in IDS[:2] and c['reclaim_owned_edge_preparations']:
                    idle_and_capacity(c, {'id': name + '-cleanup', 'family': 'edge'})
                    row['reclamation'] = reclaim_edge(c, name, prepared, archived, ownership)
                    write(archived / 'reclamation.json', row['reclamation'])
            record['operations'].append(row)
            write(root / 'control' / (stage['id'] + '.completed.json'), row)
        require([r['recipe'] for r in record['screens']] == list(IDS), 'three SCREEN results are incomplete')
        record['status'] = 'passed_three_public_recipe_screens'
    except BaseException as error:
        record.update(status='failed_preserved_no_retry', error=repr(error))
    finally:
        record['completed_unix'] = time.time()
        write(root / 'completion.json', record)
    return record


def stop_requested(signum, frame):
    global _STOPPING
    if not _STOPPING:
        _STOPPING = True
        raise KeyboardInterrupt(f'Owned queue received signal{signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-only', action='store_true', help='Validate sources/environment, no GPU query or weight reads')
    args = parser.parse_args()
    if args.execute or args.check_only:
        require(args.config_sha256 == sha(args.config), 'explicit frozen config hash required')
    c = load_config(args.config, args.execute or args.check_only)
    if args.check_only:
        source_map = sources(c)
        for family in ('edge', 'nano'):
            environment(c, family)
        print(json.dumps({'status': 'static_inputs_checked', 'source_files': len(source_map),
                          'config_sha256': sha(args.config), 'queue_source_sha256': sha(__file__),
                          'GPU_calls': 0, 'weight_hashes': 0, 'native_imports': 0}, indent=2))
        return 0
    if not args.execute:
        print(json.dumps({'status': 'planned_only', 'commands': commands(c), 'GPU_used': False}, indent=2))
        return 0
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    result = execute(c, args.config)
    return 0 if result['status'] == 'passed_three_public_recipe_screens' else 1


if __name__ == '__main__':
    raise SystemExit(main())
