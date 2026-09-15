"""Archive two exact inactive task-owned inputs to fresh tmpfs, then optionally unlink.

No model imports, downloads, SSH, process signals or environment/package edits.
Root invokes this once under /tmp/thor_gpu.lock after its main queue completes.
Tmpfs is volatile: copy the resulting archive to durable storage before shutdown.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import time

BASE = Path('/home/guanming/ifl_public_qualification_20260915_v1')
CACHE = BASE / 'dreamzero_compiler_cache_v5'
WHEEL = BASE / 'cosmos_compiler_env_v1/triton-3.6.0-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl'
WHEEL_SHA = '58d57d6796b0004076315433526fe9d4af42044d430afdee1e6cd42a76bd6d09'
WHEEL_BYTES = 176135525
V5_CONFIG_SHA = 'dbbaaf7f391c30dd78b5c9de10738780bc3347b5f0d8352e998d3d187a13e279'
V6_CONFIG_SHA = '9c86963b0adbd4b92f0465e04ca27c7b0775e7fdeb68517a4983d80500b18247'
GIB = 1 << 30
HANDLE_INSPECTOR_SHA256 = '3dbd83aba8a3cc3a15a299d351cc7ccbec0965c5a943bcd48fa2fb18daf4e2bb'
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
    return json.loads(Path(path).read_text())


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    sync_dir(Path(path).parent)


def checked(item):
    p = Path(item['path'])
    require(p.is_absolute() and p.resolve() == p and p.is_file() and sha(p) == item['sha256'],
            'bound operational input changed')
    return p


def handle_authorization(c):
    """Verify exact inspector bytes before import; host/PID policy stays explicit."""
    source = checked(c['handle_inspector'])
    require(sha(source) == HANDLE_INSPECTOR_SHA256, 'unexpected handle inspector implementation')
    snapshot = checked(c['protected_service_manifest'])
    spec = importlib.util.spec_from_file_location('bound_protected_infrastructure_inspector', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.load_policy(snapshot)
    return module, policy


def load_config(path, executing=False):
    c = read(path)
    require(c['schema'] == 'instinctflash.inactive_V5_archive.v2'
            and c['cache_source'] == str(CACHE) and c['optional_wheel_source'] == str(WHEEL),
            'only the exact V5 cache and official compiler wheel may be archived')
    require(type(c['include_wheel']) is bool and type(c['unlink_original_after_verified_archive']) is bool,
            'archive and unlink options must be explicit booleans')
    output = Path(c['archive_root'])
    require(output.is_absolute() and output.resolve() == output and output.parent == Path('/dev/shm')
            and output.name.startswith('ifl_public_dreamzero_v5_archive_'), 'fresh owned tmpfs namespace required')
    require(c['minimum_tmpfs_remaining_bytes'] >= 12 * GIB and 1 <= c['timeout_seconds'] <= 1800,
            'tmpfs floor/lifetime is out of bounds')
    if executing:
        require(c['template_only'] is False and c['helper_sha256'] == sha(__file__), 'unfrozen helper/config')
        handle_authorization(c)
        v5, v6 = checked(c['failed_queue_config']), checked(c['completed_queue_config'])
        require(sha(v5) == V5_CONFIG_SHA and sha(v6) == V6_CONFIG_SHA, 'expected V5/V6 source config changed')
        old, current = read(v5), read(v6)
        require(all(j['environment']['TORCHINDUCTOR_CACHE_DIR'] == str(CACHE / 'inductor')
                    and j['environment']['TRITON_CACHE_DIR'] == str(CACHE / 'triton') for j in old['jobs']),
                'failed queue does not own this cache')
        failure = read(checked(c['failed_queue_completion']))
        require(failure['status'] == 'failed', 'V5 failure evidence is missing')
        done = read(checked(c['completed_queue_completion']))
        require(done['status'] == 'passed' and len(done['jobs']) == len(current['jobs']) == 2,
                'main queue has not completed')
        for expected, actual in zip(current['jobs'], done['jobs']):
            require(actual['id'] == expected['id'] and type(actual['returncode']) is int
                    and actual['returncode'] == 0 and actual['receipt_status'] == 'passed', 'main command failed')
            require(actual['command'] == [expected['python'], '-I', '-m', expected['module'], *expected['arguments']],
                    'main command/recipe changed')
        pid, ticks = c['completed_queue_process']['pid'], c['completed_queue_process']['start_ticks']
        require(type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0,
                'actual predecessor PID/start-ticks binding is required')
        process = Path('/proc') / str(pid)
        if process.exists():
            fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
            require(int(fields[19]) != ticks or fields[0] == 'Z', 'completed queue process is still live')
    return c


def metadata(path):
    info = path.lstat()
    require(not stat.S_ISLNK(info.st_mode) and path.resolve() == path and info.st_uid == os.getuid(),
            'source ownership or symlink differs')
    require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), 'unsupported file type inside cache')
    if stat.S_ISREG(info.st_mode):
        require(info.st_nlink == 1, 'source has an outside hardlink; preserve it')
    return {'dev': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid,
            'mode': stat.S_IMODE(info.st_mode), 'bytes': info.st_size,
            'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns, 'blocks': info.st_blocks,
            'kind': 'file' if stat.S_ISREG(info.st_mode) else 'directory'}


def inventory(path, hashing):
    require(path.is_absolute() and path.exists(), 'exact source is absent')
    paths = [path] + sorted(path.rglob('*')) if path.is_dir() else [path]
    require(len(paths) <= 100000, 'bounded archive file count exceeded')
    result = {}
    for item in paths:
        name = str(item.relative_to(path))
        row = metadata(item)
        if hashing and row['kind'] == 'file':
            row['sha256'] = sha(item)
            require(metadata(item) == {k: v for k, v in row.items() if k != 'sha256'}, 'source changed while hashing')
        result[name] = row
    require(sum(x['bytes'] for x in result.values() if x['kind'] == 'file') <= 2 * GIB,
            'bounded cache archive size exceeded')
    return result


def gpu_idle():
    applications = subprocess.check_output(['/usr/sbin/nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    require(not applications, 'GPU is not idle; archive must wait for root scheduling')
    return {'GPU_compute_processes': [], 'created_unix': time.time()}


def inactive_handles(targets, authorization):
    """Keep file-owner UID and exclude only exact prebound protected service fields."""
    module, policy = handle_authorization(authorization)
    result = module.inspect(targets, policy)
    result.update(inspector_sha256=HANDLE_INSPECTOR_SHA256,
                  protected_service_manifest=authorization['protected_service_manifest'])
    return result


def free(path):
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def copy_tree(source, destination, original):
    for name, row in original.items():
        src = source if name == '.' else source / name
        dst = destination if name == '.' else destination / name
        require(metadata(src) == {k: v for k, v in row.items() if k != 'sha256'}, 'source metadata changed before copy')
        if row['kind'] == 'directory':
            dst.mkdir(mode=0o700)
            continue
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as inp, dst.open('xb') as out:
            shutil.copyfileobj(inp, out, 8 << 20)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(dst, row['mode'])
        os.utime(dst, ns=(row['mtime_ns'], row['mtime_ns']))
        require(sha(dst) == row['sha256'] and dst.stat().st_size == row['bytes'], 'copied file bytes differ')
    directories = [destination] + list(destination.rglob('*')) if destination.is_dir() else []
    for path in sorted((p for p in directories if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        sync_dir(path)
    sync_dir(destination.parent)


def copied_payload(original, copied):
    def content(tree):
        return {key: {'kind': row['kind'], **({'bytes': row['bytes'], 'sha256': row['sha256'],
                                             'mode': row['mode'], 'mtime_ns': row['mtime_ns']}
                                            if row['kind'] == 'file' else {})} for key, row in tree.items()}
    require(content(original) == content(copied), 'whole archive content changed')


def execute(c, config_path):
    output = Path(c['archive_root'])
    require(not output.exists(), 'fresh archive required; no overwrite/resume/retry')
    targets = [CACHE] + ([WHEEL] if c['include_wheel'] else [])
    idle = gpu_idle()
    handles = inactive_handles(targets, c)
    initial = {str(p): inventory(p, False) for p in targets}
    size = sum(r['bytes'] for tree in initial.values() for r in tree.values() if r['kind'] == 'file')
    require(free(Path('/dev/shm')) >= size + c['minimum_tmpfs_remaining_bytes'], 'tmpfs would lose required headroom')
    require(all(p.stat().st_dev != Path('/dev/shm').stat().st_dev for p in targets), 'source is not on a separate NVMe filesystem')
    output.mkdir(mode=0o700)
    before_free = free(BASE)
    result = {'status': 'copying', 'started_unix': time.time(), 'helper_sha256': sha(__file__),
              'config_sha256': sha(config_path), 'source_targets': [str(p) for p in targets],
              'archive_root': str(output), 'owner_uid': os.getuid(), 'GPU_native_work': False,
              'idle_before': idle, 'handles_before': handles, 'NVMe_free_before': before_free,
              'source_unlink_authorized': c['unlink_original_after_verified_archive'],
              'tmpfs_volatile': True, 'automatic_retry': False}
    write(output / 'ownership.json', result)
    write(output / 'config.json', c)
    try:
        snapshots = {str(p): inventory(p, True) for p in targets}
        require({p: {n: {k: v for k, v in r.items() if k != 'sha256'} for n, r in t.items()}
                 for p, t in snapshots.items()} == initial, 'source changed since initial inventory')
        if c['include_wheel']:
            row = snapshots[str(WHEEL)]['.']
            require(row['bytes'] == WHEEL_BYTES and row['sha256'] == WHEEL_SHA, 'official Triton wheel hash differs')
        write(output / 'original_inventory.json', snapshots)
        copied = {}
        for source in targets:
            destination = output / source.name
            copy_tree(source, destination, snapshots[str(source)])
            copied[str(source)] = inventory(destination, True)
            copied_payload(snapshots[str(source)], copied[str(source)])
        require({str(p): inventory(p, True) for p in targets} == snapshots, 'source changed during archival')
        write(output / 'copied_inventory.json', copied)
        acknowledgement = {'status': 'all_bytes_copied_fsynced_and_rehashed',
                           'original_inventory_sha256': sha(output / 'original_inventory.json'),
                           'copied_inventory_sha256': sha(output / 'copied_inventory.json'),
                           'logical_file_bytes': size, 'tmpfs_volatile': True,
                           'source_paths': [str(p) for p in targets]}
        write(output / 'verified_archive.json', acknowledgement)
        result['verified_archive_sha256'] = sha(output / 'verified_archive.json')
        if c['unlink_original_after_verified_archive']:
            result['idle_before_unlink'] = gpu_idle()
            result['handles_before_unlink'] = inactive_handles(targets, c)
            # New ownership/config and both complete inventories remain byte-bound.
            require(read(output / 'ownership.json')['config_sha256'] == sha(config_path)
                    and output.stat().st_uid == os.getuid() and output.stat().st_mode & 0o777 == 0o700,
                    'private archive ownership changed')
            for p in targets:
                require(inventory(p, True) == snapshots[str(p)], 'source changed before unlink')
                copied_payload(snapshots[str(p)], inventory(output / p.name, True))
            write(output / 'unlink_authorization.json', {'status': 'exact_sources_fully_archived',
                                                       'config_sha256': sha(config_path),
                                                       'verified_archive_sha256': result['verified_archive_sha256'],
                                                       'paths': [str(p) for p in targets]})
            for p in targets:
                if p == CACHE:
                    shutil.rmtree(p)
                else:
                    require(p == WHEEL, 'unallowlisted unlink target')
                    p.unlink()
                sync_dir(p.parent)
            result['status'] = 'archived_verified_original_staging_unlinked'
        else:
            result['status'] = 'archived_verified_originals_retained'
        result.update(logical_bytes_archived=size, NVMe_free_after=free(BASE), tmpfs_free_after=free(Path('/dev/shm')),
                      sde1_NVMe_floor_satisfied=free(BASE) >= GIB)
    except BaseException as error:
        result.update(status='failed_preserved_no_retry', error=repr(error))
    finally:
        result['completed_unix'] = time.time()
        result['original_paths_still_present'] = [str(p) for p in targets if p.exists()]
        write(output / 'completion.json', result)
    return result


def deadline(signum, frame):
    global _STOPPING
    if not _STOPPING:
        _STOPPING = True
        signal.alarm(0)
        raise TimeoutError('Bounded archive lifetime exceeded or stop requested')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        require(args.config_sha256 == sha(args.config), 'explicit frozen archive config hash required')
    c = load_config(args.config, args.execute)
    if not args.execute:
        print(json.dumps({'status': 'planned_only', 'cache': str(CACHE),
                          'wheel': str(WHEEL) if c['include_wheel'] else None,
                          'archive': c['archive_root'], 'GPU_queries': 0, 'source_reads': 0}, indent=2))
        return 0
    for value in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(value, deadline)
    signal.alarm(c['timeout_seconds'])
    try:
        result = execute(c, args.config)
    finally:
        signal.alarm(0)
    print(json.dumps({'status': result['status'], 'archive': c['archive_root']}))
    return 0 if result['status'].startswith('archived_verified_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
