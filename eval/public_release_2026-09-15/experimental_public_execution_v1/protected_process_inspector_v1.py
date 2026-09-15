"""Read-only /proc inspection with six explicitly bound infrastructure exceptions.

No sudo, signals, file cleanup, native imports or GPU queries. A protected field
is excluded only for an exact current identity captured from the six prior
authorized service PIDs. All readable fields are still checked. Unknown/new
protected processes, including transient SSH workers, remain fatal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

KNOWN = {2039: 'systemd', 2093: '(sd-pam)', 88574: 'tailscaled', 90733: 'tailscaled',
         90749: 'tailscaled', 96704: 'tailscaled'}
PRIOR = {'precheck.json': 'd2b77c00293bc428e295c92f775e09da847c74f0d946185491607cffe4b36e5d',
         'authorized_reclaim_plan.json': '3ad16b0f61b01a0fbdd0889c072177f6e3160d56f7b74c45c54fdf0c2f5519de',
         'completion.json': '05d45a4a72768e6cb258fb215279fff6f820796a2ef19dc0bb539e9d494b4b7d'}
FIELDS = ('cwd', 'root', 'fd', 'maps', 'environ')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def basic_identity(pid):
    process = Path('/proc') / str(pid)
    raw = (process / 'stat').read_text()
    fields = raw.rsplit(') ', 1)[1].split()
    status = {row.split(':', 1)[0]: row.split(':', 1)[1].strip()
              for row in (process / 'status').read_text().splitlines() if ':' in row}
    return {'pid': pid, 'start_ticks': int(fields[19]), 'ppid': int(fields[1]),
            'comm': (process / 'comm').read_text().rstrip('\n'),
            'uids': [int(x) for x in status['Uid'].split()],
            'gids': [int(x) for x in status['Gid'].split()]}, fields[0]


def service_identity(pid):
    identity, state = basic_identity(pid)
    process = Path('/proc') / str(pid)
    command = (process / 'cmdline').read_bytes()
    argv0 = command.split(b'\0', 1)[0].decode('utf-8')
    identity.update(argv0=argv0, cmdline_sha256=hashlib.sha256(command).hexdigest(),
                    cgroup_sha256=sha(process / 'cgroup'))
    if identity['ppid']:
        identity['parent'], _ = basic_identity(identity['ppid'])
    else:
        identity['parent'] = None
    require(identity['comm'] == KNOWN[pid] and identity['uids'] == [os.getuid()] * 4,
            'known service comm/UID differs')
    names = {'systemd': {'systemd'}, '(sd-pam)': {'(sd-pam)', 'systemd'}, 'tailscaled': {'tailscaled'}}
    require(Path(argv0).name in names[identity['comm']], 'known service executable argv0 differs')
    require(state != 'Z', 'known infrastructure process is a zombie')
    return identity


def field_paths(process, field):
    if field in ('cwd', 'root'):
        return [str((process / field).readlink())]
    if field == 'fd':
        result = []
        for item in (process / 'fd').iterdir():
            try:
                result.append(str(item.readlink()))
            except FileNotFoundError:
                continue  # closed descriptor cannot retain a target
        return result
    if field == 'maps':
        rows = [line.split(None, 5) for line in (process / field).read_text().splitlines()]
        return [row[5] for row in rows if len(row) == 6 and row[5].startswith('/')]
    require(field == 'environ', 'unknown /proc field')
    result = []
    for item in (process / 'environ').read_bytes().split(b'\0'):
        key, _, value = item.partition(b'=')
        if key in {b'TORCHINDUCTOR_CACHE_DIR', b'TRITON_CACHE_DIR'}:
            result.append(value.decode('utf-8'))
    return result


def inspect_fields(process):
    observed, denied = {}, []
    for field in FIELDS:
        try:
            observed[field] = field_paths(process, field)
        except PermissionError:
            denied.append(field)
    return observed, denied


def capture(prior_dir):
    require(os.getuid() == 1000, 'capture must retain the actual file-owner UID1000')
    for name, expected in PRIOR.items():
        require(sha(prior_dir / name) == expected, 'original authorization proof changed')
    old = json.loads((prior_dir / 'authorized_reclaim_plan.json').read_text())
    require({r['pid']: r['comm'] for r in old['denied_service_check'] if 'comm' in r} == KNOWN,
            'prior explicit infrastructure PID set differs')
    rows, exited = [], []
    for pid in KNOWN:
        process = Path('/proc') / str(pid)
        if not process.exists():
            exited.append(pid)
            continue
        identity = service_identity(pid)
        observed, denied = inspect_fields(process)
        require(service_identity(pid) == identity, 'service identity changed during capture')
        rows.append({'identity': identity, 'denied_fields': denied, 'readable_fields': sorted(observed)})
    return {'schema': 'instinctflash.protected_infrastructure_snapshot.v1', 'status': 'captured_read_only',
            'capture_source_sha256': sha(__file__), 'boot_id': boot_id(), 'uid': os.getuid(),
            'created_unix': time.time(), 'prior_authorization_sha256': PRIOR,
            'services': rows, 'known_PIDs_now_exited': exited,
            'previously_exited_PID_never_authorized': 288084,
            'scope': 'Only exactly bound protected fields on six prior authorized infrastructure identities; no global handle-absence claim.',
            'unknown_protected_processes': 'fatal, including new transient SSH/Tailscale connections',
            'GPU_queries': 0, 'native_imports': 0, 'process_signals': 0}


def load_policy(path):
    policy = json.loads(Path(path).read_text())
    require(policy['schema'] == 'instinctflash.protected_infrastructure_snapshot.v1'
            and policy['status'] == 'captured_read_only' and policy['capture_source_sha256'] == sha(__file__)
            and policy['boot_id'] == boot_id() and policy['uid'] == os.getuid() == 1000,
            'protected-process snapshot host/source/UID differs')
    require(policy['prior_authorization_sha256'] == PRIOR, 'prior authorization binding differs')
    rows = policy['services']
    require(len({r['identity']['pid'] for r in rows}) == len(rows)
            and {r['identity']['pid'] for r in rows} | set(policy['known_PIDs_now_exited']) == set(KNOWN),
            'service identity coverage differs')
    for row in rows:
        identity = row['identity']
        require(identity['comm'] == KNOWN[identity['pid']] and identity['uids'] == [1000] * 4
                and type(identity['start_ticks']) is int and identity['start_ticks'] > 0
                and set(row['denied_fields']) <= set(FIELDS), 'unexpected protected service identity/fields')
    return policy


def authorize_denied(pid, identity, denied, policy):
    rows = [row for row in policy['services'] if row['identity']['pid'] == pid]
    require(len(rows) == 1 and identity == rows[0]['identity'],
            'unreadable unknown/restarted protected process; scratch retained')
    require(set(denied) <= set(rows[0]['denied_fields']), 'new protected field outside captured exception')
    return {'identity': identity, 'excluded_unreadable_fields': denied,
            'scope': 'bound protected infrastructure only; readable fields still checked'}


def inspect(targets, policy):
    checked_pids, other_uid, exceptions = [], [], []
    def under(value):
        value = Path(value.removesuffix(' (deleted)'))
        return any(value == p or value.is_relative_to(p) for p in targets)
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        pid = int(process.name)
        try:
            if process.stat().st_uid != os.getuid():
                other_uid.append(pid)
                continue
            basic, state = basic_identity(pid)
            if state == 'Z':
                continue
            observed, denied = inspect_fields(process)
            require(not any(under(value) for values in observed.values() for value in values),
                    'live same-UID reference to scratch/cache target')
            if denied:
                require(pid in KNOWN, 'unreadable unknown/model/Python process; scratch retained')
                identity = service_identity(pid)
                exceptions.append(authorize_denied(pid, identity, denied, policy))
                require(service_identity(pid) == identity, 'protected service changed during inspection')
            require(basic_identity(pid)[0] == basic, 'same-UID identity changed during inspection')
            checked_pids.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            if process.exists():
                raise ValueError('live process metadata changed during inspection; scratch retained') from None
        except PermissionError:
            if process.exists():
                raise ValueError('identity itself unreadable; scratch retained') from None
    return {'scope': 'same_effective_uid_with_exact_protected_infrastructure_exceptions',
            'checked_pids': checked_pids, 'other_uid_pids_excluded': other_uid,
            'protected_infrastructure_exclusions': exceptions, 'global_handle_absence_claimed': False,
            'unknown_protected_processes_ignored': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.prior_dir)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'snapshot': str(args.output), 'sha256': sha(args.output),
                      'services': [{'pid': r['identity']['pid'], 'comm': r['identity']['comm'],
                                    'start_ticks': r['identity']['start_ticks'], 'denied_fields': r['denied_fields']}
                                   for r in result['services']], 'read_only': True}))


if __name__ == '__main__':
    main()
