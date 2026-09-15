"""Snapshot a committed checkout, run Thor regression via SSH, retrieve receipts.

Suitable for a systemd timer. No credentials are stored or needed beyond the
operator's existing SSH setup. Never runs uncommitted work or modifies Thor's
model installation. Contention returns nonzero rather than mixing workloads.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-baseline', action='store_true', help='Explicitly accept this passing run as the performance baseline')
    parser.add_argument('--host', required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--python', required=True, help='Existing Thor Cosmos interpreter')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.host.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.@-]+', args.host):
        parser.error('Invalid SSH host')
    for value in (args.remote_root, args.python):
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+', value) or '..' in Path(value).parts:
            parser.error('Remote paths must be absolute paths without shell metacharacters')
    try:
        return run_once(args)
    except Exception as error:
        args.output_root.mkdir(parents=True, exist_ok=True)
        latest = args.output_root / 'latest.json'
        temp = latest.with_suffix('.tmp')
        temp.write_text(json.dumps({'status': 'ERROR', 'exit_code': 1, 'error': repr(error)}, indent=2))
        temp.replace(latest)
        return 1


def run_once(args):
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + revision[:12]
    local = args.output_root.resolve() / stamp
    local.mkdir(parents=True)
    (args.output_root / 'latest.json').write_text(json.dumps({'run': str(local), 'status': 'RUNNING', 'exit_code': None}))
    remote = args.remote_root.rstrip('/') + '/' + stamp
    def ssh(command, **kwargs):
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host, command], check=True, **kwargs)
    ssh('mkdir -p ' + shlex.quote(remote + '/source'))
    archive = subprocess.run(['git', 'archive', revision], cwd=root, stdout=subprocess.PIPE, check=True).stdout
    ssh('tar -x -C ' + shlex.quote(remote + '/source'), input=archive)
    remote_path = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host, 'printf %s "$HOME/.local/bin:$PATH"'], text=True)
    command = 'cd ' + shlex.quote(remote + '/source') + ' && ' + shlex.join([
        'env', 'PATH=' + str(Path(args.python).parent) + ':/usr/local/cuda/bin:' + remote_path,
        'PYTHONPATH=' + remote + '/source', args.python, '-m', 'benchmarks.regression.run_cosmos', remote + '/results'])
    with (local / 'controller.log').open('w') as log:
        run = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host, command], stdout=log, stderr=subprocess.STDOUT)
    fetched = subprocess.run(['rsync', '-a', args.host + ':' + remote + '/results/', str(local / 'results')])
    (local / 'run.json').write_text(json.dumps({'revision': revision, 'remote': remote,
        'exit_code': run.returncode, 'retrieval_exit_code': fetched.returncode}, indent=2))
    from benchmarks.regression.history import record, compare_history
    accepted_path = args.output_root.resolve() / 'accepted-baseline.json'
    history = {'status': 'INVALID', 'error': 'Paired regression did not pass'}
    if run.returncode == fetched.returncode == 0:
        current = {}
        for family in ('edge', 'nano'):
            report = json.loads((local / 'results' / f'{family}-current.json').read_text())
            comparison = json.loads((local / 'results' / f'{family}-comparison.json').read_text())
            current[family] = record(report, comparison)
        if args.accept_baseline:
            accepted_path.write_text(json.dumps(current, indent=2))
            history = {'status': 'PASS', 'accepted_from': str(local)}
        elif accepted_path.exists():
            history = compare_history(current, json.loads(accepted_path.read_text()))
        else:
            history = {'status': 'UNBASELINED', 'error': 'Run once with --accept-baseline after reviewing the protocol'}
    (local / 'history.json').write_text(json.dumps(history, indent=2))
    exit_code = run.returncode or fetched.returncode or (0 if history['status'] == 'PASS' else 1)
    latest = args.output_root.resolve() / 'latest.json'
    temp = latest.with_suffix('.tmp')
    temp.write_text(json.dumps({'run': str(local), 'exit_code': exit_code, 'status': 'PASS' if exit_code == 0 else 'FAIL',
                              'retrieval_exit_code': fetched.returncode}, indent=2))
    temp.replace(latest)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
