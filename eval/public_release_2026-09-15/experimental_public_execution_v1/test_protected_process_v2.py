"""Bounded CPU proof of the explicit infrastructure exception, no remote work."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import types

import pytest

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inspector = load('protected_process_inspector_v2')
queue = load('run_queue_v2')
archive = load('archive_inactive_v5_cache_v2')
SNAPSHOT = HERE / 'protected_services_snapshot_v1.json'
ACTUAL_SHA = '39d58cb80df248a27df9a2a57265b30f0acf5d20afac963aca9986c4cf82f3fa'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def policy(monkeypatch):
    result = json.loads(SNAPSHOT.read_text())
    assert sha(SNAPSHOT) == ACTUAL_SHA
    monkeypatch.setattr(inspector, 'boot_id', lambda: result['boot_id'])
    monkeypatch.setattr(inspector.os, 'getuid', lambda: 1000)
    return inspector.load_policy(SNAPSHOT)


def test_actual_capture_binds_all_six_with_no_dynamic_pid_class(policy):
    assert {r['identity']['pid'] for r in policy['services']} == set(inspector.KNOWN)
    assert policy['known_PIDs_now_exited'] == []
    for row in policy['services']:
        identity = row['identity']
        receipt = inspector.authorize_denied(identity['pid'], identity, list(inspector.FIELDS), policy)
        assert receipt['identity'] == identity
        assert receipt['excluded_unreadable_fields'] == list(inspector.FIELDS)
    assert 288084 not in inspector.KNOWN


@pytest.mark.parametrize('field', ['start_ticks', 'comm', 'argv0', 'cmdline_sha256', 'cgroup_sha256',
                                   'parent', 'uids', 'gids', 'ppid'])
def test_any_protected_identity_drift_is_fatal(policy, field):
    identity = copy.deepcopy(policy['services'][-1]['identity'])
    identity[field] = 'changed'
    with pytest.raises(ValueError, match='unknown/restarted'):
        inspector.authorize_denied(identity['pid'], identity, ['maps'], policy)


@pytest.mark.parametrize('pid', [288084, 1234567])
def test_new_transient_or_prior_exited_pid_is_never_a_service_class_exception(policy, pid):
    identity = copy.deepcopy(policy['services'][-1]['identity'])
    identity['pid'] = pid
    with pytest.raises(ValueError, match='unknown/restarted'):
        inspector.authorize_denied(pid, identity, ['maps'], policy)


def test_new_denied_field_not_in_captured_scope_is_fatal(policy):
    policy = copy.deepcopy(policy)
    row = policy['services'][0]
    row['denied_fields'] = ['maps']
    with pytest.raises(ValueError, match='new protected field'):
        inspector.authorize_denied(row['identity']['pid'], row['identity'], ['maps', 'fd'], policy)


def fake_process_scan(monkeypatch, identity, observed, denied):
    process = types.SimpleNamespace(name=str(identity['pid']), stat=lambda: types.SimpleNamespace(st_uid=1000),
                                    exists=lambda: True)
    real_path = Path
    monkeypatch.setattr(inspector, 'Path', lambda p: types.SimpleNamespace(iterdir=lambda: [process])
                        if str(p) == '/proc' else real_path(p))
    monkeypatch.setattr(inspector, 'basic_identity', lambda _: (identity, 'S'))
    monkeypatch.setattr(inspector, 'service_identity', lambda _: copy.deepcopy(identity))
    monkeypatch.setattr(inspector, 'inspect_fields', lambda _: (observed, denied))


def test_known_service_readable_target_reference_is_not_waived(policy, monkeypatch):
    identity = policy['services'][0]['identity']
    fake_process_scan(monkeypatch, identity, {'fd': ['/owned/prepared/checkpoint (deleted)']}, ['maps'])
    with pytest.raises(ValueError, match='live same-UID reference'):
        inspector.inspect([Path('/owned/prepared')], policy)


def test_unknown_protected_python_aborts_without_exception(policy, monkeypatch):
    identity = {**policy['services'][0]['identity'], 'pid': 7654321, 'comm': 'python'}
    fake_process_scan(monkeypatch, identity, {}, ['maps'])
    with pytest.raises(ValueError, match='unknown/model/Python'):
        inspector.inspect([Path('/owned/prepared')], policy)


def test_same_comm_new_tailscale_parent_is_not_waived(policy, monkeypatch):
    identity = {**policy['services'][-1]['identity'], 'pid': 7654321}
    fake_process_scan(monkeypatch, identity, {}, list(inspector.FIELDS))
    with pytest.raises(ValueError, match='unknown/model/Python'):
        inspector.inspect([Path('/owned/prepared')], policy)


def test_exact_known_service_reports_truthful_limited_scope(policy, monkeypatch):
    identity = policy['services'][0]['identity']
    fake_process_scan(monkeypatch, identity, {}, list(inspector.FIELDS))
    result = inspector.inspect([Path('/owned/prepared')], policy)
    assert result['global_handle_absence_claimed'] is False
    assert result['unknown_protected_processes_ignored'] is False
    assert result['protected_infrastructure_exclusions'][0]['identity'] == identity


def test_actual_open_descriptor_remains_fatal(tmp_path, monkeypatch):
    target = tmp_path / 'opened'
    target.write_bytes(b'CPU test file')
    own = Path('/proc') / str(os.getpid())
    real_path = Path
    class ProcRoot:
        def iterdir(self):
            return iter([own])
        def __truediv__(self, name):
            return real_path('/proc') / name
    monkeypatch.setattr(inspector, 'Path', lambda p: ProcRoot() if str(p) == '/proc' else real_path(p))
    with target.open('rb'), pytest.raises(ValueError, match='live same-UID reference'):
        inspector.inspect([target], {'services': []})


@pytest.mark.parametrize('module', [queue, archive])
def test_inspector_source_rejected_before_import(module, tmp_path):
    source = tmp_path / 'wrong.py'
    source.write_text("raise RuntimeError('must never execute')\n")
    with pytest.raises(ValueError, match='unexpected handle inspector'):
        module.handle_authorization({'handle_inspector': {'path': str(source), 'sha256': sha(source)}})


@pytest.mark.parametrize('module', [queue, archive])
def test_snapshot_bytes_rejected_before_host_policy(module, tmp_path):
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(SNAPSHOT.read_text() + ' ')
    source = HERE / 'protected_process_inspector_v2.py'
    with pytest.raises(ValueError, match='bound.*input changed'):
        module.handle_authorization({'handle_inspector': {'path': str(source.absolute()), 'sha256': sha(source)},
                                     'protected_service_manifest': {'path': str(snapshot), 'sha256': ACTUAL_SHA}})


def functions(name):
    return {n.name: n for n in ast.parse((HERE / name).read_text()).body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize('old,new,changed', [
    ('run_queue_v1.py', 'run_queue_v2.py', {'load_config', 'same_uid_handles', 'reclaim_edge'}),
    ('archive_inactive_v5_cache_v1.py', 'archive_inactive_v5_cache_v2.py', {'load_config', 'inactive_handles', 'execute'}),
])
def test_native_and_archival_functions_remain_exact(old, new, changed):
    before, after = functions(old), functions(new)
    assert set(after) - set(before) == {'handle_authorization'}
    for name in set(before) - changed:
        assert ast.dump(before[name]) == ast.dump(after[name]), name
    # The two file-operation methods differ only by passing explicit config to inspection.
    target = 'reclaim_edge' if new.startswith('run_queue') else 'execute'
    class RemoveConfig(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in {'same_uid_handles', 'inactive_handles'}:
                assert len(node.args) == 2 and isinstance(node.args[-1], ast.Name) and node.args[-1].id == 'c'
                node.args.pop()
            return node
    assert ast.dump(before[target]) == ast.dump(RemoveConfig().visit(after[target]))


def test_native_command_schedule_is_unchanged():
    old = load('run_queue_v1')
    config = json.loads((HERE / 'config_template_v1.json').read_text())
    assert old.commands(config) == queue.commands(config)
    assert len(queue.commands(config)) == 7


def test_mixed_readable_and_denied_fd_preserves_target_reference(policy, monkeypatch):
    class FD:
        def __init__(self, value=None):
            self.value = value
        def readlink(self):
            if self.value is None:
                raise PermissionError('synthetic protected descriptor')
            return Path(self.value)
    class Process:
        def __truediv__(self, field):
            if field == 'fd':
                return types.SimpleNamespace(iterdir=lambda: iter([FD('/owned/cache/file'), FD(), FD('/elsewhere')]))
            raise PermissionError('synthetic protected field')
    observed, denied = inspector.inspect_fields(Process())
    assert observed['fd'] == ['/owned/cache/file', '/elsewhere']
    assert denied == list(inspector.FIELDS)
    identity = policy['services'][0]['identity']
    fake_process_scan(monkeypatch, identity, observed, denied)
    with pytest.raises(ValueError, match='live same-UID reference'):
        inspector.inspect([Path('/owned/cache')], policy)
