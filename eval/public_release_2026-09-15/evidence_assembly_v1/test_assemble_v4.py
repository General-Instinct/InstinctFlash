import importlib.util
import json
from pathlib import Path
import shutil

import pytest


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('evidence_assembly_v4', HERE / 'assemble_v4.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ORIGINAL_STUDY = m.STUDY


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def test_original_mapping_preserves_failed_nano_and_legacy_parity_gate():
    roots, binding = m.load_selection(m.STUDY / 'qualification')
    assert roots['nano'] == m.STUDY / 'qualification/nano'
    assert binding['overrides'] == {}
    with pytest.raises(ValueError, match='historical actions differ: nano-runtime_selected'):
        m.validate_historical(roots['nano'], m.REPO / 'eval/user_e2e_2026-09-14')
    assert m.sha(HERE / 'assemble_v3.py') == m.V3_ASSEMBLER_SHA


def test_actual_original_68_evidence_files_still_match_frozen_inventory():
    result = m.preserve_original_evidence(
        m.STUDY / 'qualification',
        m.ref(HERE / 'stage_v7_partial_v1/copied_evidence_inventory.json'))
    assert result['original_files_rehashed'] == 68


@pytest.fixture
def selected(tmp_path, monkeypatch):
    study = tmp_path / 'study'
    q = study / 'qualification'
    inv = HERE / 'stage_v7_partial_v1/copied_evidence_inventory.json'
    dest = study / 'evidence_assembly_v1/stage_v7_partial_v1/copied_evidence_inventory.json'
    dest.parent.mkdir(parents=True)
    shutil.copyfile(inv, dest)
    for relative in m.read(inv):
        if relative.split('/')[0] in {'edge', 'nano'}:
            target = q / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ORIGINAL_STUDY / 'qualification' / relative, target)
    monkeypatch.setattr(m, 'STUDY', study)
    dependency = study / 'nano_compiler_diagnostic_v1/unit_dependency.json'
    write(dependency, {'scope': 'CPU selection unit fixture, not native evidence'})
    d = ORIGINAL_STUDY / 'nano_compiler_diagnostic_v1'
    comparison = dependency.parent / 'comparison.json'
    write(comparison, {
        'schema': 'instinctflash.nano_compiler_diagnostic_comparison.v1',
        'status': 'diagnostic_comparison_complete',
        'source': m.ref(d / 'analyze_compiler_diagnostic_v1.py'),
        'worker': m.ref(d / 'run_compiler_diagnostic_v1.py'),
        'compiler_binding': m.ref(d / 'compiler_binding_v1.json'),
        **{key: m.ref(dependency) for key in ('current', 'previous_control', 'external_completion', 'queue_config')},
        'historical_action_bytes_recovered_on_tested_inputs': True,
        'actions': {'corrected_vs_historical': {'exact_bytes': True}},
        'task_quality_certified': False,
        'normal_benchmark_or_serving_qualified': False,
    })
    overrides = {}
    for family in ('edge', 'nano'):
        root = q / family / 'pytorch_triton_v1'
        for name in m.SELECTION_FILES:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(q / family / name, target)
        for name in ('run/run.json', 'serving/receipt.json'):
            data = m.read(root / name)
            data['interpreter'] = m.CORRECTED_PYTHON
            write(root / name, data)
        overrides[family] = {'root': f'{family}/pytorch_triton_v1',
                             'files': {name: m.sha(root / name) for name in m.SELECTION_FILES}}
    manifest = study / 'selection.json'
    write(manifest, {'schema': 'instinctflash.public_reproduction_selection.v1',
                     'template_only': False, 'qualification_root': str(q),
                     'preserved_evidence_inventory': m.ref(dest),
                     'compiler_diagnostic': m.ref(comparison), 'overrides': overrides})
    return q, manifest


def test_explicit_two_nested_roots_preserve_originals_and_never_scan_alternatives(selected):
    q, manifest = selected
    old = m.sha(q / 'nano/historical_action_comparison_v1.json')
    roots, binding = m.load_selection(q, manifest, m.sha(manifest))
    assert roots['nano'] == q / 'nano/pytorch_triton_v1'
    assert roots['edge'] == q / 'edge/pytorch_triton_v1'
    assert roots['pi05'] == q / 'pi05'
    assert binding['preserved_evidence']['original_files_rehashed'] == 68
    assert m.sha(q / 'nano/historical_action_comparison_v1.json') == old
    assert not (roots['nano'] / 'historical_action_comparison_v1.json').exists()


def test_single_completed_family_selection_leaves_other_original_pending(selected):
    q, manifest = selected
    data = m.read(manifest)
    del data['overrides']['nano']
    write(manifest, data)
    roots, _ = m.load_selection(q, manifest, m.sha(manifest))
    assert roots['nano'] == q / 'nano'
    assert roots['edge'] == q / 'edge/pytorch_triton_v1'


@pytest.mark.parametrize('change,match', [
    ('old_evidence', 'original evidence changed'),
    ('new_evidence', 'selected evidence changed'),
    ('template', 'not an actual execution binding'),
    ('wrong_root', 'declared additive nested root'),
    ('wrong_family', 'only explicit Edge/Nano'),
    ('missing_binding', 'artifact coverage differs'),
    ('failed_run', 'normal benchmark/serving has not passed'),
    ('wrong_compiler_env', 'corrected compiler environment'),
    ('diagnostic_not_recovered', 'did not recover historical'),
    ('diagnostic_dependency', 'bound evidence changed'),
])
def test_selection_rejects_changed_or_incomplete_proofs(selected, change, match):
    q, manifest = selected
    data = m.read(manifest)
    if change == 'old_evidence':
        (q / 'nano/run/run.json').write_text('{}\n')
    elif change == 'new_evidence':
        (q / 'nano/pytorch_triton_v1/run/run.json').write_text('{}\n')
    elif change == 'template':
        data['template_only'] = True
    elif change == 'wrong_root':
        data['overrides']['nano']['root'] = 'nano/another_success'
    elif change == 'wrong_family':
        data['overrides'] = {'pi05': data['overrides']['nano']}
    elif change == 'missing_binding':
        del data['overrides']['nano']['files']['serving/receipt.json']
    elif change in {'failed_run', 'wrong_compiler_env'}:
        path = q / 'nano/pytorch_triton_v1/run/run.json'
        row = m.read(path)
        row['status' if change == 'failed_run' else 'interpreter'] = 'failed' if change == 'failed_run' else '/old/env/bin/python'
        write(path, row)
        data['overrides']['nano']['files']['run/run.json'] = m.sha(path)
    elif change == 'diagnostic_not_recovered':
        path = Path(data['compiler_diagnostic']['path'])
        row = m.read(path)
        row['historical_action_bytes_recovered_on_tested_inputs'] = False
        write(path, row)
        data['compiler_diagnostic'] = m.ref(path)
    else:
        path = Path(data['compiler_diagnostic']['path'])
        Path(m.read(path)['current']['path']).write_text('{}\n')
    write(manifest, data)
    with pytest.raises(ValueError, match=match):
        m.load_selection(q, manifest, m.sha(manifest))


def test_manifest_hash_is_checked_before_proof_parsing(selected):
    q, manifest = selected
    expected = m.sha(manifest)
    manifest.write_text('not json')
    with pytest.raises(ValueError, match='explicit selection hash differs'):
        m.load_selection(q, manifest, expected)


def test_source_auditor_covers_nested_cells_without_old_duplicate(tmp_path, monkeypatch):
    path = ORIGINAL_STUDY / 'qualified_stage_correspondence_v1/audit_v3.py'
    auditor = m.load_module(path, m.CORRESPONDENCE_SHA, 'test_nested_source_auditor')
    monkeypatch.setattr(auditor, 'EVIDENCE', tmp_path)
    roots = {family: tmp_path / 'qualification' / family for family in m.FAMILIES}
    for family in ('edge', 'nano'):
        old = roots[family] / 'run/cells' / f'{family}-runtime_selected/receipt.json'
        write(old, {'scope': 'unselected unit fixture'})
        roots[family] = roots[family] / 'pytorch_triton_v1'
        for arm in ('eager_native', 'runtime_default', 'runtime_selected'):
            write(roots[family] / 'run/cells' / f'{family}-{arm}/receipt.json', {})
    result = auditor.selected_receipts(roots)
    assert len(result) == 6
    assert all('pytorch_triton_v1' in p.parts for p in result)
    roots['nano'] = roots['nano'].parent / 'unapproved'
    with pytest.raises(ValueError, match='unapproved nested qualification root'):
        auditor.selected_receipts(roots)


@pytest.mark.parametrize('difference', ['none', 'value', 'dtype', 'shape'])
def test_nested_comparator_reports_real_byte_dtype_shape_deltas_without_waiver(tmp_path, monkeypatch, difference):
    import numpy as np

    spec = importlib.util.spec_from_file_location('nested_comparator', HERE / 'compare_main_historical_v2.py')
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    root = tmp_path / 'qualification/nano/pytorch_triton_v1'
    old = tmp_path / 'eval/user_e2e_2026-09-14/cells/nano-runtime_selected/receipt.json'
    current = root / 'run/cells/nano-runtime_selected/receipt.json'
    a = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
    b = a.copy()
    if difference == 'value':
        b[0, 0, 0] += .5
    elif difference == 'dtype':
        b = b.astype(np.float64)
    elif difference == 'shape':
        b = b.reshape(1, 2, 4)
    row = {'ok': True, 'cases': 1, 'model_id': 'pinned', 'revision': 'pinned',
           'precision': 'native', 'effective_schedule': {'steps': 4}}
    for path, actions in ((old, a), (current, b)):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.parent / 'receipt.npz', actions=actions)
        write(path, dict(row, actions_sha256=m.sha(path.parent / 'receipt.npz')))
    cell = {'id': 'nano-runtime_selected', 'family': 'nano'}
    write(root / 'run/matrix.json', {'cells': [cell]})
    write(root / 'run/run.json', {'status': 'passed', 'report_status': 'passed',
                                'attempts': [{'cell': cell['id'], 'exit_code': 0}]})
    write(tmp_path / 'eval/user_e2e_2026-09-14/matrix_final.json', {'scope': 'unit fixture'})
    monkeypatch.setattr(m, 'REPO', tmp_path)
    monkeypatch.setattr(m, 'historical_receipt', lambda *_: old)
    rows = comparator.compare(m, 'nano', root, {'selection': {'scope': 'unit'}}, HERE / 'assemble_v4.py')
    assert len(rows) == 1
    assert rows[0]['historical_actions_match_bytes'] is (difference == 'none')
    assert rows[0]['same_dtype'] is (difference != 'dtype')
    assert rows[0]['same_shape'] is (difference != 'shape')
    assert rows[0]['max_abs'] == (None if difference == 'shape' else .5 if difference == 'value' else 0.0)
    assert not (root / 'historical_action_comparison_v1.json').exists()
