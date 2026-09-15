"""Independent read-only source/native correspondence and portable-plan audit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlparse
import zipfile

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / 'eval/public_release_2026-09-15'
STAGE2 = Path('/home/ubuntu/ifl-public-full-stage-20260915-v2/wheels')
CORE2 = Path('/home/ubuntu/ifl-public-core-reproduce-20260915-v2/wheels/instinctflash-0.1.0-py3-none-any.whl')
NATIVE = EVIDENCE / 'thor_native_build_v1/raw_outputs'
FAMILIES = {'pi05', 'groot', 'va', 'vla4', 'vla2', 'edge', 'nano', 'dreamzero'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')


def wheel(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'duplicate wheel member')
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in names if not n.endswith('/')}


def payload(rows):
    return {n: h for n, h in rows.items() if '.dist-info/' not in n}


def delta(before, after):
    return {'changed': sorted(n for n in before.keys() & after.keys() if before[n] != after[n]),
            'added': sorted(after.keys() - before.keys()), 'removed': sorted(before.keys() - after.keys())}


def strip_remedy(value):
    if isinstance(value, dict):
        return {k: strip_remedy(v) for k, v in value.items() if k != 'remedy'}
    if isinstance(value, list):
        return [strip_remedy(v) for v in value]
    return value


def selected_receipts(selected_roots=None):
    qualification = EVIDENCE / 'qualification'
    if selected_roots is None:
        return sorted(qualification.glob('*/run/cells/*/receipt.json'))
    require(set(selected_roots) == FAMILIES, 'source audit requires all eight explicit family roots')
    receipts = []
    for family, root in sorted(selected_roots.items()):
        root = Path(root).resolve()
        allowed = {(qualification / family).resolve()}
        if family in {'edge', 'nano'}:
            allowed.add((qualification / family / 'pytorch_triton_v1').resolve())
        require(root in allowed, 'unapproved nested qualification root')
        receipts.extend(sorted((root / 'run/cells').glob('*/receipt.json')))
    return receipts


def audit(stage, output, selected_roots=None, selection_binding=None):
    require(sha(Path(__file__).with_name('audit_v2.py')) == 'ade37fc6c6751f0eddca9a6881a7feb099652cc52ecf09d5dd534bf5d8a65abd', 'preserved source auditor V2 changed')
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(stage / 'manifest.json')
    source = stage / 'source'
    refs = {'stage_manifest': {'path': str(stage / 'manifest.json'), 'sha256': sha(stage / 'manifest.json')},
            'core_reproduce_v2': {'path': str(CORE2), 'sha256': sha(CORE2)},
            'qualification_stage2_core': {'path': str(STAGE2 / 'instinctflash-0.1.0-py3-none-any.whl'),
                                          'sha256': sha(STAGE2 / 'instinctflash-0.1.0-py3-none-any.whl')}}
    archives = {}
    package_roots = {module: directory for _, (directory, module) in manifest['packages'].items()}
    for item in manifest['wheels']:
        p = Path(item['path'])
        require(p.parent == stage / 'wheels' and sha(p) == item['sha256'], 'stage wheel differs from manifest')
        archives[p.name] = wheel(p)
    qualified = {}
    for p in STAGE2.glob('*.whl'):
        if p.name.startswith(('instinctflash-', 'flash_rt-')):
            continue
        difference = delta(payload(wheel(p)), payload(archives[p.name]))
        require(not any(difference.values()), f'adapter payload drift: {p.name}')
        qualified[p.name] = {'old_sha256': sha(p), 'new_sha256': sha(stage / 'wheels' / p.name),
                             'payload_files_identical': len(payload(wheel(p))), 'delta': difference}
    core_final = archives['instinctflash-0.1.0-py3-none-any.whl']
    old_core = wheel(CORE2)
    core_delta = delta(payload(old_core), payload(core_final))
    stage2_core_delta = delta(payload(wheel(STAGE2 / 'instinctflash-0.1.0-py3-none-any.whl')), payload(core_final))
    inference_roots = ('instinctflash/runtime/', 'instinctflash/adapters/', 'instinctflash/backends/',
                       'instinctflash/planners/', 'instinctflash/passes/', 'instinctflash/descriptors/', 'instinctflash/native/')
    require(not [p for values in core_delta.values() for p in values if p.startswith(inference_roots)],
            'qualified inference source changed')
    require(not [p for values in stage2_core_delta.values() for p in values if p.startswith(inference_roots)],
            'stage2 qualified inference source changed')
    native_wheel = next((NATIVE / 'wheels').glob('*.whl'))
    require(sha(native_wheel) == '3262c54f7094dd3b77ef801e7208b157d891d8f50240374ed4f363c7f13e3c7f', 'native wheel identity differs')
    native_rows = wheel(native_wheel)
    require(native_rows == read(NATIVE / 'wheel_inventory.json'), 'native wheel members differ from build inventory')
    pure = archives['flash_rt-0.1.0-py3-none-any.whl']
    pure_payload = payload(pure)
    require(not [n for n in pure_payload if n.endswith('.so')], 'pure wheel contains an undeclared native library')
    require(all(native_rows[n] == h for n, h in pure_payload.items()), 'native and final Python frontend differ')
    native_library_rows = read(NATIVE / 'native_libraries.json')
    native_libs = {n: v['sha256'] for n, v in native_library_rows.items()}
    bf16 = NATIVE / 'native/libinstinctflash_bf16.so'
    require(sha(bf16) == native_libs['libinstinctflash_bf16.so'], 'standalone BF16 artifact differs')
    native_inputs = []
    for p, prefix in [(NATIVE / 'plan.json', ''), (EVIDENCE / 'thor_fa2_build_v1/plan.json', 'serving')]:
        plan = read(p)
        checked, changes = [], []
        for name, expected in plan['source_files'].items():
            rel = str(Path(prefix) / name)
            actual = sha(source / rel)
            checked.append({'path': rel, 'qualified_sha256': expected, 'stage_sha256': actual})
            if actual != expected:
                changes.append(rel)
        require(set(changes).issubset({'serving/README.rst'}), 'native build implementation changed')
        native_inputs.append({'plan': str(p), 'plan_sha256': sha(p), 'files': checked,
                              'nonimplementation_changes': changes})
    save(output / 'native_build_source_mapping.json', native_inputs)
    captured = []
    for p in selected_receipts(selected_roots):
        receipt = read(p)
        run_path = p.parents[2] / 'run.json'
        run = read(run_path)
        attempts = [a for a in run['attempts'] if a['cell'] == receipt['cell_id']]
        require(receipt['ok'] is True and len(attempts) == 1 and attempts[0]['exit_code'] == 0,
                'nonqualified attempt in passed run inventory')
        matches, external = [], []
        for installed, expected in receipt['sources'].items():
            if '/site-packages/' in installed:
                rel = installed.split('/site-packages/', 1)[1]
                module = rel.split('/')[0]
                if module in package_roots or module == 'benchmarks':
                    if rel.endswith('.so'):
                        actual = native_rows.get(rel)
                        origin = str(native_wheel) + '::' + rel
                    else:
                        local = source / (package_roots.get(module, '.') or '.') / rel
                        actual = sha(local) if local.is_file() else None
                        origin = str(local)
                    require(actual == expected, f'actual installed implementation differs: {installed}')
                    matches.append({'installed': installed, 'release_source': origin, 'sha256': actual})
                    continue
            if Path(installed).name in native_libs:
                require(expected == native_libs[Path(installed).name], 'actual native C library differs')
                matches.append({'installed': installed, 'release_artifact': str(bf16), 'sha256': expected})
            else:
                external.append({'installed': installed, 'sha256': expected})
        captured.append({'receipt': str(p), 'sha256': sha(p), 'run_receipt': str(run_path), 'run_sha256': sha(run_path),
                         'cell': receipt['cell_id'], 'family': receipt['family'], 'matches': matches,
                         'external_vendor_or_dependency_sources': external})
    save(output / 'actual_qualified_source_mapping.json', captured)
    with zipfile.ZipFile(CORE2) as z:
        prior_profiles = json.loads(z.read('benchmarks/regression/fixtures/deployment_profiles.json'))
    current_profiles = read(source / 'benchmarks/regression/fixtures/deployment_profiles.json')
    previous = {r['id']: r for r in prior_profiles['models']}
    require({r['id'] for r in current_profiles['models']} == FAMILIES, 'incorrect eight-family profile set')
    profile_rows = []
    env = {k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'}}
    env['CUDA_VISIBLE_DEVICES'] = ''
    for profile in current_profiles['models']:
        family = profile['id']
        require(profile['checkpoint'] == previous[family]['checkpoint'], 'checkpoint changed')
        require(strip_remedy(profile['execution_modes']) == strip_remedy(previous[family]['execution_modes']),
                'runtime precision/schedule/environment changed')
        process = subprocess.run([sys.executable, str(source / 'scripts/bootstrap_vendor.py'), 'plan', family,
                                  '--checkout', str(source)], env=env, text=True, capture_output=True, timeout=30)
        require(process.returncode == 0, 'public bootstrap plan failed')
        install_plan = json.loads(process.stdout)
        save(output / (family + '_bootstrap_plan.json'), install_plan)
        files = [source / 'release/vendor' / install_plan[k]['path'] for k in ('requirements', 'constraints')]
        texts = [p.read_text() for p in files]
        texts.append(json.dumps(install_plan))
        require(not any(marker in text for text in texts for marker in
                        ('InstinctFlash-internal', 'git@github.com', 'ssh://', 'file://', '/home/guanming', '/home/ubuntu', '/workspace/')),
                'private source/path dependency in public install plan')
        if install_plan.get('source'):
            upstream = install_plan['source']['source']
            url = urlparse(upstream['repository'])
            require(url.scheme == 'https' and url.hostname == 'github.com' and url.username is None
                    and re.fullmatch('[0-9a-f]{40}', upstream['revision']), 'source URL not public pinned HTTPS')
        profile_rows.append({'id': family, 'checkpoint': profile['checkpoint'], 'execution_modes_unchanged': True,
                             'public_bootstrap_plan': family + '_bootstrap_plan.json',
                             'public_source': install_plan.get('source', {}).get('source'),
                             'source_code_private_paths_found': False})
    process = subprocess.run([sys.executable, str(source / 'scripts/public_deploy.py'), 'plan', 'all'],
                             text=True, capture_output=True, env=env, timeout=30)
    require(process.returncode == 0, 'public deployment planning failed')
    deploy = json.loads(process.stdout)
    require(deploy['ok'] and {r['id'] for r in deploy['results']} == FAMILIES, 'deployment plan coverage differs')
    save(output / 'all_public_deployment_plans.json', deploy)
    guide = (source / 'serving/README.rst').read_text()
    base = 'https://github.com/General-Instinct/InstinctFlash/releases/download/thor-2026-09-15/'
    require(base + native_wheel.name + '#sha256=' + sha(native_wheel) in guide, 'native public wheel URL/hash does not bind actual artifact')
    require(base + bf16.name in guide and sha(bf16) in guide, 'BF16 public URL/hash does not bind actual artifact')
    result = {'kind': 'qualified_public_stage_correspondence_v1', 'status': 'passed_source_and_plan_correspondence',
              'auditor_sha256': sha(__file__), 'time': time.time(), 'stage': str(stage), 'refs': refs,
              'stage_wheels': [{'path': str(stage / 'wheels' / n), 'sha256': sha(stage / 'wheels' / n)} for n in archives],
              'adapter_payloads': qualified, 'core_reproduce_v2_to_stage_delta': core_delta,
              'qualification_stage2_core_to_stage_delta': stage2_core_delta,
              'older_core_changes_outside_inference_scope': {
                  'distillation': [p for p in stage2_core_delta['changed'] if p.startswith('instinctflash/distill/')],
                  'benchmark_or_serving_harness': [p for p in stage2_core_delta['changed'] if p.startswith('benchmarks/')],
                  'classification': 'These helpers have separate functional evidence; byte correspondence alone does not qualify their changed behavior.'},
              'inference_source_semantic_drift_detected': False,
              'selection_binding': selection_binding,
              'selected_qualification_roots': {family: str(root) for family, root in sorted((selected_roots or {}).items())},
              'actual_passed_cells': [{'cell': r['cell'], 'receipt': r['receipt'], 'matched_internal_source_files': len(r['matches']),
                                       'external_source_files_disclosed': len(r['external_vendor_or_dependency_sources'])} for r in captured],
              'families_without_passed_local_GPU_receipts_at_snapshot': sorted(FAMILIES - {r['family'] for r in captured}),
              'native': {'wheel_sha256': sha(native_wheel), 'BF16_sha256': sha(bf16),
                         'final_pure_wheel_native_libraries': 0, 'native_wheel_extra_payloads': sorted(payload(native_rows).keys() - pure_payload.keys()),
                         'frontend_python_and_configuration_common_members_identical': True},
              'eight_model_plans': profile_rows,
              'public_native_assets': {'urls': [base + native_wheel.name, base + bf16.name],
                                       'checksums_match_actual_artifacts': True, 'network_availability_verified': False,
                                       'publication_status': 'Root reports release tag/assets not yet published; upload and anonymous availability check remain required.'},
              'requires_GPU_requalification_due_to_detected_runtime_drift': False,
              'limitations': ['Only persisted passed local GPU cell receipts are mapped; missing/in-progress/failed cells are not promoted.',
                             'This is a byte/source audit, not a new speed, numerical or task-quality certification.',
                             'External vendor/dependency files remain explicitly separate; public pinned source plans are checked but no new environment install occurs.',
                             'Per-cell imported-source inventories are not complete wheel attestations. Supplied qualification wheel artifacts are independently diffed.',
                             'Final core changes in deployment metadata, distillation or evaluation helpers are not inference implementation changes and need their own CPU/functional checks.',
                             'GPU ABI portability beyond the declared CP312/aarch64/SM110 stack is not established; Cosmos uses the separate C library.',
                             'No network/download/install/GPU or environment mutation was performed by this audit.']}
    result['evidence'] = {p.name: sha(p) for p in output.glob('*.json')}
    save(output / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.stage.resolve(), args.output.resolve())
    print(json.dumps({'status': result['status'], 'actual_passed_cells': len(result['actual_passed_cells']),
                      'stage': result['stage'], 'receipt': str(args.output / 'receipt.json')}))
