"""Portable complete-run evidence bundles; preserve original plan bytes and identities."""
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from .coverage import run_evidence
from .registry import load_registry
from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic


def export_bundle(run, registry, output):
    root, output = Path(run).resolve(), Path(output).resolve()
    run_evidence(root)
    plan = load_json(root/'plan.json')
    if registry.digest != plan['registry_sha256']:
        raise ConfigurationError('export registry differs from the frozen plan')
    if output.exists():
        raise ConfigurationError('refusing to overwrite an evidence bundle')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ifl-evidence-',dir=output.parent) as temporary:
        stage = Path(temporary)/'bundle'
        stage.mkdir()
        for name in ('plan.json','report.json','environment.json','run_manifest.json','progress.json'):
            source = root/name
            if source.is_file():shutil.copy2(source,stage/name)
        for name in ('results','requests','logs','failures'):
            if (root/name).is_dir():shutil.copytree(root/name,stage/name)
        write_json_atomic(stage/'registry.json',registry.raw)
        scenes = {}
        for arm in plan['arms']:
            ref = arm['operating_point'].get('scene_manifest')
            if not ref:continue
            original = Path(ref['path'])
            if sha256_file(original) != ref['sha256']:
                raise ConfigurationError('frozen scene artifact changed before export')
            relative = f"artifacts/scenes/{ref['sha256']}.json"
            target = stage/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,target)
            scenes[ref['path']] = relative
        files = {str(p.relative_to(stage)):sha256_file(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest = {'schema_version':1,'plan_id':plan['plan_id'],'files':files,'scene_path_mapping':scenes,
                    'scope':'Portable analysis evidence. Original absolute paths remain in the immutable plan; replay requires restoring dependencies and remapping into a fresh plan.'}
        manifest['bundle_sha256'] = sha256_json(manifest)
        write_json_atomic(stage/'bundle.json',manifest)
        stage.rename(output)
    return verify_bundle(output)


def _verify_scene_artifacts(plan, manifest):
    expected = {}
    for arm in plan['arms']:
        ref = arm['operating_point'].get('scene_manifest')
        if not ref:
            continue
        previous = expected.setdefault(ref['path'], ref['sha256'])
        if previous != ref['sha256']:
            raise ConfigurationError('one original scene path has conflicting frozen hashes')
    mapping = manifest.get('scene_path_mapping', {})
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        raise ConfigurationError('bundle scene mapping does not cover the frozen plan')
    for original, digest in expected.items():
        relative = mapping[original]
        if not isinstance(relative, str) or manifest['files'].get(relative) != digest:
            raise ConfigurationError('bundled scene artifact differs from its frozen plan hash')


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = load_json(root/'bundle.json')
    unsigned = dict(manifest)
    claimed = unsigned.pop('bundle_sha256',None)
    if manifest.get('schema_version') != 1 or claimed != sha256_json(unsigned):
        raise ConfigurationError('invalid bundle manifest identity')
    paths = manifest['files']
    for name, digest in paths.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ConfigurationError('bundle contains an unsafe relative path')
        target = root/name
        if not target.is_file() or sha256_file(target) != digest:
            raise ConfigurationError(f'bundle file missing or changed: {name}')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(paths)|{'bundle.json'}:
        raise ConfigurationError('bundle file inventory changed')
    plan = load_json(root/'plan.json')
    _verify_scene_artifacts(plan, manifest)
    registry = load_registry(root/'registry.json')
    if manifest['plan_id'] != plan['plan_id'] or registry.digest != plan['registry_sha256']:
        raise ConfigurationError('bundle plan/registry mismatch')
    evidence = run_evidence(root)
    return {'valid':True,'bundle':str(root),'bundle_sha256':claimed,'files':len(paths),'evidence':evidence}
