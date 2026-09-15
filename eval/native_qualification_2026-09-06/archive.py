"""Archive raw evidence and frozen source, excluding model weights and generated caches."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


DIRECTORIES = ('execution', 'final-execution', 'va-execution', 'va-final-execution',
               'worker-execution', 'worker-final-execution', 'worker-final2-execution',
               'worker-final3-execution', 'trace0', 'trace1', 'trace2', 'trace3',
               'thor-results', 'serialization-pilot', 'thor-budget-checks', 'inventory',
               'release-wheels')


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        h.update(block)
    return h.hexdigest()


def archive(root, output):
    if output.exists() or (root / 'archive-files.json').exists():
        raise ValueError('refusing to overwrite an archive or its inventory')
    files = {p for pattern in ('*.json', '*.log', '*.py', '*.npz')
             for p in root.glob(pattern) if p.is_file()}
    for directory in DIRECTORIES:
        files.update(p for p in (root / directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and '.git' not in p.parts)
    inventory = []
    for p in sorted(files):
        with p.open('rb') as stream:
            inventory.append({'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
                              'sha256': digest(stream)})
    manifest = root / 'archive-files.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'files': inventory,
        'scope': 'Raw evidence and source. Model weights, caches and prior external bundles are not copied.'}, indent=2) + '\n')
    with tarfile.open(output, 'w:gz', dereference=True) as tar:
        for p in sorted(files | {manifest}):
            tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
    with tarfile.open(output, 'r:gz') as tar:
        expected = {r['path']: r['sha256'] for r in inventory}
        assert len(tar.getmembers()) == len(expected) + 1
        for name, sha in expected.items():
            with tar.extractfile(name) as stream:
                assert digest(stream) == sha, name
    with output.open('rb') as stream:
        sha = digest(stream)
    result = {'path': str(output.resolve()), 'bytes': output.stat().st_size,
              'sha256': sha, 'verified_files': len(inventory), 'inventory': str(manifest)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive(args.root, args.output)
