"""Fingerprint code AND native binaries before a multi-model regression.

Checkpoint tensors and external vendor environments have separate inventories.
This utility neither loads native code nor imports Torch/CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SUFFIXES = {'.py', '.pyi', '.cu', '.cpp', '.cc', '.c', '.h', '.hpp',
            '.yaml', '.yml', '.toml', '.so', '.pyd', '.dll'}


def _native_binary(path):
    return path.suffix in {'.pyd', '.dll'} or bool(re.search(r'\.so(?:\.\d+)*$', path.name))


def build_manifest(root: Path, *, required_binaries=()):
    root = root.resolve()
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and (path.suffix in SUFFIXES or _native_binary(path)) and '__pycache__' not in path.parts:
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    matches = {}
    for pattern in required_binaries:
        paths = [p for p in root.glob(pattern) if p.is_file() and _native_binary(p)]
        if not paths:
            raise ValueError(f'Missing required native artifact: {pattern}')
        names = [str(p.relative_to(root)) for p in sorted(paths)]
        if not all(name in files for name in names):
            raise ValueError('Required artifacts must be inside the inventoried bundle')
        matches[pattern] = names
    return dict(schema=1, files=files, required_binaries=matches)


def verify_manifest(root: Path, manifest):
    current = build_manifest(root, required_binaries=manifest['required_binaries'])
    if current != manifest:
        raise ValueError('Runtime bundle changed (source or native binary inventory)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--require-binary', action='append', default=[])
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify_manifest(args.root, json.loads(args.manifest.read_text()))
    else:
        if args.manifest.exists():
            parser.error('Refusing to replace an existing bundle manifest')
        args.manifest.write_text(json.dumps(build_manifest(args.root,
            required_binaries=args.require_binary), indent=2)+'\n')


if __name__ == '__main__':
    main()
