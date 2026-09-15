"""Freeze complete local asset roots and bind unchanged native remote asset URLs.

Remote copies are evidence, never renderer substitutions. The driver re-reads
the original URLs with Isaac's own omni.client and compares their bytes to the
frozen copies. Run hydration before freezing, never during a paired episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .util import ConfigurationError, load_json, sha256_file


def remote_url(value: str) -> bool:
    return urlsplit(value).scheme.lower() in ("http", "https", "omniverse")


def _validate_url(url: str) -> None:
    parts = urlsplit(url)
    if (not remote_url(url) or not parts.netloc or parts.username or parts.password
            or parts.query or parts.fragment):
        raise ConfigurationError("asset URL must be an explicit credential-free resource URL")


def _verify_root(root_spec: dict, *, paths: set[str]) -> list[dict]:
    files = root_spec.get("files")
    if not isinstance(files, dict) or not files:
        raise ConfigurationError("asset inventory is empty")
    root = Path(root_spec["root"]).resolve(strict=True)
    if not root.is_dir():
        raise ConfigurationError("asset manifest root is not a directory")
    if any(not isinstance(relative, str) or Path(relative).is_absolute()
           or ".." in Path(relative).parts for relative in files):
        raise ConfigurationError("asset manifest path escapes its root")
    # Symlinked directories otherwise disappear from rglob's traversal and can
    # conceal an unlisted texture tree. Asset roots are deliberately explicit.
    if any(path.is_symlink() and path.is_dir() for path in root.rglob("*")):
        raise ConfigurationError("asset root contains a symlinked directory")
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual_files != set(files):
        raise ConfigurationError("asset manifest does not cover the complete actual asset root")
    verified = []
    for relative, expected_sha in files.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ConfigurationError("asset manifest path escapes its root")
        if not path.is_file() or str(path) in paths:
            raise ConfigurationError("asset paths must identify unique local files")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ConfigurationError("invalid asset digest")
        if sha256_file(path) != expected_sha:
            raise ConfigurationError("asset file hash differs from the frozen inventory")
        paths.add(str(path))
        verified.append({"path": str(path), "sha256": expected_sha, "bytes": path.stat().st_size})
    return verified


def verify_asset_inventory(manifest: dict) -> dict:
    """Keep schema 1 unchanged; schema 2 adds disjoint roots and exact URL bindings."""
    version = manifest.get("schema_version")
    if version == 1:
        if not isinstance(manifest.get("files"), dict):
            raise ConfigurationError("asset manifest requires schema_version=1, root, and relative file hashes")
        return {"schema_version": 1,
                "files": sorted(_verify_root(manifest, paths=set()), key=lambda row: row["path"])}
    if version != 2 or not isinstance(manifest.get("roots"), dict) or not manifest["roots"]:
        raise ConfigurationError("asset manifest requires schema_version=1 or 2 with complete roots")
    roots = manifest["roots"]
    paths: set[str] = set()
    files = []
    root_paths: dict[str, Path] = {}
    for name, root in sorted(roots.items()):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ConfigurationError("asset root name must be a stable identifier")
        path = Path(root["root"]).resolve(strict=True)
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in root_paths.values()):
            raise ConfigurationError("asset roots must be disjoint")
        root_paths[name] = path
        files.extend(_verify_root(root, paths=paths))
    expected = {row["path"]: row for row in files}
    bindings = manifest.get("url_bindings", {})
    if not isinstance(bindings, dict):
        raise ConfigurationError("asset URL bindings must be a mapping")
    verified_bindings = {}
    for url, binding in sorted(bindings.items()):
        _validate_url(url)
        if not isinstance(binding, dict) or set(binding) != {"root", "relative", "sha256"}:
            raise ConfigurationError("asset URL binding requires root, relative and sha256")
        if binding["root"] not in root_paths:
            raise ConfigurationError("asset URL binding names an unbound root")
        relative = Path(binding["relative"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ConfigurationError("asset URL binding escapes its root")
        path = str((root_paths[binding["root"]] / relative).resolve(strict=True))
        if path not in expected or expected[path]["sha256"] != binding["sha256"]:
            raise ConfigurationError("asset URL binding is absent from the frozen file inventory")
        verified_bindings[url] = expected[path]
    return {"schema_version": 2, "roots": {name: str(path) for name, path in root_paths.items()},
            "files": sorted(files, key=lambda row: row["path"]), "url_bindings": verified_bindings}


def read_native_url(url: str) -> bytes:
    """Use the renderer's native client; do not redirect it to our evidence copy."""
    import omni.client

    result, _version, content = omni.client.read_file(url)
    if result != omni.client.Result.OK:
        raise ConfigurationError(f"native asset URL read failed: {url} ({result})")
    return memoryview(content).tobytes()


def verify_remote_assets(inventory: dict, *, reader=None) -> dict:
    """Recheck every declared remote dependency, including MDL texture children."""
    reader = read_native_url if reader is None else reader
    receipts = {}
    for url, expected in inventory.get("url_bindings", {}).items():
        actual = reader(url)
        digest = hashlib.sha256(actual).hexdigest()
        if digest != expected["sha256"] or len(actual) != expected["bytes"]:
            raise ConfigurationError(f"native remote asset differs from frozen bytes: {url}")
        if sha256_file(Path(expected["path"])) != digest:
            raise ConfigurationError("frozen remote asset copy changed during native read")
        receipts[url] = {"sha256": digest, "bytes": len(actual), "file": expected["path"]}
    return receipts


def verified_asset_file(identifier: str, inventory: dict) -> dict:
    """Resolve only already bound local files/URLs; no filename or material exceptions."""
    if remote_url(identifier):
        row = inventory.get("url_bindings", {}).get(identifier)
        if row is None:
            raise ConfigurationError(f"unbound remote USD asset dependency: {identifier}")
        return row
    parsed = urlsplit(identifier)
    if parsed.scheme:
        # Native USD should expose a local filesystem path. Guessing at URI
        # escaping could bind different bytes than the actual resolver loaded.
        raise ConfigurationError(f"asset resolver returned an unsupported URI: {identifier}")
    path = str(Path(identifier).resolve(strict=True))
    expected = {row["path"]: row for row in inventory["files"]}
    row = expected.get(path)
    if row is None or sha256_file(Path(path)) != row["sha256"]:
        raise ConfigurationError(f"unbound external USD asset dependency: {path}")
    return row


def freeze_roots(roots: dict[str, Path], *, url_bindings: dict | None = None) -> dict:
    manifest = {"schema_version": 2, "roots": {}, "url_bindings": url_bindings or {}}
    for name, root in sorted(roots.items()):
        root = root.resolve(strict=True)
        manifest["roots"][name] = {
            "root": str(root),
            "files": {str(path.relative_to(root)): sha256_file(path)
                      for path in sorted(root.rglob("*")) if path.is_file()},
        }
    verify_asset_inventory(manifest)
    return manifest


def mdl_texture_references(source: str) -> list[str]:
    # Preserve strings while removing comments; commented-out default texture
    # examples in stock MDL files must not become invented dependencies.
    token = r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*'
    clean = re.sub(token, lambda m: m[0] if m[0].startswith('"') else "", source, flags=re.S)
    return sorted(set(re.findall(r'texture_(?:2d|3d|cube|ptex)\s*\(\s*"([^"\\]+)"', clean)))


def hydrate_urls(urls: list[str], output: Path, *, root_name: str = "remote_assets",
                 reader=None, max_files: int = 128, max_bytes: int = 128 * 1024 * 1024) -> dict:
    """Bounded additive copies of native remote resources and literal MDL textures.

    This is not a general MDL/USD dependency resolver. Imported MDL modules must
    also reside in explicit frozen roots; final loaded-stage gates stay required.
    """
    reader = read_native_url if reader is None else reader
    if not urls:
        raise ConfigurationError("remote asset hydration URL list is empty")
    for url in urls:
        _validate_url(url)
    allowed_hosts = {urlsplit(url).netloc for url in urls}
    output.mkdir(parents=True, exist_ok=False)
    pending = sorted(set(urls))
    bindings = {}
    total = 0
    while pending:
        url = pending.pop(0)
        if url in bindings:
            continue
        _validate_url(url)
        if urlsplit(url).netloc not in allowed_hosts or len(bindings) >= max_files:
            raise ConfigurationError("remote dependency exceeds prospective hydration scope")
        content = reader(url)
        total += len(content)
        if total > max_bytes:
            raise ConfigurationError("remote asset hydration byte bound exceeded")
        relative = hashlib.sha256(url.encode()).hexdigest() + Path(urlsplit(url).path).suffix
        with (output / relative).open("xb") as stream:
            stream.write(content)
        bindings[url] = {"root": root_name, "relative": relative,
                         "sha256": hashlib.sha256(content).hexdigest()}
        if urlsplit(url).path.lower().endswith(".mdl"):
            for reference in mdl_texture_references(content.decode("utf-8")):
                child = urljoin(url, reference)
                if child not in bindings and child not in pending:
                    pending.append(child)
    return bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--root", action="append", required=True, help="NAME=/absolute/path; repeat per complete root")
    freeze.add_argument("--url-bindings", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    hydrate = subparsers.add_parser("hydrate")
    hydrate.add_argument("--urls", type=Path, required=True, help="JSON array of exact original URLs")
    hydrate.add_argument("--asset-root", type=Path, required=True)
    hydrate.add_argument("--root-name", default="remote_assets")
    hydrate.add_argument("--standalone-native-client", action="store_true",
                         help="Initialize/shut down omni.client outside an Isaac process; no renderer starts")
    hydrate.add_argument("--output", type=Path, required=True, help="Binding JSON, outside asset-root")
    args = parser.parse_args()
    if args.command == "freeze":
        roots = {}
        for value in args.root:
            name, path = value.split("=", 1)
            if name in roots:
                raise ConfigurationError("duplicate asset root identifier")
            roots[name] = Path(path)
        if any(args.output.resolve().is_relative_to(root.resolve()) for root in roots.values()):
            raise ConfigurationError("manifest must be outside frozen asset roots")
        value = freeze_roots(roots, url_bindings=load_json(args.url_bindings) if args.url_bindings else None)
    else:
        if args.output.resolve().is_relative_to(args.asset_root.resolve()):
            raise ConfigurationError("bindings must be outside hydrated asset root")
        native_client = None
        if args.standalone_native_client:
            import omni.client

            native_client = omni.client
            if not native_client.initialize():
                raise ConfigurationError("standalone native asset client initialization failed")
        try:
            value = hydrate_urls(load_json(args.urls), args.asset_root, root_name=args.root_name)
        finally:
            if native_client is not None:
                native_client.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
