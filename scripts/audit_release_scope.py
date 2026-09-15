#!/usr/bin/env python3
"""Audit a proposed source boundary without copying, importing or publishing packages.

Python 3.11+, or Python 3.10 with the test dependencies (tomli). This is a static
review aid, not a license, dependency-closure,
wheel-content, hardware-performance or task-quality certificate.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from fnmatch import fnmatchcase
import hashlib
from importlib.util import resolve_name
import json
from pathlib import Path, PurePosixPath
import subprocess

try:
    import tomllib
except ImportError:  # Python 3.10; tomli is also part of the pytest environment.
    import tomli as tomllib


CLASSES = {"public_candidate", "hold", "implementation_review", "release_review"}
GENERATED = {"__pycache__", ".git", "build", "dist", "node_modules", ".venv", "venv"}


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("scope paths must be nonempty repository-relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"invalid scope path: {value!r}")
    return value


def validate_scope(scope: dict) -> None:
    if scope.get("schema") != "instinctflash.oss_scope.v1":
        raise ValueError("unsupported scope schema")
    seen = set()
    for rule in scope["rules"]:
        prefix = relative_path(rule["prefix"])
        if prefix in seen or rule["classification"] not in CLASSES:
            raise ValueError(f"duplicate prefix or unknown classification: {prefix}")
        seen.add(prefix)
    for root in scope["inventory_roots"]:
        relative_path(root)
    for root in scope["package_roots"]:
        relative_path(root["path"])
        if not all(part.isidentifier() for part in root["module"].split(".")):
            raise ValueError("invalid module root")


def classify(relative: str, scope: dict) -> dict:
    matches = [r for r in scope["rules"]
               if relative == r["prefix"] or relative.startswith(r["prefix"] + "/")]
    if not matches:
        return {"classification": "unclassified", "group": "unclassified"}
    return max(matches, key=lambda r: len(r["prefix"]))


def module_name(relative: str, scope: dict) -> str | None:
    if not relative.endswith(".py"):
        return None
    for root in sorted(scope["package_roots"], key=lambda r: len(r["path"]), reverse=True):
        prefix = root["path"] + "/"
        if relative.startswith(prefix):
            suffix = relative[len(prefix):-3].split("/")
            if suffix[-1] == "__init__":
                suffix.pop()
            return ".".join([root["module"], *suffix])
    return None


def imports(source: str, module: str, *, package_init: bool):
    tree = ast.parse(source)
    package = module if package_init else module.rpartition(".")[0]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    import_functions = {"import_module", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"importlib", "builtins"}:
            import_functions.update(alias.asname or alias.name for alias in node.names
                                    if alias.name in {"import_module", "__import__"})
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                try:
                    base = resolve_name("." * node.level + base, package)
                except (ImportError, ValueError):
                    yield {"module": "", "line": node.lineno, "kind": "unresolved_relative_import"}
                    continue
            names = [base, *(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")]
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name in import_functions:
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    name = node.args[0].value
                    if name and not name.startswith("."):
                        names = [name]
                    else:
                        yield {"module": "", "line": node.lineno, "kind": "dynamic_import_review"}
                else:
                    yield {"module": "", "line": node.lineno, "kind": "dynamic_import_review"}
        if not names:
            continue
        ancestor = parents.get(node)
        nested = False
        while ancestor is not None and not isinstance(ancestor, ast.Module):
            nested = True
            ancestor = parents.get(ancestor)
        for name in names:
            yield {"module": name, "line": node.lineno, "kind": "import",
                   "placement": "deferred_or_conditional" if nested else "module_level"}


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def audit(repository: Path, scope: dict, paths: list[str]) -> dict:
    validate_scope(scope)
    repository = repository.resolve()
    files, sources, findings = {}, {}, []
    project_content = None
    for relative in sorted(set(paths)):
        relative_path(relative)
        if any(p in GENERATED or p.endswith(".egg-info") or p.startswith(".venv-")
               for p in PurePosixPath(relative).parts):
            continue
        path = repository / relative
        rule = classify(relative, scope)
        entry = {"classification": rule["classification"], "group": rule["group"]}
        resolved = path.resolve()
        if not resolved.is_relative_to(repository):
            entry["symlink_target"] = "outside_repository"
            findings.append({"kind": "symlink_outside_repository", "path": relative,
                             "severity": "blocker"})
            files[relative] = entry
            continue  # Check parent directory links too; never read an out-of-tree target.
        if resolved != path:
            target = resolved.relative_to(repository).as_posix()
            entry["symlink_target"] = target
            entry["target_classification"] = classify(target, scope)["classification"]
            if entry["classification"] == "public_candidate" and entry["target_classification"] != "public_candidate":
                findings.append({"kind": "public_symlink_crosses_boundary", "path": relative,
                                 "target": target, "severity": "blocker"})
        if not path.is_file():
            findings.append({"kind": "missing_source", "path": relative, "severity": "blocker"})
            files[relative] = entry
            continue
        content = path.read_bytes()
        if relative == "pyproject.toml":
            project_content = content
        entry.update(sha256=_sha(content), bytes=len(content))
        files[relative] = entry
        if relative.endswith(".py"):
            sources[relative] = content.decode("utf-8")
        if entry["classification"] == "unclassified":
            findings.append({"kind": "unclassified_source", "path": relative, "severity": "blocker"})

    modules = {name: relative for relative in sources if (name := module_name(relative, scope))}
    for relative, source in sources.items():
        if files[relative]["classification"] != "public_candidate":
            continue
        try:
            dependencies = list(imports(source, module_name(relative, scope) or "",
                                        package_init=relative.endswith("/__init__.py")))
        except (SyntaxError, ValueError) as error:
            findings.append({"kind": "parse_error", "path": relative,
                             "line": getattr(error, "lineno", None), "severity": "blocker"})
            continue
        seen = set()
        for dependency in dependencies:
            if dependency["kind"] != "import":
                findings.append({**dependency, "path": relative, "severity": "review"})
                continue
            parts = dependency["module"].split(".")
            targets = {modules[name] for end in range(1, len(parts) + 1)
                       if (name := ".".join(parts[:end])) in modules}
            # Python initializes parent packages too. An omitted held module
            # must not disappear just because a public parent was inventoried.
            for root in scope["package_roots"]:
                prefix = root["module"] + "."
                if dependency["module"].startswith(prefix):
                    inferred = root["path"] + "/" + dependency["module"][len(prefix):].replace(".", "/")
                    if (classify(inferred, scope)["classification"] == "hold"
                            and dependency["module"] not in modules
                            and not any(files[t]["classification"] == "hold" for t in targets)):
                        key = (dependency["line"], inferred)
                        if key not in seen:
                            seen.add(key)
                            findings.append({**dependency, "kind": "uninventoried_held_import", "path": relative,
                                             "target": inferred, "severity": "blocker"})
            for target in sorted(targets):
                if files[target]["classification"] == "public_candidate":
                    continue
                key = (dependency["line"], target)
                if key in seen:
                    continue
                seen.add(key)
                target_class = files[target]["classification"]
                findings.append({**dependency, "kind": "import_crosses_boundary", "path": relative,
                                 "target": target, "target_classification": target_class,
                                 "severity": "blocker" if target_class in {"hold", "unclassified"} else "review"})

    # Inspect discovery declarations, not a built wheel. Data files and custom
    # build hooks still require an actual artifact-content review.
    if project_content is not None:
        config = tomllib.loads(project_content.decode("utf-8"))
        package_config = config.get("tool", {}).get("setuptools", {}).get("packages")
        explicit = package_config if isinstance(package_config, list) else None
        discovery = package_config.get("find", {}) if isinstance(package_config, dict) else {}
        include, exclude = discovery.get("include", ["*"]), discovery.get("exclude", [])
        where = discovery.get("where", ["."])
        if where != ["."] or package_config is None:
            findings.append({"kind": "custom_package_discovery_review", "path": "pyproject.toml",
                             "severity": "review"})
        else:
            for relative in sources:
                package = str(PurePosixPath(relative).parent).replace("/", ".")
                selected = (package in explicit if explicit is not None else
                            any(fnmatchcase(package, pattern) for pattern in include)
                            and not any(fnmatchcase(package, pattern) for pattern in exclude))
                if files[relative]["classification"] != "public_candidate" and selected:
                    findings.append({"kind": "current_core_discovery_includes_nonpublic_candidate",
                                     "path": relative, "package": package,
                                     "classification": files[relative]["classification"], "severity": "blocker"})

    summary = dict(Counter(entry["classification"] for entry in files.values()))
    return {
        "schema": "instinctflash.oss_scope_audit.v1", "status": "local_review_complete",
        "scope_status": scope["status"], "publication_ready": False,
        "scope_sha256": _sha(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()),
        "files": files, "counts": summary, "findings": findings,
        "blockers": sum(f["severity"] == "blocker" for f in findings),
        "reviews": sum(f["severity"] == "review" for f in findings),
        "limitations": [
            "Static source review only; no package/model import, GPU, network, copying or publication.",
            "Conditional imports are recorded, not assumed to be mandatory or harmless.",
            "Computed imports, native loads, external vendor dependencies and build hooks need separate validation.",
            "Source classifications are proposals, not license determinations or publication authorization.",
            "Current discovery check is not an inspection of wheel or sdist contents.",
            "No clean installation, performance or task-quality qualification is implied.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; existing reports are not replaced")
    parser.add_argument("--check", action="store_true", help="exit 1 for unresolved blockers or reviews")
    args = parser.parse_args()
    repository = args.repository.resolve()
    scope_path = args.scope or repository / "release/oss_scope.json"
    scope = json.loads(scope_path.read_text())
    validate_scope(scope)
    paths = subprocess.check_output([
        "git", "-C", str(repository), "ls-files", "-z", "--cached", "--others",
        "--exclude-standard", "--", *scope["inventory_roots"],
    ]).decode().rstrip("\0").split("\0")
    report = audit(repository, scope, [p for p in paths if p])
    report["head"] = subprocess.check_output([
        "git", "-C", str(repository), "rev-parse", "HEAD",
    ]).decode().strip()
    report["auditor_sha256"] = _sha(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.output), "files": len(report["files"]),
                      "blockers": report["blockers"], "reviews": report["reviews"],
                      "publication_ready": False}))
    return int(args.check and bool(report["blockers"] or report["reviews"]))


if __name__ == "__main__":
    raise SystemExit(main())
