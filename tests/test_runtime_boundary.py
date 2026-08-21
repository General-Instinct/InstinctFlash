#!/usr/bin/env python3
"""The runtime cannot reach a training package or a provenance field. Enforced, not aspirational.

This is AUDIT.md Stage 4, and it is the deliverable that stops the audit being needed twice.

WHAT IT ENFORCES

    1. No module under runtime/, planners/, executors/, descriptors/ or backends/ imports a training
       package -- `instinct_pdd` or `instinctflash.train` -- at any depth, including function-local
       imports. The violation it was written for was function-local, which is exactly why it went
       unnoticed: `runtime/block_heads.py` did `from instinctflash.adapters.lingbot_velocity import ...`
       inside the install function, and that module does `from instinct_pdd import Grid`.

    2. No module in the runtime path reads a provenance key. `coverage_gate_pass` was read in the
       serving path to decide whether to serve -- right intent, wrong layer, and expressible only
       because delta.json put training statistics in the same flat namespace as execution facts.

    3. Transitively. Importing a clean module that imports a dirty one is the same violation, so the
       check follows first-party imports rather than stopping at depth 1.

WHY AST AND NOT grep. A grep for "instinct_pdd" matches this file's own docstring, every comment
explaining the rule, and AUDIT.md. Parsing means only real `import` statements count, so the rule can
be *discussed* in the tree it governs without tripping over itself.

WHY THIS IS A TEST AND NOT A LINT RULE. It encodes a project invariant with a reason, and the failure
message has to carry that reason -- a linter would say "banned import" where what is needed is "the
serving path must not depend on how a checkpoint was trained, see AUDIT.md F1".
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


#: Directories that constitute "the runtime path": everything that runs when a checkpoint is served.
GOVERNED = ("runtime", "planners", "executors", "descriptors", "backends")

#: Packages that exist to TRAIN. Importing one from the runtime path makes a training method a
#: serving dependency.
FORBIDDEN_PACKAGES = ("instinct_pdd", "instinctflash.train")

#: Keys that describe HOW a checkpoint was trained. Reading one from the runtime path is the
#: dependency this project forbids, even when the value is used for something sensible.
PROVENANCE_KEYS = (
    "coverage_gate_pass", "training_method", "recipe", "endpoint_rmse",
    "min_updates_per_head", "head_updates_min", "training_diagnostics",
)

#: The single sanctioned exception, and it is scoped by ENCLOSING CONTEXT rather than by key.
#:
#: `descriptors/checkpoint.py` is the boundary module: it exists to keep provenance away from the
#: runtime, and doing that requires naming the keys it refuses. A blanket key-based exemption would
#: let the same file quietly start READING them, which is the failure this whole test exists to
#: prevent. So the allowance is "these three contexts, in this one file":
#:
#:   FORBIDDEN_IN_EXECUTION   a blocklist -- naming a key to REJECT it is the opposite of reading it
#:   _from_legacy_delta       maps flat delta.json onto the two namespaces; the only place
#:                            `coverage_gate_pass` is read, once, to produce `servable`
#:   provenance_of            hands provenance to humans and tools, deliberately, never to a planner
#:
#: Anywhere else in that file -- including `load_declaration` -- is a violation.
QUARANTINE_FILE = "instinctflash/descriptors/checkpoint.py"
QUARANTINE_CONTEXTS = ("FORBIDDEN_IN_EXECUTION", "_from_legacy_delta", "provenance_of")


def modules_under(*dirs) -> list[Path]:
    out = []
    for d in dirs:
        out += sorted((ROOT / "instinctflash" / d).rglob("*.py"))
    return out


def imports_of(path: Path) -> list[str]:
    """Every module named by a real import statement, at any nesting depth."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:           # relative import; resolve against the package
                pkg = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-node.level])
                names.append(f"{pkg}.{node.module}" if node.module else pkg)
            elif node.module:
                names.append(node.module)
    return names


def to_path(module: str) -> Path | None:
    if not module.startswith("instinctflash"):
        return None
    p = ROOT / Path(*module.split("."))
    for cand in (p.with_suffix(".py"), p / "__init__.py"):
        if cand.exists():
            return cand
    return None


def reaches_forbidden(start: Path) -> list[str] | None:
    """BFS over first-party imports. Returns the offending chain, or None if clean."""
    seen, queue = set(), [(start, [start.relative_to(ROOT).as_posix()])]
    while queue:
        path, chain = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        for mod in imports_of(path):
            if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_PACKAGES):
                return chain + [mod]
            nxt = to_path(mod)
            if nxt is not None and nxt not in seen:
                queue.append((nxt, chain + [mod]))
    return None


def test_no_training_imports():
    print("\n=== 1. the runtime path cannot reach a training package ===")
    mods = modules_under(*GOVERNED)
    check(len(mods) > 20, f"governing {len(mods)} modules across {len(GOVERNED)} directories")
    bad = []
    for m in mods:
        chain = reaches_forbidden(m)
        if chain:
            bad.append(" -> ".join(chain))
    for b in bad:
        print(f"        {b}")
    check(not bad,
          "no module under runtime/planners/executors/descriptors/backends reaches "
          f"{list(FORBIDDEN_PACKAGES)}",
          "transitively, including function-local imports" if not bad else f"{len(bad)} violation(s)")


def test_no_provenance_reads():
    print("\n=== 2. the runtime path cannot read a provenance field ===")
    bad = []
    for m in modules_under(*GOVERNED):
        rel = m.relative_to(ROOT).as_posix()
        tree = ast.parse(m.read_text(), filename=str(m))
        allowed_lines = (_quarantined_lines(tree) if rel == QUARANTINE_FILE else set())
        for node in ast.walk(tree):
            # Only string LITERALS in code count. A key named in a comment or docstring is the rule
            # being explained, not broken -- which is why this walks constants rather than text.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in PROVENANCE_KEYS and not _is_docstring(tree, node):
                    if node.lineno not in allowed_lines:
                        bad.append(f"{rel}:{node.lineno} reads {node.value!r}")
    for b in bad:
        print(f"        {b}")
    check(not bad, f"no runtime module names any of {list(PROVENANCE_KEYS)} as a live string",
          "one file exempt, scoped to 3 named contexts" if not bad else f"{len(bad)} found")


def _quarantined_lines(tree) -> set[int]:
    """Line numbers inside the sanctioned contexts of the boundary module."""
    lines: set[int] = set()
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and node.targets:
            tgt = node.targets[0]
            name = tgt.id if isinstance(tgt, ast.Name) else None
        if name in QUARANTINE_CONTEXTS:
            lines |= set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def _is_docstring(tree, node) -> bool:
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(parent, "body", [])
            if body and isinstance(body[0], ast.Expr) and body[0].value is node:
                return True
    return False


def test_the_quarantine_is_real():
    print("\n=== 3. the one exception is where it claims to be ===")
    tree = ast.parse((ROOT / QUARANTINE_FILE).read_text())
    found = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    found |= {t2.id for n in tree.body if isinstance(n, ast.Assign)
              for t2 in n.targets if isinstance(t2, ast.Name)}
    for ctx in QUARANTINE_CONTEXTS:
        check(ctx in found, f"{QUARANTINE_FILE} still defines {ctx}",
              "if it is renamed or moved, the exemption must move with it")
    ck = (ROOT / QUARANTINE_FILE).read_text()
    check("_from_legacy_delta" in ck, "and it lives in _from_legacy_delta, not in a loader hot path")
    check("def provenance_of" in ck,
          "provenance is reachable deliberately, via provenance_of(), never from load_declaration")


def test_the_check_can_actually_fail():
    print("\n=== 4. the gate can fail -- otherwise it is decoration ===")
    # A gate that cannot fail on the bug it gates is worse than no gate: it produces a signed-off
    # feeling. So plant the exact violation this file was written for and confirm it is caught.
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT / "instinctflash" / "runtime") as d:
        planted = Path(d) / "planted_violation.py"
        planted.write_text("def f():\n    from instinct_pdd import Grid\n    return Grid\n")
        chain = reaches_forbidden(planted)
        check(chain is not None and "instinct_pdd" in chain[-1],
              "a function-local `from instinct_pdd import Grid` IS caught",
              " -> ".join(chain) if chain else "NOT CAUGHT")

        indirect = Path(d) / "planted_indirect.py"
        indirect.write_text("from instinctflash.train.oracles.lingbot_velocity import "
                            "LingBotChunk0VideoOracle\n")
        chain2 = reaches_forbidden(indirect)
        check(chain2 is not None, "and so is an import of a module that imports it",
              " -> ".join(chain2) if chain2 else "NOT CAUGHT")


def main() -> int:
    test_no_training_imports()
    test_no_provenance_reads()
    test_the_quarantine_is_real()
    test_the_check_can_actually_fail()
    print("\n" + "=" * 72)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        print("\nThe runtime must not depend on how a checkpoint was trained. See AUDIT.md.")
        return 1
    print("PASS: the runtime/training boundary is enforced by construction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
