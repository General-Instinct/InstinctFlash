"""Shared, typed configuration and output plumbing for the `instinctflash` CLI verbs.

The public syntax intentionally follows the same shape as the dataclasses::

    --certify.margin=-0.05 --output.format=json

One parser, not one ``argparse.Namespace`` per command: config-file and command-line calls stay
equivalent, unknown fields are hard errors instead of silently-ignored misspellings, YAML config
then dotted CLI overrides is the precedence, errors in JSON mode use one stable schema, and
--output.path writes are atomic. This module carries NO command registry of its own — it is
plumbing for the verbs in ``instinctflash/cli.py`` and nothing else.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import contextlib
import io
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Mapping, Union, get_args, get_origin, get_type_hints


CLI_SCHEMA = 1


class ConfigError(ValueError):
    """The command line or config file does not match the command schema."""


class UnsupportedCapability(RuntimeError):
    """A named workflow/plugin exists conceptually but is not installed or implemented here."""


class HelpRequested(Exception):
    pass


@dataclass
class ModelConfig:
    path: str = ""
    revision: str | None = None
    strict: bool = True


@dataclass
class RuntimeConfig:
    device: str | None = None
    placement: Literal["auto", "in_process", "worker"] = "auto"
    nfe: dict[str, int] = field(default_factory=dict)
    tier_ceiling: Literal["bitexact", "numeric", "behavioral"] = "bitexact"
    exclude_passes: list[str] = field(default_factory=list)


@dataclass
class OutputConfig:
    format: Literal["text", "json"] = "text"
    path: Path | None = None


@dataclass
class CommandReport:
    result: Any
    text: str
    ok: bool = True
    exit_code: int = 0


def _load_yaml(text: str, where: str) -> Any:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - dependency is declared by the package
        raise ConfigError("YAML configuration requires PyYAML; install instinctflash again") from e
    try:
        return yaml.safe_load(text)
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"{where}: invalid YAML/JSON: {e}") from e


def _config_file(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config_path is not a file: {p}")
    value = _load_yaml(p.read_text(), str(p))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{p}: top-level config must be a mapping")
    return value


def _split_cli(argv: list[str]) -> tuple[str | None, list[tuple[str, str]]]:
    config_path = None
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("-h", "--help"):
            raise HelpRequested
        if not token.startswith("--"):
            raise ConfigError(f"unexpected positional argument {token!r}; use --section.field=value")
        raw = token[2:]
        if "=" in raw:
            key, value = raw.split("=", 1)
        else:
            key = raw
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ConfigError(f"--{key} requires a value (booleans use true/false)")
            i += 1
            value = argv[i]
        if key == "config_path":
            config_path = value
        else:
            out.append((key, value))
        i += 1
    return config_path, out


def _field_types(cls: type) -> dict[str, Any]:
    try:
        return get_type_hints(cls)
    except Exception:  # pragma: no cover - defensive for third-party config classes
        return {f.name: f.type for f in fields(cls)}


def _dataclass_type(tp: Any) -> type | None:
    if isinstance(tp, type) and is_dataclass(tp):
        return tp
    for arg in get_args(tp):
        if isinstance(arg, type) and is_dataclass(arg):
            return arg
    return None


def _schema_paths(cls: type, prefix: str = "") -> dict[str, Any]:
    hints = _field_types(cls)
    out: dict[str, Any] = {}
    for f in fields(cls):
        path = f"{prefix}.{f.name}" if prefix else f.name
        tp = hints.get(f.name, f.type)
        nested = _dataclass_type(tp)
        if nested is not None:
            out.update(_schema_paths(nested, path))
        else:
            out[path] = tp
    return out


def _default_for_field(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return f.default_factory()
    raise ConfigError(f"missing required field {f.name}")


def _default_mapping(cls: type) -> dict[str, Any]:
    return asdict(cls())


def _merge_known(base: dict[str, Any], incoming: Mapping[str, Any], cls: type,
                 prefix: str = "") -> None:
    hints = _field_types(cls)
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(incoming) - set(known))
    if unknown:
        at = prefix or "config"
        raise ConfigError(f"unknown field(s) under {at}: {', '.join(unknown)}")
    for key, value in incoming.items():
        tp = hints.get(key, known[key].type)
        nested = _dataclass_type(tp)
        path = f"{prefix}.{key}" if prefix else key
        if nested is not None:
            if not isinstance(value, Mapping):
                raise ConfigError(f"{path} must be a mapping")
            current = base.setdefault(key, {})
            _merge_known(current, value, nested, path)
        else:
            base[key] = value


def _parse_scalar(raw: str, tp: Any, path: str) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (Union, UnionType):
        if raw.lower() in ("null", "none") and type(None) in args:
            return None
        candidates = [a for a in args if a is not type(None)]
        errors = []
        for candidate in candidates:
            try:
                return _parse_scalar(raw, candidate, path)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(errors[-1] if errors else f"{path}: invalid value {raw!r}")
    if origin is Literal:
        value = _load_yaml(raw, path)
        allowed = list(args)
        if value not in allowed:
            raise ConfigError(f"{path} must be one of {allowed}, got {value!r}")
        return value
    if tp is str:
        return raw
    if tp is Path:
        return Path(raw)
    if tp is bool:
        if raw.lower() not in ("true", "false"):
            raise ConfigError(f"{path} must be true or false")
        return raw.lower() == "true"
    if tp is int:
        try:
            return int(raw)
        except ValueError as e:
            raise ConfigError(f"{path} must be an integer, got {raw!r}") from e
    if tp is float:
        try:
            return float(raw)
        except ValueError as e:
            raise ConfigError(f"{path} must be a number, got {raw!r}") from e
    if origin in (dict, list, tuple) or tp in (dict, list, tuple):
        value = _load_yaml(raw, path)
        want = origin or tp
        if want is dict and not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        if want in (list, tuple) and not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list")
        return tuple(value) if want is tuple else value
    return _load_yaml(raw, path)


def _set_path(doc: dict[str, Any], path: str, raw: str, schema: dict[str, Any]) -> None:
    tp = schema.get(path)
    # Mapping fields accept either one mapping or convenient per-key overrides:
    # --runtime.nfe.action=4.
    if tp is None:
        parents = [(p, t) for p, t in schema.items() if path.startswith(p + ".")
                   and (get_origin(t) is dict or t is dict)]
        if not parents:
            raise ConfigError(f"unknown option --{path}")
        parent, parent_tp = max(parents, key=lambda x: len(x[0]))
        suffix = path[len(parent) + 1:]
        if "." in suffix or not suffix:
            raise ConfigError(f"unknown option --{path}")
        value_tp = get_args(parent_tp)[1] if len(get_args(parent_tp)) == 2 else Any
        value = _parse_scalar(raw, value_tp, path)
        target = doc
        bits = parent.split(".")
        for bit in bits[:-1]:
            target = target.setdefault(bit, {})
        target.setdefault(bits[-1], {})[suffix] = value
        return
    value = _parse_scalar(raw, tp, path)
    target = doc
    bits = path.split(".")
    for bit in bits[:-1]:
        target = target.setdefault(bit, {})
    target[bits[-1]] = value


def _coerce(value: Any, tp: Any, path: str) -> Any:
    origin, args = get_origin(tp), get_args(tp)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        candidates = [a for a in args if a is not type(None)]
        errors = []
        for candidate in candidates:
            try:
                return _coerce(value, candidate, path)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(errors[-1] if errors else f"{path}: invalid value {value!r}")
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{path} must be one of {list(args)}, got {value!r}")
        return value
    nested = _dataclass_type(tp)
    if nested is not None:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        return _build(nested, value, path)
    if tp is Path:
        return None if value is None else Path(value)
    if tp is bool and not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    if tp is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise ConfigError(f"{path} must be an integer")
    if tp is float and not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    if tp is str and not isinstance(value, str):
        raise ConfigError(f"{path} must be a string")
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        kt, vt = args or (Any, Any)
        return {_coerce(k, kt, path): _coerce(v, vt, f"{path}.{k}") for k, v in value.items()}
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list")
        item = args[0] if args else Any
        converted = [_coerce(v, item, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(converted) if origin is tuple else converted
    return value


def _build(cls: type, doc: Mapping[str, Any], prefix: str = ""):
    hints = _field_types(cls)
    values = {}
    for f in fields(cls):
        value = doc[f.name] if f.name in doc else _default_for_field(f)
        path = f"{prefix}.{f.name}" if prefix else f.name
        values[f.name] = _coerce(value, hints.get(f.name, f.type), path)
    return cls(**values)


def parse_config(cls: type, argv: list[str]):
    """Build ``cls`` from defaults, config file, then dotted CLI overrides."""
    config_path, overrides = _split_cli(argv)
    doc = _default_mapping(cls)
    if config_path:
        _merge_known(doc, _config_file(config_path), cls)
    schema = _schema_paths(cls)
    for path, raw in overrides:
        _set_path(doc, path, raw, schema)
    return _build(cls, doc)


def help_text(prog: str, description: str, cls: type) -> str:
    instance = _default_mapping(cls)
    lines = [f"usage: {prog} [--config_path=FILE] [--section.field=value ...]", "", description,
             "", "options:", "  --config_path=FILE"]
    for path, tp in _schema_paths(cls).items():
        value: Any = instance
        for bit in path.split("."):
            value = value[bit]
        default = json.dumps(value, default=str, ensure_ascii=False)
        lines.append(f"  --{path}=VALUE  (default: {default})")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def emit(command: str, output: OutputConfig, report: CommandReport, *, version: str) -> int:
    if output.format == "json":
        payload = {
            "instinctflash_cli_schema": CLI_SCHEMA,
            "command": command,
            "version": version,
            "ok": bool(report.ok),
            "result": _jsonable(report.result),
        }
        content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    else:
        content = report.text.rstrip() + "\n"
    if output.path is None:
        print(content, end="")
    else:
        _atomic_write(output.path, content)
    return report.exit_code


def _requested_json(argv: list[str]) -> bool:
    for i, token in enumerate(argv):
        if token == "--output.format=json":
            return True
        if token == "--output.format" and i + 1 < len(argv) and argv[i + 1] == "json":
            return True
    return False


def execute(command: str, cls: type, run_fn, argv: list[str] | None, *, prog: str,
            description: str) -> int:
    """Parse, run, and render a command with stable exit/error semantics."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        cfg = parse_config(cls, args)
    except HelpRequested:
        print(help_text(prog, description, cls))
        return 0
    except ConfigError as e:
        if _requested_json(args):
            print(json.dumps({"instinctflash_cli_schema": CLI_SCHEMA, "command": command,
                              "version": _version(), "ok": False, "result": {},
                              "error": {"code": "CONFIG_ERROR", "message": str(e)}}, indent=2))
        else:
            print(f"{prog}: config error: {e}", file=sys.stderr)
        return 2

    output = getattr(cfg, "output", OutputConfig())
    captured = io.StringIO()
    try:
        # Business code (third-party plugins in particular) may still use print for progress.  Keep
        # that useful output, but route it to stderr so JSON mode owns stdout completely.
        with contextlib.redirect_stdout(captured):
            report = run_fn(cfg)
        if not isinstance(report, CommandReport):
            raise TypeError(f"{run_fn.__module__}.run() must return CommandReport")
    except UnsupportedCapability as e:
        report = CommandReport({}, f"UNSUPPORTED CAPABILITY: {e}", False, 3)
    except ConfigError as e:
        report = CommandReport({}, f"CONFIG ERROR: {e}", False, 2)
    except Exception as e:  # noqa: BLE001 - commands are a traceback-free product boundary
        report = CommandReport({}, f"{type(e).__name__}: {e}", False, 1)
    progress = captured.getvalue()
    if progress:
        print(progress, end="" if progress.endswith("\n") else "\n", file=sys.stderr)

    if not report.ok and output.format == "json":
        # Keep successful and failed JSON in one stable top-level shape.
        payload = {
            "instinctflash_cli_schema": CLI_SCHEMA, "command": command, "version": _version(),
            "ok": False, "result": _jsonable(report.result),
            "error": {"code": ("UNSUPPORTED_CAPABILITY" if report.exit_code == 3 else
                               "CONFIG_ERROR" if report.exit_code == 2 else "OPERATION_FAILED"),
                      "message": report.text},
        }
        content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        if output.path is None:
            print(content, end="")
        else:
            _atomic_write(output.path, content)
        return report.exit_code
    return emit(command, output, report, version=_version())


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("instinctflash")
    except Exception:  # pragma: no cover - source tree without package metadata
        return "0.1.0"


__all__ = [
    "CLI_SCHEMA", "CommandReport", "ConfigError", "ModelConfig", "OutputConfig",
    "RuntimeConfig", "UnsupportedCapability", "emit", "execute", "help_text", "parse_config",
]
