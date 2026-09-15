#!/usr/bin/env python3
"""Repair one explicitly pinned public wheel's metadata, without loading its code.

This is not a generic compatibility override. The allowlist binds original
public artifact bytes, one internal platform tag, and the native ELF machine.
The caller downloads the original wheel; this tool never installs a package.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def inspect(z: zipfile.ZipFile, rule: dict) -> tuple[dict, bytes, bytes]:
    infos = z.infolist()
    names = [i.filename for i in infos]
    if len(names) != len(set(names)):
        raise ValueError("duplicate wheel member")
    for i in infos:
        p = PurePosixPath(i.filename)
        if (p.is_absolute() or ".." in p.parts or "\\" in i.filename or
                p.as_posix() != i.filename or i.is_dir() or
                stat.S_ISLNK(i.external_attr >> 16) or i.flag_bits & 1):
            raise ValueError("unsafe wheel member")
    base = rule["dist_info"]
    wheel_name, record_name = f"{base}/WHEEL", f"{base}/RECORD"
    if any(".dist-info/" in n and not n.startswith(base + "/") for n in names):
        raise ValueError("unexpected distribution metadata")
    wheel, record = z.read(wheel_name), z.read(record_name)
    rows = list(csv.reader(io.StringIO(record.decode())))
    rows = [r for r in rows if r]  # A trailing blank line is retained in the output.
    if any(len(r) != 3 for r in rows) or len(rows) != len(names):
        raise ValueError("invalid RECORD rows")
    if len({r[0] for r in rows}) != len(rows) or {r[0] for r in rows} != set(names):
        raise ValueError("RECORD must cover each wheel member exactly once")
    inventory = {}
    elf_count = 0
    for name, expected_hash, expected_size in rows:
        h, size, prefix = hashlib.sha256(), 0, b""
        with z.open(name) as stream:
            for block in iter(lambda: stream.read(4 << 20), b""):
                if not prefix:
                    prefix = block[:64]
                h.update(block)
                size += len(block)
        record_hash = "sha256=" + base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()
        if name == record_name:
            if expected_hash or expected_size:
                raise ValueError("RECORD self-entry must be unhashed")
        elif expected_hash != record_hash or expected_size != str(size):
            raise ValueError(f"RECORD content mismatch: {name}")
        if prefix.startswith(b"\x7fELF"):
            if len(prefix) < 20 or prefix[4:6] != b"\x02\x01":
                raise ValueError("native payload is not ELF64 little-endian")
            if struct.unpack_from("<H", prefix, 18)[0] != rule["elf_machine"]:
                raise ValueError("native ELF architecture mismatch")
            elf_count += 1
        inventory[name] = {"bytes": size, "sha256": h.hexdigest()}
    if elf_count == 0:
        raise ValueError("no native ELF payload proved the target architecture")
    return inventory, wheel, record


def repair(original: Path, output: Path, rule: dict) -> dict:
    if original.name != rule["filename"] or output.name != rule["filename"]:
        raise ValueError("unexpected wheel filename")
    if original.stat().st_size != rule["bytes"] or sha256(original) != rule["sha256"]:
        raise ValueError("original public wheel hash/size mismatch")
    if output.exists() or output.is_symlink() or output.resolve() == original.resolve():
        raise ValueError("repair destination must be new and separate")
    with zipfile.ZipFile(original) as z:
        before, wheel, record = inspect(z, rule)
        old_line = f"Tag: {rule['old_tag']}".encode()
        tags = [x for x in wheel.splitlines() if x.startswith(b"Tag:")]
        if tags != [old_line]:
            raise ValueError("wheel does not have exactly the authorized old tag")
        new_wheel = wheel.replace(old_line, f"Tag: {rule['new_tag']}".encode(), 1)
        base = rule["dist_info"]
        wheel_name, record_name = f"{base}/WHEEL", f"{base}/RECORD"
        old_row = f"{wheel_name},{digest(wheel)},{len(wheel)}".encode()
        if record.splitlines().count(old_row) != 1:
            raise ValueError("unexpected WHEEL RECORD encoding")
        new_row = f"{wheel_name},{digest(new_wheel)},{len(new_wheel)}".encode()
        new_record = record.replace(old_row, new_row, 1)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as raw, zipfile.ZipFile(raw, "w") as out:
            out.comment = z.comment
            for entry in z.infolist():
                info = copy.copy(entry)
                if entry.filename in (wheel_name, record_name):
                    out.writestr(info, new_wheel if entry.filename == wheel_name else new_record)
                else:
                    with z.open(entry) as src, out.open(info, "w") as dst:
                        for block in iter(lambda: src.read(4 << 20), b""):
                            dst.write(block)
    with zipfile.ZipFile(output) as z:
        after, final_wheel, final_record = inspect(z, rule)
    changed = sorted(n for n in before if before[n] != after[n])
    if changed != sorted([wheel_name, record_name]) or final_wheel != new_wheel or final_record != new_record:
        raise ValueError("repair changed bytes beyond the authorized metadata")
    if sha256(original) != rule["sha256"]:
        raise ValueError("original wheel changed during repair")
    return {
        "schema": "instinctflash.public_wheel_metadata_repair.v1", "status": "passed",
        "public_source": rule, "original": {"path": str(original), "sha256": rule["sha256"]},
        "repaired": {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size},
        "changed_members": changed, "members_before": before, "members_after": after,
        "all_other_payload_bytes_preserved": True, "installed": False,
        "mandatory_next_gate": "Install in the new target environment, then require both pip check and uv pip check.",
        "repair_source_sha256": sha256(Path(__file__)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--rule", default="nvidia-cusparselt-cu13-0.8.0-aarch64")
    p.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "release/vendor/wheel_repairs.json")
    p.add_argument("--receipt", required=True, type=Path)
    a = p.parse_args()
    if a.receipt.exists():
        raise ValueError("receipt already exists")
    result = repair(a.original, a.output, json.loads(a.catalog.read_text())["repairs"][a.rule])
    with a.receipt.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status": result["status"], "repaired": result["repaired"], "receipt": str(a.receipt)}))


if __name__ == "__main__":
    main()
