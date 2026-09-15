"""CPU-only, byte-preserving tensor overlays; no pickle, Torch or model execution."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative(name):
    require(isinstance(name, str) and name and "\\" not in name
            and not PurePosixPath(name).is_absolute()
            and all(x not in {"", ".", ".."} for x in name.split("/")), "unsafe file name")
    return name


def inside(root, name):
    root = Path(root).resolve()
    path = root / relative(name)
    require(path.resolve().is_relative_to(root), "symlink escapes input directory")
    require(path.is_file(), f"missing input: {name}")
    return path


def checked_file(root, row):
    path = inside(root, row["path"])
    require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
            f"changed input: {row['path']}")
    return path


def read_header(path):
    """Validate the complete unambiguous BF16 safetensors layout used by this recipe."""
    with Path(path).open("rb") as stream:
        size = stream.read(8)
        require(len(size) == 8, "truncated safetensors header")
        length = struct.unpack("<Q", size)[0]
        require(0 < length <= 16 * 1024 * 1024, "invalid safetensors header size")
        raw = stream.read(length)
    require(len(raw) == length, "truncated safetensors header")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    entries = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(entries, dict), "invalid safetensors header")
    tensors = {key: value for key, value in entries.items() if key != "__metadata__"}
    intervals = []
    widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4,
              "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    for key, item in tensors.items():
        require(isinstance(key, str) and key and isinstance(item, dict), "invalid tensor entry")
        shape, offsets = item.get("shape"), item.get("data_offsets")
        require(isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape),
                "invalid tensor shape")
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(v) is int and v >= 0 for v in offsets), "invalid tensor offsets")
        require(item.get("dtype") in widths, "unsupported tensor dtype")
        start, end = offsets
        require(end - start == math.prod(shape) * widths[item["dtype"]], "tensor byte size mismatch")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        require(start == cursor and end >= start, "tensor gaps or overlaps")
        cursor = end
    require(8 + length + cursor == Path(path).stat().st_size, "tensor payload length mismatch")
    return raw, tensors, 8 + length


def tensor_digest(stream, start, size, destination=None):
    stream.seek(start)
    result = hashlib.sha256()
    remaining = size
    while remaining:
        block = stream.read(min(8 * 1024 * 1024, remaining))
        require(bool(block), "truncated tensor payload")
        result.update(block)
        if destination is not None:
            destination.write(block)
        remaining -= len(block)
    return result.hexdigest()


def export_overlay(shard, tensors, output):
    """Publisher helper: select only reviewed absolute BF16 replacement bytes."""
    shard, output = Path(shard).resolve(), Path(output).resolve()
    require(not output.exists() and output != shard, "use a fresh overlay output")
    _, header, offset = read_header(shard)
    require(tensors and len({r["safetensors_key"] for r in tensors}) == len(tensors),
            "duplicate or empty tensor selection")
    cursor, selected = 0, {}
    for row in sorted(tensors, key=lambda r: r["safetensors_key"]):
        key = row["safetensors_key"]
        entry = header[key]
        require(entry["dtype"] == row["dtype"] == "BF16" and entry["shape"] == row["shape"],
                "replacement tensor metadata differs")
        require(entry["data_offsets"][1] - entry["data_offsets"][0] == row["bytes"],
                "replacement tensor byte count differs")
        selected[key] = {"dtype": "BF16", "shape": row["shape"],
                         "data_offsets": [cursor, cursor + row["bytes"]]}
        cursor += row["bytes"]
    raw = json.dumps(selected, separators=(",", ":"), sort_keys=True).encode()
    raw += b" " * (-len(raw) % 8)
    with shard.open("rb") as source, output.open("xb") as target:
        target.write(struct.pack("<Q", len(raw)))
        target.write(raw)
        for row in sorted(tensors, key=lambda r: r["safetensors_key"]):
            start = header[row["safetensors_key"]]["data_offsets"][0]
            actual = tensor_digest(source, offset + start, row["bytes"], target)
            require(actual == row["sha256"], "changed producer tensor")
        target.flush()
        os.fsync(target.fileno())
    verify_overlay(output, tensors)
    return {"path": output.name, "bytes": output.stat().st_size, "sha256": digest(output)}


def verify_overlay(path, tensors):
    _, header, offset = read_header(path)
    require(len(tensors) == len({r["safetensors_key"] for r in tensors}), "duplicate tensor selection")
    require(set(header) == {r["safetensors_key"] for r in tensors}, "overlay tensor names differ")
    with Path(path).open("rb") as source:
        for row in tensors:
            item = header[row["safetensors_key"]]
            require(item["dtype"] == row["dtype"] == "BF16" and item["shape"] == row["shape"],
                    "overlay tensor metadata differs")
            start, end = item["data_offsets"]
            require(end - start == row["bytes"], "overlay tensor size differs")
            require(tensor_digest(source, offset + start, end - start) == row["sha256"],
                    "overlay tensor hash differs")


def apply_shard(base, overlay, tensors, output, expected):
    """Copy the base shard and replace bytes at unchanged, validated tensor offsets."""
    base, overlay, output = map(Path, (base, overlay, output))
    require(not output.exists() and output.resolve() not in {base.resolve(), overlay.resolve()},
            "use a fresh independent shard output")
    raw, header, offset = read_header(base)
    require(hashlib.sha256(raw).hexdigest() == expected["header_sha256"], "base shard header differs")
    verify_overlay(overlay, tensors)
    _, overlay_header, overlay_offset = read_header(overlay)
    with base.open("rb") as source:
        for row in tensors:
            item = header[row["safetensors_key"]]
            require(item["shape"] == row["shape"] and item["dtype"] == "BF16", "base tensor differs")
            require(tensor_digest(source, offset + item["data_offsets"][0], row["bytes"])
                    == row["base_tensor_sha256"], "base tensor hash differs")
    with base.open("rb") as source, output.open("xb") as destination:
        shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
    with overlay.open("rb") as source, output.open("r+b") as destination:
        for row in tensors:
            destination.seek(offset + header[row["safetensors_key"]]["data_offsets"][0])
            require(tensor_digest(source, overlay_offset + overlay_header[row["safetensors_key"]]["data_offsets"][0],
                                  row["bytes"], destination) == row["sha256"], "overlay changed during copy")
        destination.flush()
        os.fsync(destination.fileno())
    require(output.stat().st_size == expected["bytes"] and digest(output) == expected["sha256"],
            "reconstructed shard differs from original exported shard")
    return output
