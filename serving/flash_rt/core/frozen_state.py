"""CPU-only, non-pickle schema for immutable pi0.5 execution state."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile

SCHEMA = "flashrt-pi05-frozen-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def scale_bits(scales):
    return {k: struct.pack("<f", float(v)).hex() for k, v in sorted(scales.items())}


def decode_scales(values):
    if not isinstance(values, dict) or not values:
        raise ValueError("missing frozen scales")
    result = {}
    for name, encoded in values.items():
        if not isinstance(name, str) or not isinstance(encoded, str) or len(encoded) != 8:
            raise ValueError("malformed FP32 scale")
        try:
            value = struct.unpack("<f", bytes.fromhex(encoded))[0]
        except (ValueError, struct.error) as exc:
            raise ValueError("malformed FP32 scale") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("frozen scales must be finite and positive")
        result[name] = value
    return result


def seal(payload):
    return {"payload": payload, "sha256": digest(payload)}


def validate(document, identity=None):
    if not isinstance(document, dict) or set(document) != {"payload", "sha256"}:
        raise ValueError("malformed frozen state envelope")
    payload = document["payload"]
    if document["sha256"] != digest(payload):
        raise ValueError("frozen state checksum mismatch")
    required = {"schema", "identity", "prompt", "calibration", "scale_bits", "token_lengths", "gemm"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != SCHEMA:
        raise ValueError("unsupported frozen state schema")
    if identity is not None and payload["identity"] != identity:
        raise ValueError("frozen state model, source, binary or dependency identity differs")
    if not isinstance(payload["prompt"], str) or not payload["prompt"].strip():
        raise ValueError("missing frozen prompt")
    decode_scales(payload["scale_bits"])
    lengths = payload["token_lengths"]
    if not isinstance(lengths, list) or not lengths or any(type(n) is not int or not 1 <= n <= 200 for n in lengths):
        raise ValueError("invalid frozen token-length profiles")
    if lengths != sorted(set(lengths)):
        raise ValueError("duplicate or unordered token-length profiles")
    calibration = payload["calibration"]
    if not isinstance(calibration, dict) or not calibration.get("observations_sha256"):
        raise ValueError("missing calibration provenance")
    hashes = calibration["observations_sha256"]
    percentile = calibration.get("percentile")
    if (not isinstance(hashes, list) or any(not isinstance(h, str) or len(h) != 64
            or any(c not in "0123456789abcdef" for c in h) for h in hashes)
            or type(percentile) not in (int, float) or not 0 <= percentile <= 100):
        raise ValueError("invalid calibration provenance")
    gemm = payload["gemm"]
    if not isinstance(gemm, dict) or set(gemm) != {"identity", "entries"} or not isinstance(gemm["identity"], str):
        raise ValueError("invalid GEMM metadata")
    if not isinstance(gemm["entries"], list) or not 1 <= len(gemm["entries"]) <= 4096:
        raise ValueError("invalid GEMM entry count")
    keys = set()
    for entry in gemm["entries"]:
        if not isinstance(entry, list) or len(entry) != 5:
            raise ValueError("malformed GEMM record")
        kind, m, n, k, blob = entry
        if type(kind) is not int or kind not in (0, 1, 2, 4, 5) or any(
                type(v) is not int or not 1 <= v <= 1 << 20 for v in (m, n, k)):
            raise ValueError("invalid GEMM type or dimensions")
        if not isinstance(blob, str) or len(blob) != 128:
            raise ValueError("invalid cuBLASLt algorithm bytes")
        try:
            if len(bytes.fromhex(blob)) != 64:
                raise ValueError("incomplete algorithm bytes")
        except ValueError as exc:
            raise ValueError("invalid cuBLASLt algorithm bytes") from exc
        key = tuple(entry[:4])
        if key in keys:
            raise ValueError("duplicate GEMM descriptor")
        keys.add(key)
    return payload


def save(path, payload):
    document = seal(payload)
    validate(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic, refuses to overwrite an existing artifact.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return document["sha256"]


def load(path, identity=None):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("frozen state exceeds size limit")
    return validate(json.loads(path.read_text()), identity)
