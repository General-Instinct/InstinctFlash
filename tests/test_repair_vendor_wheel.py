import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import zipfile

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/repair_vendor_wheel.py"
spec = importlib.util.spec_from_file_location("repair_vendor_wheel", SOURCE)
wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel)


def fixture(tmp_path, *, machine=183, tag="py3-none-manylinux2014_sbsa", corrupt_record=False):
    name = "example-1-py3-none-manylinux2014_aarch64.whl"
    root = "example-1.dist-info"
    elf = bytearray(80)
    elf[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", elf, 18, machine)
    members = {"package/lib.so": bytes(elf), f"{root}/WHEEL": f"Wheel-Version: 1.0\nTag: {tag}\n\n".encode(),
               f"{root}/METADATA": b"Name: example\nVersion: 1\n", "package/LICENSE": b"original notice"}
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for n, b in members.items():
        h = base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b"=").decode()
        writer.writerow([n, "sha256=" + h, len(b) + int(corrupt_record)])
    writer.writerow([f"{root}/RECORD", "", ""])
    members[f"{root}/RECORD"] = record.getvalue().encode()
    original = tmp_path / "original" / name
    original.parent.mkdir()
    with zipfile.ZipFile(original, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in members.items():
            z.writestr(n, b)
    rule = {"filename": name, "dist_info": root, "bytes": original.stat().st_size,
            "sha256": wheel.sha256(original), "old_tag": "py3-none-manylinux2014_sbsa",
            "new_tag": "py3-none-manylinux2014_aarch64", "elf_machine": 183}
    return original, tmp_path / "repaired" / name, rule, members


def test_all_native_and_notice_bytes_preserved_and_original_unchanged(tmp_path):
    original, output, rule, members = fixture(tmp_path)
    receipt = wheel.repair(original, output, rule)
    assert wheel.sha256(original) == rule["sha256"]
    assert receipt["all_other_payload_bytes_preserved"]
    with zipfile.ZipFile(output) as z:
        for name, content in members.items():
            if not name.endswith(("/WHEEL", "/RECORD")):
                assert z.read(name) == content
        assert b"Tag: py3-none-manylinux2014_aarch64\n" in z.read("example-1.dist-info/WHEEL")
    with pytest.raises(ValueError, match="new and separate"):
        wheel.repair(original, output, rule)


@pytest.mark.parametrize("failure", ["hash", "size", "machine", "record", "tag", "multiple_tag"])
def test_rejects_unproved_or_broader_compatibility_changes(tmp_path, failure):
    kwargs = {}
    if failure == "machine":
        kwargs["machine"] = 62
    if failure == "record":
        kwargs["corrupt_record"] = True
    if failure == "tag":
        kwargs["tag"] = "py3-none-manylinux2014_x86_64"
    if failure == "multiple_tag":
        kwargs["tag"] = "py3-none-manylinux2014_sbsa\nTag: py3-none-any"
    original, output, rule, _ = fixture(tmp_path, **kwargs)
    if failure == "hash":
        rule["sha256"] = "0" * 64
    if failure == "size":
        rule["bytes"] += 1
    with pytest.raises(ValueError):
        wheel.repair(original, output, rule)
    assert not output.exists()


def test_destination_cannot_be_original_or_symlink(tmp_path):
    original, output, rule, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="new and separate"):
        wheel.repair(original, original, rule)
    output.parent.mkdir()
    output.symlink_to(original)
    with pytest.raises(ValueError, match="new and separate"):
        wheel.repair(original, output, rule)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "other-1.dist-info/METADATA"])
def test_unsafe_or_other_distribution_members_fail(tmp_path, name):
    original, output, rule, _ = fixture(tmp_path)
    with zipfile.ZipFile(original, "a") as z:
        z.writestr(name, b"extra")
    rule.update(bytes=original.stat().st_size, sha256=wheel.sha256(original))
    with pytest.raises(ValueError):
        wheel.repair(original, output, rule)
