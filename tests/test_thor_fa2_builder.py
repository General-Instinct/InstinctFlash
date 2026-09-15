"""CPU checks for the standalone builder's isolation and full ABI coverage."""

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "thor_fa2_builder", ROOT / "serving/scripts/build_thor_fa2.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def arguments(tmp_path, *extra):
    return BUILDER.parser().parse_args([
        "--build-root", str(tmp_path / "build"),
        "--output-dir", str(tmp_path / "output"),
        "--python", sys.executable, "--cmake", sys.executable,
        "--nvcc", sys.executable, "--cxx", sys.executable, *extra,
    ])


def test_plan_is_read_only_and_covers_real_vendored_abi(tmp_path):
    plan = BUILDER.plan(arguments(tmp_path))
    assert not (tmp_path / "build").exists()
    assert not (tmp_path / "output").exists()
    assert plan["jobs"] == 2 and plan["nvcc_threads_per_job"] == 1
    assert len(BUILDER.kernel_sources()) == 16
    assert all(name in plan["source_files"] for name in BUILDER.kernel_sources())
    assert plan["causal"] == {"head_dim": 128, "dtype": "bf16"}
    text = BUILDER.cmake_source()
    assert "arch=compute_110,code=sm_110" in text
    assert "CMAKE_CUDA_ARCHITECTURES OFF" in text
    assert "compute_120" not in text
    assert "--threads 1" in text
    assert "find_package(Python3 3.12 EXACT" in text
    assert "Torch" not in text


@pytest.mark.parametrize("name", ["build", "output"])
def test_existing_evidence_is_never_reused(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    marker = path / "keep"
    marker.write_bytes(b"original evidence")
    with pytest.raises(ValueError, match="must both be new"):
        BUILDER.plan(arguments(tmp_path))
    assert marker.read_bytes() == b"original evidence"


def test_nested_output_is_rejected_through_parent_symlink(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    args = arguments(tmp_path, "--output-dir", str(tmp_path / "link/build/output"))
    with pytest.raises(ValueError, match="disjoint"):
        BUILDER.plan(args)


def test_source_manifest_detects_content_and_refuses_symlinks(tmp_path):
    source = tmp_path / "source"
    for name in BUILDER.SOURCE_GROUPS:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix:
            path.write_bytes(b"source")
        else:
            path.mkdir()
    target = source / "csrc/attention/fa2_causal_inst/extra.cu"
    target.write_bytes(b"before")
    original = BUILDER.sha(target)
    target.write_bytes(b"after")
    assert original != BUILDER.sha(target)
    assert target in BUILDER.source_files(source)
    (target.parent / "link.cu").symlink_to(target)
    with pytest.raises(ValueError, match="source symlink"):
        BUILDER.source_files(source)


def test_more_than_two_compilers_rejected(tmp_path):
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--jobs", "3")


def test_cuda_132_empty_ptx_message_is_not_a_ptx_artifact():
    assert BUILDER.no_ptx_in_listing("")
    assert BUILDER.no_ptx_in_listing(
        "cuobjdump info    : No PTX file found to extract from '/tmp/library.so'. "
        "You may try with -all option.\n")
    assert not BUILDER.no_ptx_in_listing("PTX file    1: library.sm_110.ptx")
    assert not BUILDER.no_ptx_in_listing("cuobjdump fatal: Could not open input file")
