"""CPU admission tests for source-preserving Thor native wheel builds."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "serving/scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location("thor_native_builder", SCRIPTS / "build_thor_native.py")
    BUILDER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(BUILDER)
finally:
    sys.path.pop(0)


def wheel(tmp_path, *, extra=None, omit=None, tag="cp312-cp312-linux_aarch64"):
    path = tmp_path / f"flash_rt-0.1.0-{tag}.whl"
    files = {
        "flash_rt/flash_rt_kernels.cpython-312-aarch64-linux-gnu.so": b"new kernels",
        "flash_rt/flash_rt_fa2.cpython-312-aarch64-linux-gnu.so": b"new FA2",
        "flash_rt/libfmha_fp16_strided.so": b"new FMHA",
        "flash_rt/frontends/torch/vla4b_thor.py": b"public executable source",
        "flash_rt/notices/FlashAttention-LICENSE.txt": b"FA2 license",
        "flash_rt/notices/CUTLASS-FA2-LICENSE.txt": b"CUTLASS FA2 license",
        "flash_rt/notices/CUTLASS-engine-LICENSE.txt": b"CUTLASS engine license",
        "flash_rt-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n",
    }
    files.update(extra or {})
    files.pop(omit, None)
    with zipfile.ZipFile(path, "x") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return path


def test_wheel_requires_all_native_libraries_and_keeps_source(tmp_path):
    inventory = BUILDER.validate_wheel(wheel(tmp_path))
    assert len(inventory) == 8


@pytest.mark.parametrize("omission", [
    "flash_rt/flash_rt_fa2.cpython-312-aarch64-linux-gnu.so",
    "flash_rt/frontends/torch/vla4b_thor.py",
    "flash_rt/notices/CUTLASS-engine-LICENSE.txt",
])
def test_incomplete_or_source_hidden_wheel_rejected(tmp_path, omission):
    with pytest.raises(ValueError):
        BUILDER.validate_wheel(wheel(tmp_path, omit=omission))


@pytest.mark.parametrize("name", ["flash_rt/old.so", "flash_rt/api.pyc"])
def test_stale_native_or_bytecode_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        BUILDER.validate_wheel(wheel(tmp_path, extra={name: b"old"}))


def test_native_wheel_cannot_claim_pure_portable_tag(tmp_path):
    with pytest.raises(ValueError, match="wheel tag"):
        BUILDER.validate_wheel(wheel(tmp_path, tag="py3-none-any"))


def test_fresh_source_snapshot_excludes_existing_binary(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "api.py").write_text("public source")
    (source / "old.so").write_bytes(b"private historical binary")
    (source / "api.pyc").write_bytes(b"old bytecode")
    assert BUILDER.regular_files(tmp_path, ("source",)) == [source / "api.py"]


def test_readme_follows_actual_package_metadata(tmp_path):
    serving = tmp_path / "serving"
    serving.mkdir()
    metadata = serving / "pyproject.toml"
    metadata.write_text('[project]\nreadme = "README.rst"\n')
    assert "serving/README.rst" in BUILDER.repository_groups(tmp_path)
    assert "serving/README.md" not in BUILDER.repository_groups(tmp_path)
    metadata.write_text('[project]\nreadme = "../private.txt"\n')
    with pytest.raises(ValueError, match="inside serving"):
        BUILDER.repository_groups(tmp_path)


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_optional_jax_configure_block(tmp_path, enabled):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is not installed")
    text = (ROOT / "serving/CMakeLists.txt").read_text()
    start = text.index('option(FLASH_RT_BUILD_JAX_FFI ')
    end = text.index('if(JAX_FFI_DETECT_RESULT EQUAL 0', start)
    marker = tmp_path / "python_was_invoked"
    probe = tmp_path / "python-probe"
    probe.write_text(f"#!/bin/sh\nprintf invoked > '{marker}'\n")
    probe.chmod(0o755)
    script = tmp_path / "probe.cmake"
    script.write_text(f'set(Python3_EXECUTABLE "{probe}")\n' + text[start:end])
    command = [cmake]
    if not enabled:
        command += ["-DFLASH_RT_BUILD_JAX_FFI=OFF"]
    subprocess.run([*command, "-P", str(script)], check=True, capture_output=True)
    assert marker.exists() == enabled
