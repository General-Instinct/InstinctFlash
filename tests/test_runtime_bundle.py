import pytest

from benchmarks.regression.runtime_bundle import build_manifest, verify_manifest


def test_required_binary_missing_is_not_a_complete_source_snapshot(tmp_path):
    (tmp_path/'model.py').write_text('source')
    with pytest.raises(ValueError, match='Missing required native artifact'):
        build_manifest(tmp_path, required_binaries=['*.so'])


def test_binary_replacement_or_addition_invalidates_bundle(tmp_path):
    (tmp_path/'model.py').write_text('source')
    binary = tmp_path/'kernels.so'
    binary.write_bytes(b'qualified binary')
    manifest = build_manifest(tmp_path, required_binaries=['kernels.so'])
    verify_manifest(tmp_path, manifest)
    binary.write_bytes(b'different binary')
    with pytest.raises(ValueError, match='bundle changed'):
        verify_manifest(tmp_path, manifest)
    binary.write_bytes(b'qualified binary')
    (tmp_path/'unexpected.so').write_bytes(b'extra')
    with pytest.raises(ValueError, match='bundle changed'):
        verify_manifest(tmp_path, manifest)


def test_versioned_shared_libraries_are_inventoried(tmp_path):
    (tmp_path/'libattention.so.1.2').write_bytes(b'versioned ELF')
    manifest = build_manifest(tmp_path, required_binaries=['libattention.so.*'])
    assert 'libattention.so.1.2' in manifest['files']
    verify_manifest(tmp_path, manifest)
