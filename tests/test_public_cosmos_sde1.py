"""CPU-only tests for the explicit experimental recipe and immutable tensor overlays."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "examples/cosmos3_sde1"
sys.path.insert(0, str(PACKAGE))
from cosmos3_sde1 import __main__ as cli  # noqa: E402
from cosmos3_sde1 import overlay  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tensor_file(path, values):
    header, payload = {}, b""
    for key, value in values.items():
        start = len(payload)
        payload += value
        header[key] = {"dtype": "BF16", "shape": [len(value) // 2],
                       "data_offsets": [start, len(payload)]}
    raw = json.dumps(header, sort_keys=True).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return sha(raw)


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.base = self.root / "base.safetensors"
        self.student = self.root / "student.safetensors"
        self.header_sha = tensor_file(self.base, {"a": b"abcd", "b": b"efgh"})
        tensor_file(self.student, {"a": b"1234", "b": b"efgh"})
        self.rows = [{"native_key": "a", "safetensors_key": "a", "shape": [2],
                      "dtype": "BF16", "bytes": 4, "sha256": sha(b"1234"),
                      "base_tensor_sha256": sha(b"abcd")}]
        self.target = {"bytes": self.student.stat().st_size, "sha256": overlay.digest(self.student),
                       "header_sha256": self.header_sha}

    def export(self):
        path = self.root / "overlay.safetensors"
        info = overlay.export_overlay(self.student, self.rows, path)
        self.assertEqual(info["sha256"], overlay.digest(path))
        return path

    def test_exact_merge_preserves_unselected_bytes_and_base(self):
        before = self.base.read_bytes()
        path = self.export()
        result = overlay.apply_shard(self.base, path, self.rows, self.root / "out", self.target)
        self.assertEqual(result.read_bytes(), self.student.read_bytes())
        self.assertEqual(before, self.base.read_bytes())

    def test_overlay_contains_only_selected_tensor(self):
        path = self.export()
        self.assertEqual(set(overlay.read_header(path)[1]), {"a"})

    def test_wrong_producer_bytes_fail_and_preserve_partial(self):
        self.rows[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "producer tensor"):
            self.export()
        self.assertTrue((self.root / "overlay.safetensors").exists())

    def test_refuse_existing_overlay(self):
        self.export()
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.export()

    def test_duplicate_selection_rejected(self):
        self.rows *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.export()

    def test_wrong_base_tensor_rejected(self):
        path = self.export()
        tensor_file(self.base, {"a": b"xxxx", "b": b"efgh"})
        with self.assertRaisesRegex(ValueError, "base tensor hash"):
            overlay.apply_shard(self.base, path, self.rows, self.root / "out", self.target)
        self.assertFalse((self.root / "out").exists())

    def test_unchanged_tensor_corruption_rejected_by_final_shard_hash(self):
        path = self.export()
        tensor_file(self.base, {"a": b"abcd", "b": b"xxxx"})
        with self.assertRaisesRegex(ValueError, "exported shard"):
            overlay.apply_shard(self.base, path, self.rows, self.root / "out", self.target)
        self.assertTrue((self.root / "out").exists())

    def test_overlay_corruption_rejected(self):
        path = self.export()
        path.write_bytes(path.read_bytes()[:-1] + b"x")
        with self.assertRaisesRegex(ValueError, "tensor hash"):
            overlay.verify_overlay(path, self.rows)

    def test_header_overlap_rejected(self):
        raw = json.dumps({"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                          "b": {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]}}).encode()
        self.base.write_bytes(struct.pack("<Q", len(raw)) + raw + b"12345678")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            overlay.read_header(self.base)

    def test_malformed_headers(self):
        for raw in [b"{}", b'{"x":{},"x":{}}', b'{"x":{"dtype":"BF16","shape":[true],"data_offsets":[0,2]}}']:
            with self.subTest(raw=raw):
                self.base.write_bytes(struct.pack("<Q", len(raw)) + raw + b"ab")
                with self.assertRaises(ValueError):
                    overlay.read_header(self.base)

    def test_source_escape_and_unknown_hf_link_rejected(self):
        directory = self.root / "directory"
        directory.mkdir()
        (directory / "link").symlink_to(self.base)
        with self.assertRaisesRegex(ValueError, "symlink"):
            overlay.inside(directory, "link")
        with self.assertRaisesRegex(ValueError, "symlink"):
            cli.source_file(directory, "link")

    def test_standard_hf_blob_link_allowed(self):
        model = self.root / "model"
        base = model / "snapshots" / ("a" * 40)
        base.mkdir(parents=True)
        blob = model / "blobs" / ("b" * 64)
        blob.parent.mkdir()
        blob.write_bytes(b"content")
        (base / "weights").symlink_to(blob)
        self.assertEqual(cli.source_file(base, "weights").read_bytes(), b"content")

    def test_relative_path_validation(self):
        for name in ["../secret", "/etc/passwd", "nested//file", "a/./b", "a\\b"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                overlay.relative(name)


class RecipeTests(unittest.TestCase):
    def downloads_fixture(self, root, value):
        root.mkdir()
        base = root / "snapshots" / value["base"]["revision"]
        base.mkdir(parents=True)
        auxiliary = root / "vae.pth"
        auxiliary.write_bytes(b"vae")
        value["auxiliary"] = {"bytes": 3, "sha256": sha(b"vae")}
        record = {"schema": 1, "recipe": "nano-original", "recipe_sha256": cli.recipe_digest(value),
                  "base": str(base), "auxiliary": str(auxiliary), "overlay": None}
        (root / "downloads.json").write_text(json.dumps(record))
        return record

    def test_prepare_downloads_passes_bound_paths_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "downloads"
            value = copy.deepcopy(cli.recipe("nano-original"))
            record = self.downloads_fixture(root, value)
            with mock.patch.object(cli, "recipe", return_value=value), mock.patch.object(cli, "materialize") as prepare:
                cli.prepare_downloads("nano-original", root, Path(directory) / "output")
                prepare.assert_called_once_with("nano-original", Path(record["base"]), Path(directory) / "output", None,
                                                link_unmodified=False)

    def test_prepare_downloads_rejects_recipe_drift_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "downloads"
            value = copy.deepcopy(cli.recipe("nano-original"))
            self.downloads_fixture(root, value)
            value["historical_p50_ms"] = 1
            with mock.patch.object(cli, "recipe", return_value=value), mock.patch.object(cli, "materialize") as prepare:
                with self.assertRaisesRegex(ValueError, "binding differs"):
                    cli.prepare_downloads("nano-original", root, Path(directory) / "output")
                prepare.assert_not_called()

    def test_prepare_downloads_rejects_changed_auxiliary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "downloads"
            value = copy.deepcopy(cli.recipe("nano-original"))
            record = self.downloads_fixture(root, value)
            Path(record["auxiliary"]).write_bytes(b"bad")
            with mock.patch.object(cli, "recipe", return_value=value):
                with self.assertRaisesRegex(ValueError, "VAE differs"):
                    cli.prepare_downloads("nano-original", root, Path(directory) / "output")

    def test_exact_three_screen_recipes(self):
        for name, count in [("edge-seed12031", 28), ("edge-seed12032", 28), ("nano-original", 0)]:
            value = cli.recipe(name)
            self.assertEqual(len(value.get("tensors", [])), count)
            self.assertFalse(value["task_quality_certified"])
            self.assertFalse(value["execution"]["servable"])

    def test_plan_uses_only_stdlib_and_does_not_mutate(self):
        code = ("import sys;sys.path.insert(0," + repr(str(PACKAGE)) + ");"
                "from cosmos3_sde1.__main__ import plan;plan('nano-original');"
                "assert 'torch' not in sys.modules and 'numpy' not in sys.modules")
        subprocess.run([sys.executable, "-S", "-B", "-c", code], check=True)

    def test_missing_overlay_url_fails_before_network_or_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            value = copy.deepcopy(cli.recipe("edge-seed12031"))
            value["overlay"]["url"] = None
            with mock.patch.object(cli, "recipe", return_value=value):
                with self.assertRaisesRegex(ValueError, "HTTPS URL"):
                    cli.fetch("edge-seed12031", output)
            self.assertFalse(output.exists())

    def test_reviewed_release_urls_are_file_specs_not_upload_receipts(self):
        prefix = "https://github.com/General-Instinct/InstinctFlash/releases/download/thor-2026-09-15/"
        for name in ("edge-seed12031", "edge-seed12032"):
            value = cli.recipe(name)["overlay"]
            self.assertEqual(value["url"], prefix + value["path"])
        preparation = json.loads((cli.DATA / "release_artifacts.json").read_text())
        self.assertFalse(preparation["uploaded"])

    def test_exact_helper_sources_and_license(self):
        provenance = json.loads((cli.DATA / "provenance.json").read_text())
        helpers = [x for x in provenance["files"] if x["repository"] == "InstinctCompress"]
        self.assertEqual(len(helpers), 8)
        for item in helpers:
            self.assertEqual(overlay.digest(PACKAGE / item["destination"]), item["sha256"])
        self.assertEqual(overlay.digest(PACKAGE / "LICENSE"), provenance["helper_license_sha256"])

    def test_nano_safe_materialization_inventory_and_originals(self):
        self.materialize("nano")

    def test_edge_public_binding_is_inside_strict_artifact_inventory(self):
        self.materialize("edge")

    def test_nano_links_only_unchanged_weights_and_keeps_source_config(self):
        self.materialize("nano", link_unmodified=True)

    def test_edge_never_links_overlay_modified_shard_or_sidecars(self):
        self.materialize("edge", link_unmodified=True)

    def test_cross_filesystem_link_error_has_no_copy_fallback(self):
        import errno
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.safetensors"
            tensor_file(source, {"a": b"abcd"})
            expected = {"path": "source.safetensors", "bytes": source.stat().st_size,
                        "sha256": overlay.digest(source)}
            destination = root / "new/source.safetensors"
            with mock.patch.object(cli.os, "link", side_effect=OSError(errno.EXDEV, "cross-device link")):
                with self.assertRaises(OSError):
                    cli.link_checked_weight(source, destination, expected)
            self.assertFalse(destination.exists())
            self.assertEqual(overlay.digest(source), expected["sha256"])

    def materialize(self, family, *, link_unmodified=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_text(json.dumps({"model": {"config": {}}}))
            (base / "checkpoint.json").write_text("{}")
            tensor_file(base / "unchanged.safetensors", {"b": b"zzzz"})
            header_sha = tensor_file(base / "model.safetensors", {"a": b"abcd"})
            before = {p.name: p.read_bytes() for p in base.iterdir()}
            value = copy.deepcopy(cli.recipe("nano-original"))
            value["family"] = family
            value["files"] = [{"path": p.name, "bytes": p.stat().st_size, "sha256": overlay.digest(p)}
                              for p in base.iterdir()]
            archive = None
            if family == "edge":
                student = root / "student"
                tensor_file(student, {"a": b"1234"})
                rows = [{"native_key": "a", "safetensors_key": "a", "shape": [2], "dtype": "BF16",
                         "bytes": 4, "sha256": sha(b"1234"), "base_tensor_sha256": sha(b"abcd")}]
                archive = root / "overlay"
                value["overlay"] = overlay.export_overlay(student, rows, archive)
                value["tensors"] = rows
                next(row for row in value["files"] if row["path"] == "model.safetensors").update(
                    kind="overlay_shard", header_sha256=header_sha, sha256=overlay.digest(student))
            with mock.patch.object(cli, "recipe", return_value=value):
                result = cli.materialize("test", base, root / "output", archive,
                                         link_unmodified=link_unmodified)
                self.assertFalse(result["task_quality_certified"])
                for name in ("config.json", "checkpoint.json"):
                    self.assertFalse(cli.os.path.samefile(base / name, root / "output" / name))
                self.assertEqual(cli.os.path.samefile(base / "unchanged.safetensors", root / "output/unchanged.safetensors"),
                                 link_unmodified)
                self.assertEqual(cli.os.path.samefile(base / "model.safetensors", root / "output/model.safetensors"),
                                 link_unmodified and family == "nano")
                cli.validate_prepared("test", root / "output")
                if family == "edge":
                    from instinct_compress.artifacts import verify_artifact
                    artifact = verify_artifact(root / "output")
                    self.assertIn("public_preparation.json", {row["path"] for row in artifact["files"]})
                with self.assertRaisesRegex(ValueError, "fresh output"):
                    cli.materialize("test", base, root / "output", archive)
                (root / "output/checkpoint.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed input"):
                    cli.validate_prepared("test", root / "output")
            self.assertEqual(before, {p.name: p.read_bytes() for p in base.iterdir()})

    def test_safe_fixture_geometry(self):
        from cosmos3_sde1.fixture import load_frames
        frames = load_frames(ROOT / "benchmarks/regression/fixtures/recorded_inputs_v1.npz")
        self.assertEqual(len(frames), 12)
        self.assertEqual(frames[0].shape, (540, 640, 3))

    def test_exact_historical_jpeg_sequence_excludes_initial_history_frame(self):
        import io
        import numpy as np
        from PIL import Image
        from cosmos3_sde1.fixture import load_frames

        original = ROOT / "eval/native_total_2026-09-10/fixtures/va_eval_obs.npz"
        if not original.is_file():
            self.skipTest("optional trusted historical provenance archive is not distributed")
        self.assertEqual(overlay.digest(original), "d6f08f968287b78eadd0ef3001e90f0a46397b17294dbd2c46c854feb5b93172")
        with np.load(original, allow_pickle=True) as archive:
            expected = [np.asarray(Image.open(io.BytesIO(bytes(row[0]))).convert("RGB").resize((640, 540)))
                        for row in archive["jpeg_0"][:12]]
        actual = load_frames(ROOT / "benchmarks/regression/fixtures/recorded_inputs_v1.npz")
        for a, b in zip(actual, expected, strict=True):
            self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
