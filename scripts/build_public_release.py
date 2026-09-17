#!/usr/bin/env python3
"""Stage selected public source outside the checkout; optionally build CPU wheels.

No model imports, GPU operations, binary injection or publication.
Reviewed vendored source and its notices are explicitly selected.
The staged files are a reviewable distribution candidate, not release qualification.
"""

from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

try:
    import tomllib
except ImportError:
    import tomli as tomllib


PACKAGES = {
    "instinctflash": (".", "instinctflash"),
    "pi05-iwm": ("examples/pi05_vla", "pi05_iwm"),
    "lingbot-vla-iwm": ("examples/lingbot_vla", "lingbot_vla_iwm"),
    "lingbot-vla-v2-iwm": ("examples/lingbot_vla_v2", "lingbot_vla_v2_iwm"),
    "groot-n17-iwm": ("examples/groot_n17", "groot_n17_iwm"),
    "cosmos3-iwm": ("examples/cosmos3_policy", "cosmos3_iwm"),
    "dreamzero-iwm": ("examples/dreamzero", "dreamzero_iwm"),
}
REGRESSION_FILES = {
    "__init__.py",
    "user_e2e.py",
    "user_report.py",
    "user_provenance.py",
    "native_reference.py",
    "native_loaded.py",
    "runtime_bundle.py",
    "reproduce.py",
    "serve_smoke.py",
}
CONFIG_FILES = {
    "adapters.json",
    "arms.ci.json",
    "arms.instinctflash.json",
    "arms.template.json",
    "campaign.groot_libero.json",
    "campaign.pi05_libero_capture.json",
    "campaign.vla2_robotwin.json",
    "campaign.vla4_robotwin.json",
    "environment.lock.json",
    "registry.json",
    "simulator_routes.json",
    "sweep.pi05_libero_nfe.json",
    "sweep.pi05_libero_nfe1_cert.json",
    "sweep.wan_va_guidance_nfe.json",
}
ROOT_FILES = {"README.md", "LICENSE", "INSTALL.rst", "REPRODUCE.rst", "pyproject.toml"}
ADAPTER_LICENSE_FILES = {directory + "/LICENSE" for directory, _ in PACKAGES.values() if directory != "."}
TOOL_FILES = {
    "scripts/build_public_release.py",
    "scripts/audit_release_scope.py",
    "scripts/check_installed_package.py",
    "scripts/prepare_pi05_libero.py",
    "scripts/install_pi05_transformers.py",
    "scripts/summarize_pi05_sm120.py",
    "scripts/analyze_pi05_diagnostic.py",
    "scripts/summarize_pi05_frozen.py",
    "scripts/summarize_pi05_sm120_fusion.py",
    "examples/pi05_vla/requirements-sm120.lock",
    "examples/pi05_vla/verify_checkpoint_reference.py",
    "examples/pi05_vla/sm120_fp8_results.json",
    "examples/pi05_vla/sm120_libero_screen_results.json",
    "examples/pi05_vla/sm120_checkpoint_reference_results.json",
    "examples/pi05_vla/sm120_libero_matched_results.json",
    "examples/pi05_vla/sm120_task5_diagnostic_results.json",
    "examples/pi05_vla/sm120_frozen_results.json",
    "examples/pi05_vla/sm120_fusion_results.json",
    "examples/pi05_vla/verify_sm120_fp8.py",
    "examples/pi05_vla/reproduce_sm120_libero.py",
}
SCOPE_PATH = "release/oss_scope.json"
PROFILE_MIRROR = "benchmarks/regression/fixtures/deployment_profiles.json"
HELD_PREFIXES = ("instinctflash/train", "instinctflash/distill", "serving", "eval")
WORKER = "instinctflash/runtime/lingbot_worker.py"
WORKER_SOURCE = "eval/lingbot_va_robotwin/serve_variant.py"
WORKER_SHA = "7f1cae4602c95ed5053e69e2631f7b2d33f9f6ebd61115791792ed96237b942d"
# Root owns the portable non-pickle fixture/profile selection. No historical archive fallback.
SELECTED_DATA: dict[str, str] = {
    "benchmarks/regression/fixtures/recorded_inputs_v1.npz": "37843e22fa6dd9a2abf2bae390ccb8e5c4319446e3d3fab5d0d477860065f411",
    "benchmarks/regression/fixtures/recorded_inputs_v1.json": "67ae26485f141e491951ee80803d30c6e73a6661f6175a078ca154ed60324ae7",
    "benchmarks/regression/fixtures/LICENSE.RoboTwin.txt": "c695d421718e54e6f3a60858f0c601125c3ecc60c199332c15947360561e2a9f",
    "benchmarks/regression/fixtures/recorded_inputs_v1_attribution.json": "dd3f214f3e6aaaedf66debd24b56f0e7af0f9534ddd556a69a1ad359fcba6e7a",
}
FULL_TEST_FIXTURES: dict[str, str] = {
    "eval/cosmos3_task_quality_2026-09-14/preflight/robolab120_task_inventory.json": "f3d2995cca4c3eaa39928454b4c0c1cd8985cc4ac16a3dcdad0895e243902822",
    "eval/cosmos3_task_quality_2026-09-14/rtx_preparation/robolab_smoke_live_bindings_v1.json": "cfa6614cba7051aa92bb7a75663c95555fb0fb581553e6b7deaeea66dc88e620",
}
SELECTED_CONTROLS: dict[str, str] = {
    "release/deployment_profiles.json": "8187485677cb6ab1192960fa80481b913ab8e03d9cab9de8599589fc1038cfac",
    "scripts/bootstrap_vendor.py": "2b50642ead7542aa7239fabd3da5f11a7c3e1b2e083ccf772826204acd1a28d8",
    "scripts/prepare_auxiliary_assets.py": "e44be6692297d24984075c2d5b1aa11823ab361776a3407fc711224e053e3a2d",
    "scripts/prepare_native_tools.py": "243720e214c94b6a5030e275c0220c0fb0a0f5a123eeffea5927213942e7e3b7",
    "scripts/public_deploy.py": "7f6ccb9e52683a7bab6cbabb86f80e110978af26473e9940c89853f3870f4fcc",
    "scripts/repair_vendor_wheel.py": "1f62f053a37b29966202111da5262cb3da0464bba2d4697721cce97f1ea310ff"
}
PUBLIC_VENDOR_FILES: dict[str, str] = {
    "release/vendor/README.rst": "5e6159d73dd42d682c22f1c1c73f9dd1f347f7c026e23312ef6c39ad790b99f7",
    "release/vendor/asset_profiles.json": "ba58e0939bb1a00a96e0c4f341361d0842d5f74088c08ab970c665323d5b5304",
    "release/vendor/auxiliary_assets.json": "68c8d84e7290920bdcb090b2eb858668211463b7869e6dee5f78c2e51f3c5183",
    "release/vendor/cosmos/LICENSE": "6bd3fdb9356edb6e4c1f00ad9cd6639a1cf06ca0a415a5071fd70d68799209e6",
    "release/vendor/cosmos/NOTICE": "0a3ebe37fb632d02a6a4d1cfce96109bb026b7833d8a3f0853091bb303c21ad9",
    "release/vendor/cosmos/hf_tool.json": "7f12f1e464690b79cd9a1822e16ec23b337be4e7e585afd7e6ce7adc662c100c",
    "release/vendor/cosmos/hf_tool_constraints.txt": "c0d7ed87af5e6b8909031f42dfc7fd8f4b959359875ac3cf4401fc3af9945edc",
    "release/vendor/cosmos/inference_packaging.patch": "f2384b6829b0e1df1067043064859c922df4886afcbad367a2ce35a52db309d0",
    "release/vendor/cosmos/recorded_thor.patch": "f04f1af5d70c7ec95665fa0fe11b1ef3849454688de40b4102ca668b631403f5",
    "release/vendor/cosmos/source.json": "60ea2649af252573aae7475db83a7fb1fe108cc6c9e9d5505638ee6b92643df0",
    "release/vendor/dreamzero/LICENSE": "8f2f9955895e664d84a98ff47debe8d4aaa76665afec7d8d3f1efa36c395d849",
    "release/vendor/dreamzero/bootstrap.json": "799a14d8091b8691e609a05a748e4724df0cf5b6f66dec302c3a032dfed94cd9",
    "release/vendor/dreamzero/constraints.txt": "3ee61277c758f6f2f0a860e2523172e1527049cba058bae95abad3fc0c9e533c",
    "release/vendor/dreamzero/inference_packaging.patch": "0b0d010dabb034f6aeda4433e986c3f5a78dd1d3083391300879ad58b51a9d6f",
    "release/vendor/dreamzero/inference_requirements.txt": "0ed2162cdd19f280d441557cc774ff82a3693f43f82f2bc797fabae3220da246",
    "release/vendor/dreamzero/pins.json": "b7662cdb76d4fd862b495dc0ea4c85ae6a1bb6e177a6d4ce3bfdaaab68518b2f",
    "release/vendor/dreamzero/source.json": "981683765d33cf74f5b40ca4dbd478d57b01a23b2ea66effbfca283f411fb5a7",
    "release/vendor/edge/bootstrap.json": "40507a7e4737b8558302ef1a594910ce39c9ba38a9000f3925fb24f858e47ed6",
    "release/vendor/edge/constraints.txt": "c19a61c904c304b17c8430782c97994bf68ad2e8529885f8bb7925cc6c4a0c32",
    "release/vendor/edge/inference_requirements.txt": "b496c347752b7654ec3f949c8681f2c76e46992167277b9ef0a74c1dcd6eab9f",
    "release/vendor/edge/pins.json": "d55770b413b9a9da5f407369140f70899c215082973529426eb77735d171f783",
    "release/vendor/groot/LICENSE": "93363510f62ce25ad1a47605b0b06ab335edd82f1a444531c58dd30b6e5a5a0c",
    "release/vendor/groot/bootstrap.json": "bc982c9985c06035b191fbf3aadffde70374f3038603a40f9b9d41cca6fb9cf0",
    "release/vendor/groot/constraints.txt": "14606111390b2995c4bfb9e3bcf8463bf221aa7c26b113b210810f8e0ad65b08",
    "release/vendor/groot/inference_packaging.patch": "26b41d18c6a48e0ba0647f769f845b406f398a0743358381ae3fcdb3cb52dbfb",
    "release/vendor/groot/inference_requirements.txt": "26e532621126044dfb603dd47eabe93f7b89d4e0ff5bf85580b95706c59f6473",
    "release/vendor/groot/pins.json": "d3464fae01e66b0e0eb60754b4e38e5a0eaf3bba52c08d12967c6fc402dfcb69",
    "release/vendor/groot/source.json": "0f7f40b5bc522abb95b1680172f244979cf32d281e812d58cc0c3da05d778ff3",
    "release/vendor/nano/bootstrap.json": "dfa300b91d35e0c52e661b4235c7926ffbb8c3b6ac36ad58effc932512275021",
    "release/vendor/nano/constraints.txt": "c19a61c904c304b17c8430782c97994bf68ad2e8529885f8bb7925cc6c4a0c32",
    "release/vendor/nano/inference_requirements.txt": "b496c347752b7654ec3f949c8681f2c76e46992167277b9ef0a74c1dcd6eab9f",
    "release/vendor/nano/pins.json": "155ea8551b612cf46d32e874d1e6a9c70b3bfe3d14b5567b3243ebc7673fcdf3",
    "release/vendor/pi05/bootstrap.json": "142e10fed092cecc388bf28a6a6ca0fb3f5556bded4bb239e767d196019d53bd",
    "release/vendor/pi05/constraints.txt": "905c172e03803e94334ead5e5dd548ac1d6c2774bdad144aca2aca0b912f5482",
    "release/vendor/pi05/inference_requirements.txt": "9444ae3caae3d37abb71c8e1755f70cb7435d658167c18c1601c17130f743f09",
    "release/vendor/pi05/pins.json": "eeff3c94ac9e01c433e7f13cc1fdc78e1585bc7565002db570fdc12403501af3",
    "release/vendor/qualification.json": "765f42b37c6916fe623069f23ca8fb8797ba51c4b42bed98a060225d3bde1f3e",
    "release/vendor/va/LICENSE.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "release/vendor/va/bootstrap.json": "531db91ca886922fe5903e124efe03561a9cb7ebc70678edec4c0f295fbbf5bd",
    "release/vendor/va/constraints.txt": "918ac1422e84629c289428486c603cbabdab8d098a884ff325cf6b0984ad4252",
    "release/vendor/va/inference_packaging.patch": "9e255aaf463683181f13c0104b678323817680833d0537543629ce767cbf2f72",
    "release/vendor/va/inference_requirements.txt": "8d9c6b5fea6a0278b41ba3b6302272d8ef8175d0065c6eb350ddc44606630c20",
    "release/vendor/va/pins.json": "b2df32a505ee3086db3a7d7428806392a85970374286142a4d03c3fe8e5f2093",
    "release/vendor/va/recorded_thor.patch": "1292afe206e94a3aa1538af72abb71d50a690c58aa96430a855de4a3ba736730",
    "release/vendor/va/source.json": "c126540cb08954c0b6976258d966bd5d6e2ae27f2c58eee8d01ab6db277c3a0c",
    "release/vendor/vla2/LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "release/vendor/vla2/bootstrap.json": "acb0847ee46ef0219eeb843ee130283f2fc27254defef198bc05cd939ddf55d5",
    "release/vendor/vla2/bootstrap_constraints.txt": "bfc0bd893830a1c247e13382bfeb2dc11cb1aec47df65f275ee5d7ddb13bdf4f",
    "release/vendor/vla2/constraints.txt": "c37366ae64ac9582bb5617cc5eae199fd4140df8c3544b75350ea423df5f6122",
    "release/vendor/vla2/inference_packaging.patch": "0ab6d53c4d2e0cee3af5656a19fa313080561dcac96f5f67e074d990e3626d63",
    "release/vendor/vla2/inference_requirements.txt": "bdafe50b25049c529941ddb43d1d3a8f391f28d95ce4594ef5ddf5119d090797",
    "release/vendor/vla2/lerobot_inference.json": "40c38453be0d4815023779523e784f3ca28f22a1c905c9edb1a4538e48c736ec",
    "release/vendor/vla2/pins.json": "df00853acb2952291cb824e6294f9d42a66f84ce7b983839e4e2cc1e03dc5678",
    "release/vendor/vla2/recorded_thor.patch": "ef8a413873b43f0ee6a916b2860b62108fa06d4818dccae0560c7e9f7b45c4ee",
    "release/vendor/vla2/source.json": "8312825b7e56b586d437f53188952f42f05869af752b68a47b9e962acaa14874",
    "release/vendor/vla4/LICENSE": "63b052b2e017d82d657a9cbcb6289fb61f830a61efe366eaa0bca9a9ca8875cd",
    "release/vendor/vla4/bootstrap.json": "86c8223119889485765a465feef1eb18e5fadf7058b5641b5b18fa7ea4017691",
    "release/vendor/vla4/bootstrap_constraints.txt": "9ae49b0e047f34eb4f91959ac8503476f7f4c10a61ee2762333a97566c34a7c6",
    "release/vendor/vla4/constraints.txt": "f99cb224e4dcf67a1c2ab7ede9e05cea28e7d2a0016077b4558a68d3196d881d",
    "release/vendor/vla4/inference_packaging.patch": "0512b4429a27bd80425aa77e538a7958893402b3d82b127d4ff0ac36fac9d7e7",
    "release/vendor/vla4/inference_requirements.txt": "4cb58e6a108160dc5e168c64115aeb3ad7db8171145741c44751477504fea8e6",
    "release/vendor/vla4/lerobot_inference.json": "40c38453be0d4815023779523e784f3ca28f22a1c905c9edb1a4538e48c736ec",
    "release/vendor/vla4/pins.json": "6fbe3e5e39b05b97cde2d3f0ce1ee0c74a8f3ab511ef979a1eaa33f99b9b693d",
    "release/vendor/vla4/recorded_thor.patch": "40b964dc7333ee0eab260e2af1af84c3d1ce783722f828f34c8034a1348562a3",
    "release/vendor/vla4/source.json": "32ec13ce8c34eb2f0557eea8c16def9bff4c709e0e87d540b3c5f4b4c6358ff0",
    "release/vendor/wheel_repairs.json": "f8fc3e75d4cdb3dc95f3f3f90ae8dfe619c64d6c10b5919aa73ec66baefdedbb"
}

GENERATED = {
    ".git",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
    ".venv",
    "results",
    "logs",
}
FULL_SOURCE_PREFIXES = (
    "instinctflash",
    "serving/flash_rt",
    "serving/csrc",
    "serving/flash_wm",
    "serving/training",
    "serving/tests",
    "serving/tools",
    "serving/examples",
    "examples/cosmos3_sde1",
)
FULL_EXCLUDED_PREFIXES = ("serving/third_party", "serving/training/_runs")
FULL_EXTENSIONS = {
    ".py",
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".cuh",
    ".cmake",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".rst",
    ".txt",
}
FULL_FILES = {
    # Exact existing adapter source/benchmark surfaces required by public tests.
    "examples/cosmos3_policy/README.md",
    "examples/cosmos3_policy/launch_robolab_stock.py",
    "examples/cosmos3_policy/measure_openpi_ws.py",
    "examples/cosmos3_policy/measure_predict.py",
    "examples/cosmos3_policy/reproduce_h100.sh",
    "examples/dreamzero/README.md",
    "examples/dreamzero/cfg_batch.py",
    "examples/dreamzero/diag_batch.py",
    "examples/dreamzero/measure_dreamzero.py",
    "examples/dreamzero/reproduce_h100.sh",
    "examples/dreamzero/step_cache.py",
    "examples/dreamzero/verify_cfg_batch.py",
    "examples/groot_n17/README.md",
    "examples/groot_n17/benchmark_runtime.py",
    "examples/groot_n17/config.json",
    "examples/groot_n17/fastpath_results.json",
    "examples/groot_n17/instinctflash.json",
    "examples/groot_n17/profile_runtime.py",
    "examples/groot_n17/reproduce_h100.py",
    "examples/groot_n17/reproduce_h100.sh",
    "examples/groot_n17/reproduce_h100_results.json",
    "examples/groot_n17/static_capture.py",
    "examples/groot_n17/verify_backbone_fastpath.py",
    "examples/groot_n17/verify_fast_decode.py",
    "examples/groot_n17/verify_fastpaths.py",
    "examples/groot_n17/verify_static_capture.py",
    "examples/lingbot_vla/README.md",
    "examples/lingbot_vla/profile_infer.py",
    "examples/lingbot_vla/reproduce_h100.sh",
    "examples/lingbot_vla/static_capture_results.json",
    "examples/lingbot_vla/verify_static_capture.py",
    "examples/lingbot_vla_v2/README.md",
    "examples/lingbot_vla_v2/instinctwm.json",
    "examples/lingbot_vla_v2/moe_kernel_results.json",
    "examples/lingbot_vla_v2/profile_infer.py",
    "examples/lingbot_vla_v2/reproduce_h100.py",
    "examples/lingbot_vla_v2/reproduce_h100.sh",
    "examples/lingbot_vla_v2/reproduce_h100_results.json",
    "examples/lingbot_vla_v2/static_capture.py",
    "examples/lingbot_vla_v2/static_capture_results.json",
    "examples/lingbot_vla_v2/verify_moe_kernel.py",
    "examples/lingbot_vla_v2/verify_static_capture.py",
    "examples/pi05_vla/README.md",
    "examples/pi05_vla/certify_tf32_closed_loop.py",
    "examples/pi05_vla/emit_tf32_closed_loop_outcomes.py",
    "examples/pi05_vla/instinctwm.json",
    "examples/pi05_vla/measure_chunk_cost.py",
    "examples/pi05_vla/reproduce_h100.py",
    "examples/pi05_vla/reproduce_h100.sh",
    "examples/pi05_vla/run_pi05_end_to_end.py",
    "examples/pi05_vla/run_tf32_closed_loop.py",
    "examples/pi05_vla/static_capture_results.json",
    "examples/pi05_vla/tf32_closed_loop_preregistration.json",
    "examples/pi05_vla/tf32_static_h100_results.json",
    "examples/pi05_vla/tf32_v044_static_h100_results.json",
    "examples/pi05_vla/verify_capture_equivalence.py",
    "examples/pi05_vla/verify_static_capture.py",
    "examples/pi05_vla/verify_tf32_operating_point.py",
    "serving/pyproject.toml",
    "serving/setup.py",
    "serving/CMakeLists.txt",
    "serving/LICENSE",
    "serving/scripts/build_thor_fa2.py",
    "serving/scripts/build_thor_fa2.sh",
    "serving/scripts/build_thor_native.py",
    "serving/scripts/build_thor_native.sh",
    "scripts/check_release.sh",
    "tests/test_public_release_builder.py",
    "tests/test_public_reproduce.py",
    "scripts/build_clean_wheels.py",
    "scripts/bootstrap_framework_compare.py",
    "release/OSS_RELEASE.rst",
    "benchmarks/regression/FRAMEWORK_COMPARISON.rst",
    "benchmarks/regression/SCREEN_REPLAY.rst",
    "benchmarks/regression/fixtures/frameworks/catalog.json",
    "benchmarks/regression/fixtures/frameworks/sources.json",
    "benchmarks/regression/fixtures/frameworks/omni_thor_startup_patch.json",
    "benchmarks/regression/fixtures/frameworks/omni_thor_startup_patch_portable_v2.json",
    "tests/test_public_deploy.py",
    "tests/test_public_reproduction_inputs.py",
    "tests/test_user_report.py",
    "tests/test_public_serving_smoke.py",
    "tests/test_public_schedule_ranking.py",
    "tests/test_thor_fa2_builder.py",
    "tests/test_thor_native_builder.py",
    "benchmarks/regression/systemd/instinctflash-thor-regression.service",
    "benchmarks/regression/systemd/instinctflash-thor-regression.timer",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution_modules(distribution, module):
    # This reviewed wheel intentionally includes only the eight-file Compress helper subset.
    return (
        (module, "instinct_compress")
        if distribution == "instinctflash-cosmos3-sde1"
        else (module,)
    )


def safe_relative(value):
    require(
        isinstance(value, str)
        and value
        and "\\" not in value
        and not PurePosixPath(value).is_absolute()
        and all(part not in {"", ".", ".."} for part in value.split("/")),
        "invalid relative source path",
    )
    return value


def under(path, prefix):
    return path == prefix or path.startswith(prefix + "/")


def held(path, scope="core"):
    if scope == "full" and path in FULL_TEST_FIXTURES:
        return False
    prefixes = HELD_PREFIXES if scope == "core" else ("eval", *FULL_EXCLUDED_PREFIXES)
    return any(under(path, prefix) for prefix in prefixes)


def full_source(relative):
    path = PurePosixPath(relative)
    if any(
        part in GENERATED or part.endswith(".egg-info") or part.startswith(".venv-")
        for part in path.parts
    ):
        return False
    if held(relative, "full"):
        return False
    if not any(under(relative, prefix) for prefix in FULL_SOURCE_PREFIXES):
        return False
    return (
        path.suffix in FULL_EXTENSIONS
        or path.name in {"CMakeLists.txt", "LICENSE", "NOTICE", "COPYING"}
        or path.name.startswith(("LICENSE.", "NOTICE."))
    )


def classification(path, scope):
    matches = [rule for rule in scope["rules"] if under(path, rule["prefix"])]
    return (
        max(matches, key=lambda rule: len(rule["prefix"]))["classification"]
        if matches
        else "unclassified"
    )


def sha(content):
    return hashlib.sha256(content).hexdigest()


def package_data_matches(relative, pattern):
    # setuptools package-data uses path globs: a plain '*' does not cross '/'.
    parts, patterns = relative.split("/"), pattern.split("/")
    require(
        "**" not in patterns,
        "recursive package-data glob needs an explicit staging review",
    )
    return len(parts) == len(patterns) and all(
        fnmatch.fnmatchcase(part, glob) for part, glob in zip(parts, patterns)
    )


def git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args])


def source_bytes(repository, relative):
    """Resolve every component before reading, including paths with symlink ancestors."""
    safe_relative(relative)
    original = repository / relative
    resolved = original.resolve(strict=True)
    require(
        resolved.is_relative_to(repository) and resolved.is_file(),
        f"source target must be a file inside checkout: {relative}",
    )
    target = resolved.relative_to(repository).as_posix()
    content = resolved.read_bytes()
    require(
        original.resolve(strict=True) == resolved,
        f"source target changed while reading: {relative}",
    )
    return content, target, original != resolved


def selected_python(relative):
    path = PurePosixPath(relative)
    if (
        path.suffix != ".py"
        or any(
            part in GENERATED or part.endswith(".egg-info") or part.startswith(".venv-")
            for part in path.parts
        )
        or held(relative)
    ):
        return False
    if under(relative, "instinctflash"):
        return (
            not under(relative, "instinctflash/native")
            or relative == "instinctflash/native/__init__.py"
        )
    if relative == "benchmarks/__init__.py" or under(relative, "benchmarks/vla"):
        return True
    if relative.startswith("benchmarks/regression/"):
        return relative.removeprefix("benchmarks/regression/") in REGRESSION_FILES
    return any(
        under(relative, directory + "/" + module)
        for directory, module in PACKAGES.values()
        if directory != "."
    )


def metadata_docs(content, directory):
    project = tomllib.loads(content.decode())["project"]
    names = []
    readme = project.get("readme")
    if isinstance(readme, str):
        names.append(readme)
    elif isinstance(readme, dict) and "file" in readme:
        names.append(readme["file"])
    license_ = project.get("license")
    if isinstance(license_, dict) and "file" in license_:
        names.append(license_["file"])
    result = []
    for name in names:
        safe_relative(name)
        require(
            PurePosixPath(name).suffix.lower() in {".md", ".rst", ".txt"}
            or name == "LICENSE",
            "package metadata may select only a documentation file",
        )
        result.append(name if directory == "." else directory + "/" + name)
    return result


def replace_section(text, section, body):
    lines = text.splitlines(keepends=True)
    header = f"[{section}]"
    start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
    replacement = header + "\n" + body.rstrip() + "\n\n"
    if start is None:
        return text.rstrip() + "\n\n" + replacement
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    return "".join(lines[:start]) + replacement + "".join(lines[end:])


def staged_pyproject(content, *, has_fixture=False):
    original = tomllib.loads(content.decode())
    require(
        original.get("build-system", {}).get("build-backend")
        == "setuptools.build_meta",
        "public staging supports the reviewed setuptools backend only",
    )
    package_config = original.get("tool", {}).get("setuptools", {}).get("packages", {})
    require(
        isinstance(package_config, dict) and set(package_config) == {"find"},
        "unexpected core package discovery declaration",
    )
    discovery = {
        "where": ["."],
        "include": [
            "instinctflash*",
            "benchmarks",
            "benchmarks.vla*",
            "benchmarks.regression*",
        ],
        "exclude": ["instinctflash.train*", "instinctflash.distill*"],
    }
    data = {
        "benchmarks.vla": ["config/*.json"],
        "benchmarks.regression": ["fixtures/deployment_profiles.json"],
    }
    if has_fixture:
        data["benchmarks.regression"].extend(
            [
                "fixtures/recorded_inputs_v1.npz",
                "fixtures/recorded_inputs_v1.json",
                "fixtures/recorded_inputs_v1_attribution.json",
                "fixtures/LICENSE.RoboTwin.txt",
            ]
        )
    text = replace_section(
        content.decode(),
        "tool.setuptools.packages.find",
        "\n".join(f"{key} = {json.dumps(value)}" for key, value in discovery.items()),
    )
    text = replace_section(
        text,
        "tool.setuptools.package-data",
        "\n".join(
            f"{json.dumps(key)} = {json.dumps(value)}" for key, value in data.items()
        ),
    )
    expected = json.loads(json.dumps(original))
    expected["tool"]["setuptools"]["packages"] = {"find": discovery}
    expected["tool"]["setuptools"]["package-data"] = data
    require(
        tomllib.loads(text) == expected,
        "staged metadata transform changed unrelated configuration",
    )
    return text.encode()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def create_stage(repository, output, *, scope="core"):
    require(scope in {"core", "full"}, "unknown source scope")
    repository = Path(repository).resolve(strict=True)
    requested = Path(output).absolute()
    require(
        not requested.exists() and not requested.is_symlink(),
        "refusing existing output",
    )
    output = requested.resolve()
    require(
        not output.is_relative_to(repository) and not repository.is_relative_to(output),
        "output must be outside source checkout",
    )
    scope_content, scope_target, _ = source_bytes(repository, SCOPE_PATH)
    require(scope_target == SCOPE_PATH, "scope proposal must not be a symlink")
    proposal = json.loads(scope_content)
    require(proposal["schema"] == "instinctflash.oss_scope.v1", "unknown source scope")
    head = git(repository, "rev-parse", "HEAD").decode().strip()
    inventory = (
        git(
            repository,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "instinctflash",
            "benchmarks",
            "examples",
            "serving",
            "tests",
        )
        .decode()
        .split("\0")
    )
    deleted = set(git(repository, "ls-files", "--deleted", "-z").decode().split("\0"))
    inventory = [name for name in inventory if name not in deleted]
    selected = {name for name in inventory if name and selected_python(name)}
    packages = dict(PACKAGES)
    controls = dict(SELECTED_CONTROLS)
    if scope == "full":
        selected.update(name for name in inventory if name and full_source(name))
        selected.update(
            name
            for name in inventory
            if name
            and name.endswith(".py")
            and (under(name, "tests") or under(name, "benchmarks"))
            and not any(part in GENERATED for part in PurePosixPath(name).parts)
        )
        selected.update(FULL_FILES)
        packages["flash-rt"] = ("serving", "flash_rt")
        packages["instinctflash-cosmos3-sde1"] = (
            "examples/cosmos3_sde1",
            "cosmos3_sde1",
        )
        controls.update(PUBLIC_VENDOR_FILES)
        controls.update(FULL_TEST_FIXTURES)
    selected.update(ROOT_FILES | ADAPTER_LICENSE_FILES | TOOL_FILES | {SCOPE_PATH, PROFILE_MIRROR})
    selected.update("benchmarks/vla/config/" + name for name in CONFIG_FILES)
    selected.update(SELECTED_DATA)
    selected.update(controls)
    metadata = {}
    documentation = set(ROOT_FILES | ADAPTER_LICENSE_FILES)
    for distribution, (directory, module) in packages.items():
        relative = (
            "pyproject.toml" if directory == "." else directory + "/pyproject.toml"
        )
        content, _, _ = source_bytes(repository, relative)
        config = tomllib.loads(content.decode())
        require(
            config["project"]["name"] == distribution,
            "adapter distribution identity changed",
        )
        require(
            config["build-system"]["build-backend"] == "setuptools.build_meta",
            "unexpected package build backend",
        )
        package_init = (
            module + "/__init__.py"
            if directory == "."
            else directory + "/" + module + "/__init__.py"
        )
        require(package_init in selected, f"missing selected package: {distribution}")
        metadata[relative] = content
        documentation.update(metadata_docs(content, directory))
    selected.update(metadata)
    selected.update(documentation)
    require(WORKER in selected, "reviewed worker closure is required")
    contents, entries = {}, {}
    for relative in sorted(selected):
        content, target, linked = source_bytes(repository, relative)
        worker = relative == WORKER
        require(not held(relative, scope), f"held destination selected: {relative}")
        if worker:
            require(
                target in {WORKER_SOURCE, WORKER} and sha(content) == WORKER_SHA,
                "reviewed worker source/hash differs",
            )
        else:
            require(
                not held(target, scope)
                and (scope == "full" or classification(target, proposal) != "hold"),
                f"held symlink target: {relative}",
            )
        explicit_review = (
            relative in documentation
            or relative in metadata
            or relative == SCOPE_PATH
            or relative in controls
        )
        full_review = scope == "full" and (
            relative in FULL_FILES
            or full_source(relative)
            or under(relative, "tests")
            or under(relative, "benchmarks")
        )
        require(
            worker
            or explicit_review
            or full_review
            or classification(relative, proposal) == "public_candidate",
            f"unreviewed source selected: {relative}",
        )
        require(
            worker
            or explicit_review
            or (
                scope == "full"
                and (
                    target in FULL_FILES
                    or full_source(target)
                    or under(target, "tests")
                    or under(target, "benchmarks")
                )
            )
            or classification(target, proposal) == "public_candidate",
            f"unreviewed symlink target: {relative}",
        )
        if relative in SELECTED_DATA:
            require(
                sha(content) == SELECTED_DATA[relative],
                f"selected data hash differs: {relative}",
            )
        if relative in controls:
            require(
                target == relative and sha(content) == controls[relative],
                f"selected control path/hash differs: {relative}",
            )
        if relative.endswith(".py"):
            ast.parse(content, filename=relative)
        before = sha(content)
        transforms = []
        if relative == "pyproject.toml" and scope == "core":
            content = staged_pyproject(
                content,
                has_fixture="benchmarks/regression/fixtures/recorded_inputs_v1.npz"
                in SELECTED_DATA,
            )
            transforms.append(
                "staged core discovery excludes train/distill; package-data selects public benchmark data only"
            )
        if linked:
            transforms.append(
                "materialize regular file from validated in-checkout target"
            )
        if worker:
            transforms.append(
                "explicit reviewed historical worker exception; exact source bytes preserved"
            )
        entries[relative] = {
            "source_path": relative,
            "resolved_source_path": target,
            "source_sha256": before,
            "sha256": sha(content),
            "bytes": len(content),
            "transforms": transforms,
        }
        contents[relative] = content
    require(all(contents[name] == contents["LICENSE"] for name in ADAPTER_LICENSE_FILES),
            "adapter license copy differs from the established root license")
    profile_destination = PROFILE_MIRROR
    profile_source = "release/deployment_profiles.json"
    require(
        contents[profile_destination] == contents[profile_source],
        "packaged deployment profile mirror drifted",
    )
    entries[profile_destination]["canonical_source_path"] = profile_source
    entries[profile_destination]["transforms"].append(
        "checked source mirror equals canonical deployment profiles; no staged-only data injection"
    )
    for distribution, (directory, module) in packages.items():
        metadata_path = (
            "pyproject.toml" if directory == "." else directory + "/pyproject.toml"
        )
        package_data = (
            tomllib.loads(contents[metadata_path].decode())
            .get("tool", {})
            .get("setuptools", {})
            .get("package-data", {})
        )
        for relative, entry in entries.items():
            local = (
                relative if directory == "." else relative.removeprefix(directory + "/")
            )
            modules = distribution_modules(distribution, module)
            if (
                directory == "."
                and (under(local, module) or under(local, "benchmarks"))
            ) or (
                directory != "."
                and relative.startswith(directory + "/")
                and any(under(local, owner) for owner in modules)
            ):
                entry["wheel_required"] = local.endswith(".py") or any(
                    local.startswith(package.replace(".", "/") + "/")
                    and package_data_matches(
                        local[len(package.replace(".", "/")) + 1 :], pattern
                    )
                    for package, patterns in package_data.items()
                    for pattern in patterns
                )
    # Fail before creating output when any input, source boundary or transform is invalid.
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for relative, content in contents.items():
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
        destination.chmod(
            0o755
            if relative.endswith(".sh")
            or ("scripts/" in relative and relative.endswith(".py"))
            else 0o644
        )
    manifest = {
        "schema": "instinctflash.public_release_stage.v1",
        "status": "staged_for_review",
        "repository": str(repository),
        "head": head,
        "source": str(source),
        "files": entries,
        "scope_sha256": sha(scope_content),
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "packages": packages,
        "source_scope": scope,
        "wheels": [],
        "publication_ready": False,
        "pipeline_validated": False,
        "native_binaries": [],
        "selected_fixture": "benchmarks/regression/fixtures/recorded_inputs_v1.npz"
        if "benchmarks/regression/fixtures/recorded_inputs_v1.npz" in SELECTED_DATA
        else None,
        "limitations": [
            "Source classifications and this staging manifest do not authorize publication or certify licensing.",
            (
                "Full internal inference/native/serving/train/distill source is selected; no binaries, downloaded CUTLASS checkout, weights or historical bulk evaluation archives are staged."
                if scope == "full"
                else "No native accelerator implementations, serving implementation, training/distillation source or historical bulk evaluation archives are staged."
            ),
            "Current README/INSTALL/LICENSE are explicitly copied; historical linked evidence and external vendor dependencies are not copied automatically.",
            "No GPU inference, fine-tune compatibility, hardware speed, simulator quality or full reproduction pipeline is qualified by a CPU build.",
            "Public fixture/profile availability is reported explicitly; no pickle archive or machine-specific historical matrix is substituted.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def inspect_wheel(path, distribution, directory, module, files, *, scope="core"):
    """Inspect bytes/metadata without importing any package or extracting archive paths."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate wheel member")
        for name in names:
            safe_relative(name.rstrip("/"))
            require(not name.endswith("/"), "unexpected directory entry in wheel")
            require(
                not re.search(r"\.(so(?:\.\d+)*|pyd|dll|dylib|a|o)$", name),
                "native binary entered public wheel",
            )
            require(
                (archive.getinfo(name).external_attr >> 16) & 0o170000 != 0o120000,
                "symlink entered public wheel",
            )
            if ".dist-info/" in name:
                continue
            relative = name if directory == "." else directory + "/" + name
            require(
                relative in files and not held(relative, scope),
                f"wheel contains unselected source: {name}",
            )
            require(
                sha(archive.read(name)) == files[relative]["sha256"],
                f"wheel source bytes differ: {name}",
            )
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        require(len(metadata_names) == 1, "one wheel distribution metadata required")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))

        def normalize(value):
            return re.sub(r"[-_.]+", "-", value).lower()

        require(
            normalize(metadata["Name"]) == normalize(distribution),
            "wheel distribution identity differs",
        )
        require(module + "/__init__.py" in names, "wheel lacks its public package")
        expected = {
            relative if directory == "." else relative[len(directory) + 1 :]
            for relative in files
            if files[relative].get("wheel_required", True)
            and (under(relative, module) or under(relative, "benchmarks"))
            if directory == "."
        }
        if directory != ".":
            expected = {
                relative[len(directory) + 1 :]
                for relative in files
                if files[relative].get("wheel_required", True)
                and any(
                    under(relative, directory + "/" + owner)
                    for owner in distribution_modules(distribution, module)
                )
            }
        require(
            expected <= set(names), "wheel dropped selected Python or benchmark data"
        )
        return {
            "distribution": metadata["Name"],
            "version": metadata["Version"],
            "members": len(names),
            "selected_source_files": len(expected),
            "native_binaries": 0,
        }


def build_wheels(manifest, output, *, python):
    output = Path(output).resolve()
    source = Path(manifest["source"])
    wheels, logs = output / "wheels", output / "logs"
    wheels.mkdir(exist_ok=False)
    logs.mkdir(exist_ok=False)
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        PIP_NO_INDEX="1",
        UV_OFFLINE="1",
    )
    for key in list(environment):
        if key == "PYTHONPATH" or key.startswith("IFL_"):
            environment.pop(key)
    for distribution, (directory, module) in manifest["packages"].items():
        before = set(wheels.glob("*.whl"))
        command = [
            python,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheels),
            str(source / directory),
        ]
        with (logs / (distribution + ".log")).open("x") as log:
            subprocess.run(
                command,
                cwd=output,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        created = set(wheels.glob("*.whl")) - before
        require(
            len(created) == 1, "build must produce exactly one new wheel per package"
        )
        path = created.pop()
        inspection = inspect_wheel(
            path,
            distribution,
            directory,
            module,
            manifest["files"],
            scope=manifest["source_scope"],
        )
        manifest["wheels"].append(
            {
                "path": str(path),
                "sha256": sha(path.read_bytes()),
                "bytes": path.stat().st_size,
                "command": command,
                "inspection": inspection,
            }
        )
        write_json(output / "manifest.json", manifest)
    manifest["status"] = "wheels_built_and_inspected_CPU_only"
    write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("core", "full"),
        default="full",
        help="full current internal source (default), or earlier restricted core proposal",
    )
    parser.add_argument(
        "--build-wheels",
        action="store_true",
        help="offline setuptools builds; interpreter needs build, wheel and setuptools",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    manifest = create_stage(args.source, args.output, scope=args.scope)
    if args.build_wheels:
        build_wheels(manifest, args.output, python=args.python)
    print(
        json.dumps(
            {
                "source": manifest["source"],
                "files": len(manifest["files"]),
                "wheels": len(manifest["wheels"]),
                "publication_ready": False,
                "pipeline_validated": False,
            }
        )
    )


if __name__ == "__main__":
    main()
