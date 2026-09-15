#!/usr/bin/env python3
"""Prepare exact auxiliary assets in a new, isolated Hugging Face cache.

``plan`` is stdlib-only. Only ``prepare`` may download, and then only exact
listed files/revisions. Existing caches and native checkpoint files are never
modified. This utility neither constructs a model nor initializes CUDA.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys


CATALOG = Path(__file__).resolve().parents[1] / "release/vendor/asset_profiles.json"
FAMILIES = ("pi05", "vla4", "vla2", "groot", "va", "edge", "nano", "dreamzero")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")


def load_catalog(path: Path) -> dict:
    catalog = json.loads(path.read_text())
    if catalog.get("schema") != "instinctflash.auxiliary_asset_profiles.v1":
        raise ValueError("invalid asset profile schema")
    if set(catalog["models"]) != set(FAMILIES):
        raise ValueError("asset profiles must cover all eight variants")
    for repo, info in catalog["repositories"].items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("invalid public repository ID")
        if not re.fullmatch(r"[0-9a-f]{40}", info["revision"]):
            raise ValueError("asset revision must be a commit")
        if not info["files"]:
            raise ValueError("empty repository asset inventory")
        for name, spec in info["files"].items():
            part = PurePosixPath(name)
            if part.is_absolute() or ".." in part.parts or str(part) != name or "\\" in name:
                raise ValueError("invalid auxiliary filename")
            if type(spec["bytes"]) is not int or spec["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]):
                raise ValueError("invalid auxiliary content binding")
    for model in catalog["models"].values():
        if len(model["repositories"]) != len(set(model["repositories"])):
            raise ValueError("duplicate repository requirement")
        if set(model["repositories"]) - catalog["repositories"].keys():
            raise ValueError("unknown auxiliary repository")
        if set(model["overrides"]) - {"QWEN25_PATH", "QWEN3VL_PATH"} or set(model["overrides"].values()) - set(model["repositories"]):
            raise ValueError("invalid processor override")
        primary = model["primary_checkpoint"]
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", primary["model_id"]) or
                not re.fullmatch(r"[0-9a-f]{40}", primary["revision"])):
            raise ValueError("invalid pinned primary checkpoint")
    return catalog


def snapshot(cache: Path, repo: str, revision: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def plan(family: str, catalog: dict) -> dict:
    model = catalog["models"][family]
    repos = {name: catalog["repositories"][name] for name in model["repositories"]}
    return {"schema": "instinctflash.auxiliary_asset_plan.v1", "family": family,
            "requirements": model, "repositories": repos,
            "file_count": sum(len(v["files"]) for v in repos.values()),
            "payload_bytes": sum(s["bytes"] for v in repos.values() for s in v["files"].values()),
            "GPU_verified": False, "task_quality_certified": False}


def dreamzero_native_asset_contract(config_path: Path, catalog: dict) -> dict:
    """Check the actual checkpoint's original component targets without importing Torch.

    Both the native reference and Flash construct this same WANPolicyHead before
    applying their execution settings. Its T5/CLIP/VAE loads are unconditional;
    the existing validated full-checkpoint view skips only the base DiT reload.
    """
    config = json.loads(config_path.read_text())
    head = config["action_head_cfg"]
    prefix = "groot.vla.model.dreamzero."
    if head["_target_"] != prefix + "action_head.wan_flow_matching_action_tf.WANPolicyHead":
        raise ValueError("unqualified DreamZero action head asset contract")
    params = head["config"]
    expected_components = {
        "text_encoder_cfg": "modules.wan_video_text_encoder.WanTextEncoder",
        "image_encoder_cfg": "modules.wan_video_image_encoder.WanImageEncoder",
    }
    for name, target in expected_components.items():
        if params[name]["_target_"] != prefix + target:
            raise ValueError("unqualified DreamZero component asset contract")
    vae = params["vae_cfg"]
    defaults = {prefix + "modules.wan_video_vae.WanVideoVAE": 16,
                prefix + "modules.wan_video_vae.WanVideoVAE38": 48}
    if vae["_target_"] not in defaults:
        raise ValueError("unqualified DreamZero VAE asset contract")
    z_dim = vae.get("z_dim", defaults[vae["_target_"]])
    if type(z_dim) is not int or z_dim not in (16, 48):
        raise ValueError("unqualified DreamZero VAE latent dimension")
    wan21 = "Wan-AI/Wan2.1-I2V-14B-480P"
    vae_repo = "Wan-AI/Wan2.2-TI2V-5B" if z_dim == 48 else wan21
    targets = [(wan21, "models_t5_umt5-xxl-enc-bf16.pth"),
               (wan21, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
               (vae_repo, "Wan2.2_VAE.pth" if z_dim == 48 else "Wan2.1_VAE.pth")]
    permitted = catalog["models"]["dreamzero"]["repositories"]
    for repo, name in targets:
        if repo not in permitted or name not in catalog["repositories"][repo]["files"]:
            raise ValueError("DreamZero asset catalog omits an actual native initialization dependency")
    return {"status": "native_asset_targets_covered", "checkpoint_config_sha256": sha(config_path),
            "vae_target": vae["_target_"], "vae_z_dim": z_dim,
            "required_native_files": [{"repository": repo, "filename": name,
                                       "revision": catalog["repositories"][repo]["revision"]}
                                      for repo, name in targets],
            "scope": "Original native component construction used by native eager and Flash; no model constructed or weight loaded."}


def resolve_file(repo: str, revision: str, name: str, cache_dir: Path | None,
                 local_files_only: bool, download_cache: Path) -> Path:
    if cache_dir:
        candidate = snapshot(cache_dir, repo, revision) / name
        if candidate.is_file():
            return candidate.resolve(strict=True)
    if local_files_only:
        raise FileNotFoundError(f"exact cached asset unavailable: {repo}@{revision}/{name}")
    # Standard HF authentication is inherited; tokens are never CLI arguments or
    # receipt fields. Online downloads go to the new owned cache, never cache_dir.
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, revision=revision, filename=name,
                               cache_dir=str(download_cache), local_files_only=False)).resolve(strict=True)


def verify(path: Path, spec: dict) -> None:
    if not path.is_file() or path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]:
        raise ValueError("auxiliary asset content mismatch")


def materialize(source: Path, target: Path, spec: dict, method: str) -> str:
    verify(source, spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise ValueError("asset destination already exists")
    selected = method
    if selected == "auto":
        selected = "hardlink" if source.stat().st_dev == target.parent.stat().st_dev else "copy"
    if selected == "hardlink":
        # Never chmod or write a linked payload. Source and destination hashes
        # are checked after creation, and the sharing is explicit in receipts.
        os.link(source, target)
    else:
        if shutil.disk_usage(target.parent).free < spec["bytes"] + 64 * 1024 * 1024:
            raise OSError("insufficient free space for auxiliary copy")
        with source.open("rb") as inp, target.open("xb") as out:
            shutil.copyfileobj(inp, out, 4 * 1024 * 1024)
    verify(source, spec)
    verify(target, spec)
    return selected


def prepare(family: str, catalog: dict, root: Path, *, cache_dir: Path | None,
            local_files_only: bool, method: str, catalog_path: Path,
            include_primary: bool = False) -> dict:
    root = root.expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise ValueError("asset root must be new; existing assets are never overwritten")
    root.mkdir(parents=True, exist_ok=False)
    profile = plan(family, catalog)
    receipt = {**profile, "root": str(root), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "catalog_sha256": sha(catalog_path), "helper_sha256": sha(Path(__file__)),
               "offline": local_files_only, "files": [], "model_constructed": False,
               "existing_cache_modified": False}
    write_json(root / "plan.json", receipt)
    try:
        hub = root / "hf/hub"
        hub.mkdir(parents=True)
        for repo, info in profile["repositories"].items():
            target_snapshot = snapshot(hub, repo, info["revision"])
            for name, spec in info["files"].items():
                source = resolve_file(repo, info["revision"], name, cache_dir,
                                      local_files_only, root / "download_cache")
                target = target_snapshot / name
                selected = materialize(source, target, spec, method)
                receipt["files"].append({"repository": repo, "revision": info["revision"],
                                         "filename": name, **spec, "source": str(source),
                                         "destination": str(target), "materialization": selected})
            refs = target_snapshot.parent.parent / "refs"
            refs.mkdir()
            with (refs / "main").open("x") as out:
                out.write(info["revision"])
        if include_primary:
            primary = profile["requirements"]["primary_checkpoint"]
            repo, revision = primary["model_id"], primary["revision"]
            reference = snapshot(cache_dir, repo, revision) if cache_dir else None
            if reference is None or not reference.is_dir():
                if local_files_only:
                    raise FileNotFoundError("exact primary checkpoint snapshot unavailable")
                from huggingface_hub import snapshot_download

                reference = Path(snapshot_download(repo_id=repo, revision=revision,
                                                    cache_dir=str(root / "download_cache")))
            if reference.name != revision:
                raise ValueError("primary snapshot reference differs from requested commit")
            source_root = reference.resolve(strict=True)
            if family == "dreamzero":
                receipt["native_asset_contract"] = dreamzero_native_asset_contract(source_root / "config.json", catalog)
            targets = sorted(p for p in source_root.rglob("*") if p.is_file())
            if not targets:
                raise ValueError("primary snapshot has no files")
            primary_files = []
            for candidate in targets:
                name = candidate.relative_to(source_root).as_posix()
                source = candidate.resolve(strict=True)
                spec = {"bytes": source.stat().st_size, "sha256": sha(source)}
                target = snapshot(hub, repo, revision) / name
                selected = materialize(source, target, spec, method)
                primary_files.append({"filename": name, **spec, "source_reference": str(reference / name),
                                      "physical_source": str(source), "destination": str(target),
                                      "materialization": selected})
            refs = snapshot(hub, repo, revision).parent.parent / "refs"
            refs.mkdir()
            with (refs / "main").open("x") as out:
                out.write(revision)
            receipt["primary_checkpoint"] = {**primary, "source_snapshot_reference": str(reference),
                "physical_source_snapshot": str(source_root), "files": primary_files,
                "source_identity_scope": "Exact HF commit reference and locally measured file hashes; no independent public per-file digest claim."}
        # Isolate model cache refs without moving HF_HOME's authentication token
        # storage. The user's existing authorized Hub login remains available
        # for a later explicitly requested primary checkpoint download.
        env = {"HF_HUB_CACHE": str(hub), "HUGGINGFACE_HUB_CACHE": str(hub)}
        if family == "dreamzero":
            env["NO_ALBUMENTATIONS_UPDATE"] = "1"
        for key, repo in profile["requirements"]["overrides"].items():
            env[key] = str(snapshot(hub, repo, profile["repositories"][repo]["revision"]))
        with (root / "run.env").open("x") as out:
            # Activation must permit the user's next explicit primary download.
            # Reproduction's inference children independently force offline mode.
            out.write("unset TRANSFORMERS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE\n")
            for key, value in env.items():
                out.write(f"export {key}={shlex.quote(value)}\n")
        receipt.update(status="assets_hash_verified", environment=env,
                       run_env_sha256=sha(root / "run.env"), native_processor_loaded=False)
        write_json(root / "completion.json", receipt)
        return receipt
    except Exception as exc:
        # Do not include arbitrary remote exception text (it can carry signed
        # URLs). Retain every prepared byte and exact completed-file inventory.
        receipt.update(status="failed", error_type=type(exc).__name__, automatic_retry=False)
        write_json(root / "failure.json", receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare"))
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-primary", action="store_true",
                        help="Also copy/link and hash the exact primary HF snapshot into the new cache.")
    parser.add_argument("--primary-config", type=Path,
                        help="For DreamZero plan: check actual native component targets against the asset catalog, without imports or downloads.")
    parser.add_argument("--materialization", choices=("auto", "copy", "hardlink"), default="auto")
    args = parser.parse_args()
    try:
        catalog = load_catalog(args.catalog)
        if args.command == "plan":
            result = plan(args.family, catalog)
            if args.primary_config:
                if args.family != "dreamzero":
                    parser.error("--primary-config is currently defined only for the DreamZero native asset contract")
                result["native_asset_contract"] = dreamzero_native_asset_contract(args.primary_config, catalog)
        else:
            if args.primary_config:
                parser.error("prepare validates the resolved primary config when --include-primary is selected; --primary-config is a plan option")
            if args.root is None:
                parser.error("prepare requires --root")
            result = prepare(args.family, catalog, args.root, cache_dir=args.cache_dir,
                             local_files_only=args.local_files_only, method=args.materialization,
                             catalog_path=args.catalog, include_primary=args.include_primary)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "detail": "See the preserved root/failure.json; check exact cache files, hashes, disk space or original repository access."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
