#!/usr/bin/env python3
"""Install pinned LIBERO, prepare isolated configuration, and validate real EGL rendering.

Run using the FlashRT CUDA environment's Python. The assets and config directories
are explicit; an existing conflicting config or assets link is never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
PINS = {"hf-libero": "0.1.4", "robosuite": "1.4.0", "mujoco": "3.8.1", "bddl": "1.0.1",
        "gymnasium": "1.3.0", "numpy": "2.2.6"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--install", action="store_true", help="Install all simulator pins and their dependencies")
    parser.add_argument("--download", action="store_true", help="Download the immutable assets revision with hf")
    args = parser.parse_args()
    if args.install:
        subprocess.run([sys.executable, "-m", "pip", "install", *[f"{k}=={v}" for k, v in PINS.items()]], check=True)
    versions = {k: importlib.metadata.version(k) for k in PINS}
    if versions != PINS:
        raise RuntimeError(f"simulator versions differ; rerun with --install: {versions}")
    if args.download:
        hf = shutil.which("hf") or str(Path(sys.executable).parent / "hf")
        subprocess.run([hf, "download", "lerobot/libero-assets", "--repo-type", "dataset",
                        "--revision", ASSETS_REVISION, "--local-dir", str(args.assets)], check=True)
    assets = args.assets.resolve()
    if not all((assets / part).is_dir() for part in
               ("scenes", "articulated_objects", "stable_hope_objects", "textures")):
        raise RuntimeError("missing LIBERO assets; rerun with --download")
    spec = importlib.util.find_spec("libero")
    package = Path(next(iter(spec.submodule_search_locations))) / "libero"
    link = package / "assets"
    if link.exists() or link.is_symlink():
        if link.resolve() != assets:
            raise RuntimeError(f"conflicting assets link: {link}")
    else:
        link.symlink_to(assets, target_is_directory=True)
    import yaml
    config = {"assets": str(assets), "benchmark_root": str(package),
              "bddl_files": str(package / "bddl_files"), "init_states": str(package / "init_files"),
              "datasets": str(args.config_dir.resolve() / "datasets")}
    args.config_dir.mkdir(parents=True, exist_ok=True)
    target = args.config_dir / "config.yaml"
    if target.exists() and yaml.safe_load(target.read_text()) != config:
        raise RuntimeError(f"conflicting configuration: {target}; choose a fresh --config-dir")
    target.write_text(yaml.safe_dump(config))
    env = {**os.environ, "LIBERO_CONFIG_PATH": str(args.config_dir.resolve()),
           "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "MUJOCO_EGL_DEVICE_ID": "0"}
    subprocess.run([sys.executable, "-c", "from benchmarks.vla.pi05_sm120_libero import make_vec; "
        "v=make_vec(0); o,_=v.reset(seed=[40100]); "
        "assert all(x.mean()>3 and x.std()>1 for x in o['pixels'].values()), 'blank EGL'; "
        "v.step(__import__('numpy').zeros((1,7),dtype='float32')); v.close(); print('LIBERO EGL smoke PASS')"],
        env=env, check=True)
    print(f"LIBERO_CONFIG_PATH={args.config_dir.resolve()}")


if __name__ == "__main__":
    main()
