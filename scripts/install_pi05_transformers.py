#!/usr/bin/env python3
"""Install the exact upstream OpenPI transformers dependency required by LeRobot 0.4.4.

Run in the dedicated environment after installing requirements-sm120.lock.
Every upstream file is verified before any installation write occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
FILES = {
    "models/gemma/configuration_gemma.py": "26fd0d0f52730ab32b6f337eb73ae9a26cb71ae0915bc49eced1d0e2ed8d0ec6",
    "models/gemma/modeling_gemma.py": "17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde",
    "models/paligemma/modeling_paligemma.py": "c945d330b829264f927f3ee5339f977238631772de07a28e855e265bb85fd53c",
    "models/siglip/modeling_siglip.py": "ef2e99500f263fdd78db9d4d9a255e29d844eb92f53c25ad640f8056700ef84b",
    "models/siglip/check.py": "466e8eb7887ac5ccc98118a3169298d6ab65a284d834bae74751e3854dbc4245",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Offline transformers_replace directory")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    import transformers
    if transformers.__version__ != "4.53.2":
        raise RuntimeError("install the locked transformers==4.53.2 first")
    target = Path(transformers.__file__).resolve().parent
    if not target.is_relative_to(Path(sys.prefix).resolve()) or sys.prefix == sys.base_prefix:
        raise RuntimeError("run this installer in a dedicated virtual environment")
    contents = {}
    for name, digest in FILES.items():
        if args.verify_only:
            data = (target / name).read_bytes()
        elif args.source_root:
            data = (args.source_root / name).read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/Physical-Intelligence/openpi/{REVISION}/src/openpi/models_pytorch/transformers_replace/{name}"
            with urllib.request.urlopen(url, timeout=60) as stream:
                data = stream.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"official dependency hash mismatch: {name}")
        contents[name] = data
    if not args.verify_only:
        for name, data in contents.items():
            destination = target / name
            # Replace the inode: uv installations may share hardlinks with a
            # wheel cache or another environment, which must remain untouched.
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                stream.write(data)
                temporary = Path(stream.name)
            try:
                temporary.chmod(0o644)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        manifest = {"openpi_revision": REVISION, "transformers": "4.53.2", "files": FILES}
        (Path(sys.prefix) / "pi05-transformers-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified all {len(FILES)} official OpenPI dependency files at {REVISION}")


if __name__ == "__main__":
    main()
