"""Install the prompt-reuse candidate in a new overlay; existing runs stay frozen."""
import importlib.util
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prepare", ROOT / "prepare_thor.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
sha, run = prepare.sha, prepare.run


def main():
    candidate = ROOT / "candidate_b"
    source = candidate / "source"
    manifest = json.loads((candidate / "manifest.json").read_text())
    for path, entry in manifest["files"].items():
        assert sha(source / path) == entry["sha256"], path
    target = ROOT / "envs/vla4-update"
    assert not target.exists(), target
    logs, wheels = candidate / "logs", candidate / "wheels"
    logs.mkdir()
    wheels.mkdir()
    for name, expected in prepare.KERNELS.items():
        original = prepare.KERNEL_ROOT / name
        assert sha(original) == expected
        shutil.copy2(original, source / "serving/flash_rt" / name)
    python = prepare.HOME_DIR / "frt_env/bin/python"
    for i, package in enumerate((".", "serving")):
        run([python, "-m", "pip", "wheel", source / package, "--no-deps",
             "--no-build-isolation", "--wheel-dir", wheels], logs / f"build-{i}.log")
    built = sorted(wheels.glob("*.whl"))
    assert len(built) == 2
    assert any("cp312-cp312-linux_aarch64" in p.name for p in built)
    original_install = json.loads((ROOT / "installation.json").read_text())
    vendor = Path(original_install["environments"]["vla4"]["vendor_stack"])
    run([vendor / "bin/python", "-m", "venv", "--without-pip", target], logs / "venv.log")
    site = target / "lib/python3.12/site-packages"
    (site / "vendor_stack.pth").write_text(
        "import site; site.addsitedir(" + repr(str(vendor / "lib/python3.12/site-packages")) + ")\n")
    plugins = sorted(p for p in (ROOT / "wheels").glob("*.whl")
                     if not p.name.startswith(("instinctflash-", "flash_rt-")))
    assert len(plugins) == 6
    run([python, "-m", "pip", "--python", target / "bin/python", "install", "--no-deps",
         "--ignore-installed", *built, *plugins], logs / "install.log")
    audit = ROOT / "install_checks/vla4-update.json"
    run([target / "bin/python", "-I", source / "scripts/check_installed_package.py",
         "--require-all-adapters", "--output", audit], logs / "check.log")
    assert json.loads(audit.read_text())["ok"]
    original_install["environments"]["vla4"] = dict(python=str(target / "bin/python"),
        vendor_stack=str(vendor), overlay_site=str(site), audit=str(audit.relative_to(ROOT)),
        audit_sha256=sha(audit), installed_wheels=[p.name for p in (*built, *plugins)])
    original_install.update(parent_installation_sha256=sha(ROOT / "installation.json"),
        candidate_manifest_sha256=sha(candidate / "manifest.json"),
        wheel_sha256={p.name: sha(p) for p in (*built, *plugins)},
        scope="Separate VLA4 prompt-reuse overlay; original eight overlays unchanged")
    (ROOT / "installation_update.json").write_text(json.dumps(original_install, indent=2)+"\n")
    inventory = json.loads((ROOT / "installed_files_v2.json").read_text())
    for path in site.rglob("*"):
        if path.is_file() and path.suffix != ".pyc":
            inventory[str(path)] = sha(path)
    for path in (ROOT / "installation_update.json", candidate / "manifest.json", audit):
        inventory[str(path)] = sha(path)
    (ROOT / "installed_files_update.json").write_text(json.dumps(inventory, indent=2)+"\n")
    print(json.dumps(dict(ok=True, python=str(target / "bin/python"), bound_files=len(inventory))))


if __name__ == "__main__":
    main()
