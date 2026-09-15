"""Build target-native wheels and install isolated overlays over existing vendor stacks.

Only new directories below this study root are written. Vendor environments and
checkpoints are read-only dependencies; no CUDA model is imported or executed.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
HOME_DIR = Path("/home/guanming")
STACKS = {
    "pi05": "frt_env", "vla4": "venv_vla4b", "vla2": "venv_vla2",
    "groot": "thorcol/groot_env", "edge": "thorcol/cosmos-framework/.venv",
    "nano": "thorcol/cosmos-framework/.venv", "va": "venv_va",
    "dreamzero": "thorcol/dz_env",
}
KERNEL_ROOT = HOME_DIR / "ifl_eval/framework_comparison_20260913_thor_kernel_recovery/source/serving/flash_rt"
KERNELS = {
    "flash_rt_kernels.cpython-312-aarch64-linux-gnu.so": "26a092f35398cfd3dfee57d56871c4bea228613b15dd0d6c92d90abaed212c18",
    "libfmha_fp16_strided.so": "0d5cfcf58fa631480c5a603bb23c933cb10e09f1523deeabfc4f193d4e45a475",
    "flash_rt_fp4.cpython-312-aarch64-linux-gnu.so": "6f7c208d94ed17c48087449375c6bf30ef3a7288b5756445478bdf79b62b9b2b",
    "flash_rt_fa2.cpython-312-aarch64-linux-gnu.so": "4c3e1c9f3f43f7f36a7672ae3cfd13c02741e7e21349622d68591c1fe503d5dc",
}
PACKAGES = (".", "examples/pi05_vla", "examples/lingbot_vla", "examples/lingbot_vla_v2",
            "examples/groot_n17", "examples/cosmos3_policy", "examples/dreamzero", "serving")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    with log.open("x") as stream:
        subprocess.run([str(x) for x in command], cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    receipt = ROOT / "installation.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    wheels = ROOT / "wheels"
    wheels.mkdir()
    logs = ROOT / "install_logs"
    logs.mkdir()
    (ROOT / "native").mkdir()
    for name, expected in KERNELS.items():
        original = KERNEL_ROOT / name
        assert sha(original) == expected, name
        shutil.copy2(original, SOURCE / "serving/flash_rt" / name)
    bf16 = HOME_DIR / "ifl_eval/four_frameworks_20260913/libinstinctflash_bf16.so"
    shutil.copy2(bf16, ROOT / "native/libinstinctflash_bf16.so")
    python = HOME_DIR / "frt_env/bin/python"
    for i, package in enumerate(PACKAGES):
        run([python, "-m", "pip", "wheel", SOURCE / package, "--no-deps", "--no-build-isolation",
             "--wheel-dir", wheels], logs / f"build-{i}.log")
    built = sorted(wheels.glob("*.whl"))
    assert len(built) == 8
    flash = next(p for p in built if p.name.startswith("flash_rt-"))
    assert "cp312-cp312-linux_aarch64" in flash.name, flash.name
    environments = {}
    for family, stack in STACKS.items():
        original = HOME_DIR / stack
        target = ROOT / "envs" / family
        run([original / "bin/python", "-m", "venv", "--without-pip", target], logs / f"venv-{family}.log")
        site = target / "lib/python3.12/site-packages"
        assert site.is_dir()
        # venv --system-site-packages does not inherit a parent venv. Add the
        # existing vendor dependency environment explicitly, after this overlay.
        (site / "vendor_stack.pth").write_text("import site; site.addsitedir(" + repr(str(original / "lib/python3.12/site-packages")) + ")\n")
        run([python, "-m", "pip", "--python", target / "bin/python", "install", "--no-deps",
             "--ignore-installed", *built], logs / f"install-{family}.log")
        audit = ROOT / "install_checks" / f"{family}.json"
        run([target / "bin/python", "-I", SOURCE / "scripts/check_installed_package.py",
             "--require-all-adapters", "--output", audit], logs / f"check-{family}.log")
        assert json.loads(audit.read_text())["ok"]
        environments[family] = dict(python=str(target / "bin/python"), vendor_stack=str(original),
            overlay_site=str(site), audit=str(audit.relative_to(ROOT)), audit_sha256=sha(audit))
        print(json.dumps(dict(family=family, installed=True)), flush=True)
    report = dict(schema=1, ok=True, environments=environments,
        wheel_sha256={p.name: sha(p) for p in built},
        native_sha256={**KERNELS, "libinstinctflash_bf16.so": sha(bf16)},
        scope="Fresh installed wheels over read-only existing vendor stacks; no new driver/CUDA/vendor installation")
    receipt.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
