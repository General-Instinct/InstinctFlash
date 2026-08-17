#!/usr/bin/env python3
"""The hardware vocabulary must be closed on both sides.

FOUND WHILE STUDYING FLASHRT, whose `detect_arch()` is strict by design and refuses rather than
falling back, on the stated grounds that "silently falling back to the wrong backend would hide
latency/correctness regressions". Checking whether we had the same discipline turned up a dormant
bug of exactly that shape:

    DeviceProfile.probe()  emitted  {cuda_graphs, fp8, tma, triton, wgmma}   on H100
    CuDNNConv3d            declared requires={"cudnn"}
    => satisfied_by(probe()) == (False, "device lacks ['cudnn']")

Nothing in planners/ calls probe(), so the contradiction never fired. The day it is wired, P007 --
the shipped 1.405x NUMERIC conv-layout pass -- would become silently inapplicable while the plan
still reported a legal selection. That is the failure mode the tier system exists to prevent,
sitting inside the tier system's own plumbing.

The rule these tests pin: a backend may only require a feature the probe is capable of naming. Any
other requirement is unsatisfiable on every device in existence, which is indistinguishable from a
typo and equally undetectable.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def test_vocabulary_is_closed():
    print("\n=== 1. every declared requirement is a name the probe can emit ===")
    from instinctwm.passes.contract import KNOWN_FEATURES
    import instinctwm.backends.conv.reference as ref
    import inspect

    import re
    # EVERY declaration site, not just the conv backends. Scanning only those missed
    # `graph_capture`'s `requires={"cuda"}`, which the probe never emitted -- the identical bug
    # one directory over. A closure that is only checked in one place is not a closure.
    sources = [inspect.getsource(ref)]
    for path in sorted((ROOT / "instinctwm").rglob("*.py")):
        if "requires=frozenset" in (t := path.read_text()):
            sources.append(t)
    declared = set()
    for src in sources:
        for m in re.finditer(r"requires=frozenset\(\{([^}]*)\}\)", src):
            declared |= {t.strip().strip("\"'") for t in m.group(1).split(",") if t.strip()}
    print(f"  declared across the tree : {sorted(declared) or '(none)'}")
    print(f"  nameable by the probe    : {sorted(KNOWN_FEATURES)}")
    unknown = sorted(declared - KNOWN_FEATURES)
    check(not unknown, "no backend requires a feature the probe cannot name", str(unknown))


def test_probe_reports_vendor_libraries():
    print("\n=== 2. the probe reports what is actually installed ===")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  SKIP: needs torch")
        return
    import torch
    if not torch.cuda.is_available():
        print("  SKIP: needs a CUDA device")
        return
    from instinctwm.passes.contract import DeviceProfile, KNOWN_FEATURES
    d = DeviceProfile.probe()
    print(f"  {d.name}  sm{d.capability[0]}{d.capability[1]}  features {sorted(d.features)}")
    check(d.features <= KNOWN_FEATURES, "the probe emits only known names",
          str(sorted(d.features - KNOWN_FEATURES)))
    if torch.backends.cudnn.is_available():
        check("cudnn" in d.features, "cuDNN is installed, so the probe reports it")
    check("cublas" in d.features, "cuBLAS is reported")


def test_the_shipped_conv_backend_is_satisfiable():
    print("\n=== 3. the shipped conv selection survives its own hardware gate ===")
    # THE REGRESSION. P007 must remain applicable on the hardware it was certified on.
    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIP: needs a CUDA device")
            return
    except ImportError:
        print("  SKIP: needs torch")
        return
    from instinctwm.passes.contract import DeviceProfile, HardwareReq
    d = DeviceProfile.probe()
    ok, why = HardwareReq(requires=frozenset({"cudnn"})).satisfied_by(d)
    check(ok, "a cuDNN-requiring backend is admissible on this device", why)
    ok9, why9 = HardwareReq(min_capability=(9, 0)).satisfied_by(d)
    check(ok9 == (d.capability >= (9, 0)), "min_capability compares correctly", why9)
    bad, whybad = HardwareReq(requires=frozenset({"nvfp4"})).satisfied_by(d)
    check(bad == ("nvfp4" in d.features),
          "an unavailable feature still refuses, and says why", whybad)


def test_layout_choice_can_be_measured_not_extrapolated():
    print("\n=== 4. the conv layout decision can be measured on THIS device ===")
    # `plan_for_vae` defaulted to timings from one H100, and the backends' own measure() raises
    # NotImplementedError, so the layout decision was an extrapolation on every other machine --
    # on precisely the axis (cuDNN 3D bf16 kernel coverage) that differs most between architectures.
    # Warning about that was the previous fix; measuring is the real one.
    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIP: needs a CUDA device")
            return
    except ImportError:
        print("  SKIP: needs torch")
        return
    from instinctwm.backends.conv.apply import DEFAULTS_MEASURED_ON, measure_conv_layouts
    from instinctwm.passes.contract import DeviceProfile

    d = DeviceProfile.probe()
    m = measure_conv_layouts(device=d, iters=10, warmup=3)
    check(m is not None and len(m) == 2, "both candidate layouts are timed on this device")
    if not m:
        return
    ncdhw = next(v for (n, l), v in m.items() if l.name == "NCDHW")
    ndhwc = next(v for (n, l), v in m.items() if l.name == "NDHWC")
    print(f"  NCDHW {ncdhw:.3f} ms   NDHWC {ndhwc:.3f} ms   ratio {ncdhw/ndhwc:.2f}x")
    check(ncdhw > 0 and ndhwc > 0, "both timings are real")
    if tuple(d.capability) == DEFAULTS_MEASURED_ON:
        # The independent check: a fresh instrument should reproduce the known per-signature range.
        check(2.0 < ncdhw / ndhwc < 9.0,
              "on sm_90 the measured advantage lands in the known 4.35-7.24x band",
              f"{ncdhw/ndhwc:.2f}x")
    # cached, so a second load does not re-time
    import instinctwm.backends.conv.apply as A
    check(bool(A._MEASURED_CACHE), "the result is cached per (capability, shape)")


def main() -> int:
    test_vocabulary_is_closed()
    test_probe_reports_vendor_libraries()
    test_the_shipped_conv_backend_is_satisfiable()
    test_layout_choice_can_be_measured_not_extrapolated()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the hardware vocabulary is closed and the shipped selection is admissible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
