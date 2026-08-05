#!/usr/bin/env python3
"""Regression tests for the Triton gated residual -- specifically for FMA contraction.

WHY THIS FILE EXISTS

The first version of this kernel was 1.22x and looked fine under a loose tolerance. It was not
bit-exact: Triton 3.5.0 contracted `h + a*g` into a single `fma.rn.f32`, skipping the fp32 rounding
of the product that eager PyTorch performs. Only 33 of 1,474,560 elements differed, by exactly one
bf16 ULP -- small enough that any `allclose` would have waved it through.

Worse, the first attempted fix (`tl.where(mask, p, p)` as an optimization barrier) was removed by
the compiler. Both paths still contracted, both showed the identical delta, and that read as "FMA
is ruled out" when in truth FMA had never been disabled. The lesson is `test_ptx`: assert on the
emitted instruction, not on a differential test against your own flag.

    python tests/test_triton_residual.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from instinctwm.kernels.triton_residual import (
    _gated_residual_kernel, gated_residual, gated_residual_eager,
)

SHAPES = [(2, 240, 3072), (2, 32, 3072)]                   # video and action streams


def _inputs(shape, seed=0, hscale=1.0, ascale=1.0):
    B, N, C = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (
        (torch.randn(B, N, C, generator=g) * hscale).to("cuda", torch.bfloat16),
        (torch.randn(B, N, C, generator=g) * ascale).to("cuda", torch.bfloat16),
        torch.randn(B, N, C, generator=g).to("cuda", torch.float32),   # gate is fp32 in the model
    )


#: ABBA repeats, and the within-arm spread above which no speed verdict is offered.
REPEATS = 3
MAX_SPREAD = 0.15
#: Utilisation on ANY device above which the box counts as occupied. Deliberately low: a 48% neighbour
#: was enough to turn a 1.20x measurement into a reported regression.
BUSY_UTIL = 15


def _spread(xs):
    """Relative spread of a sample: (max-min)/mean. Cheap, and sensitive to the single slow outlier
    that contention actually produces, which a standard deviation would dilute."""
    m = sum(xs) / len(xs)
    return (max(xs) - min(xs)) / m if m > 0 else float("inf")


def _device_busy():
    """Is anything else using the GPUs? Checked by utilisation AND by foreign compute processes.

    Utilisation alone misses a neighbour that is between kernels; the process list alone misses a
    busy device whose owner is not visible to us. Either signal is enough to withhold a verdict.
    """
    import os
    import subprocess
    try:
        u = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15).stdout
        hot = [ln for ln in u.strip().split("\n")
               if ln.strip() and int(ln.split(",")[1]) >= BUSY_UTIL]
        if hot:
            return True, f"GPU utilisation {'; '.join(x.strip() for x in hot)}%"
        pids = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                               "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15).stdout.split()
        foreign = [x for x in pids if x.strip() and x.strip() != str(os.getpid())]
        if foreign:
            return True, f"{len(foreign)} other compute process(es) on the GPUs"
    except Exception as e:
        return True, f"could not determine device state ({type(e).__name__}); withholding a verdict"
    return False, ""


def _bench(f, it=200):
    for _ in range(50):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000.0                 # us


def test_bit_exact():
    print("=== 1. bit-exact against eager on both stream shapes ===")
    ok = True
    for shape in SHAPES:
        h, a, gt = _inputs(shape)
        out, ref = gated_residual(h, a, gt), gated_residual_eager(h, a, gt)
        n = (out != ref).sum().item()
        d = (out.float() - ref.float()).abs().max().item()
        print(f"  {'OK  ' if n == 0 else 'FAIL'} {str(shape):>16}  differing={n}  max|d|={d:.3e}")
        ok &= n == 0
    return ok


def test_scales():
    """Sweep the exponent range so the product straddles bf16 tie points.

    Scales are inverse to each other, keeping `a*g` near `h` in magnitude -- where a retained FMA
    guard bit is most likely to flip the final round-to-nearest.
    """
    print("=== 2. bit-exact across an adversarial scale sweep ===")
    total = elems = 0
    for trial in range(12):
        h, a, gt = _inputs((2, 240, 3072), seed=1000 + trial,
                           hscale=2.0 ** (trial - 6), ascale=2.0 ** (6 - trial))
        total += (gated_residual(h, a, gt) != gated_residual_eager(h, a, gt)).sum().item()
        elems += h.numel()
    print(f"  {'OK  ' if total == 0 else 'FAIL'} {total} differing over {elems:,} elements "
          f"across 12 scales (2^-6 .. 2^5)")
    return total == 0


def test_fma_negative_control():
    """If FMA mode ever matches eager, `test_bit_exact` is passing for free and proves nothing."""
    print("=== 3. negative control: FMA mode MUST differ ===")
    h, a, gt = _inputs((2, 240, 3072), seed=1006)          # scale with the largest disagreement
    n = (gated_residual(h, a, gt, allow_fma=True) != gated_residual_eager(h, a, gt)).sum().item()
    print(f"  {'OK  ' if n > 0 else 'FAIL'} FMA mode differs on {n} elements "
          f"({'enable_fp_fusion still controls contraction' if n else 'FLAG HAS NO EFFECT'})")
    return n > 0


def test_ptx():
    """Assert on the emitted instruction. This is the test the original bug would have failed."""
    print("=== 4. PTX: no fma.rn.f32 when contraction is disabled ===")
    h, a, gt = _inputs((2, 32, 3072))
    ok = True
    for allow_fma, want, forbid in ((False, "mul.rn.f32", "fma.rn.f32"),
                                    (True, "fma.rn.f32", None)):
        ptx = _gated_residual_kernel.warmup(
            h, a, gt, torch.empty_like(h), h.numel(), BLOCK=1024, num_warps=4,
            enable_fp_fusion=allow_fma, grid=(1,)).asm["ptx"]
        good = want in ptx and (forbid is None or forbid not in ptx)
        print(f"  {'OK  ' if good else 'FAIL'} allow_fma={str(allow_fma):5s} "
              f"has {want}={want in ptx}" + (f", has {forbid}={forbid in ptx}" if forbid else ""))
        ok &= good
    return ok


def test_faster_than_eager():
    """Performance gate. Bit-exactness is necessary, not sufficient."""
    print("=== 5. faster than eager (performance gate) ===")
    #
    # THIS GATE USED TO CRY WOLF. A bare `eager > exact` comparison on a shared box produced three
    # false failures in one session: the same binary measured 0.98x, 2.94x and 1.07x on consecutive
    # runs, once with eager itself at 459 us instead of 27. Nothing about the kernel changed; an
    # 8-GPU eval fleet was running. A perf gate that fails on contention trains people to ignore it,
    # which costs more than the gate is worth.
    #
    # So: correctness above is UNCONDITIONAL and already decided. This block reports a SEPARATE
    # verdict and is allowed to answer NOT EVALUATED.
    busy, why = _device_busy()
    if busy:
        print(f"  NOT EVALUATED -- the device is materially occupied ({why}).")
        print("  Correctness above is unaffected; run this on an idle GPU for a speed verdict.")
        return True

    ok = True
    for shape in SHAPES:
        h, a, gt = _inputs(shape)
        # ABBA per repeat: eager, exact, exact, eager. Averaging the two eager samples around the
        # two exact ones cancels a MONOTONIC drift, which is the shape this box actually has --
        # measured elsewhere as 3214 -> 3730 -> 3964 ms across three rounds of one configuration.
        # Base-then-treatment ordering charges that drift entirely to the treatment.
        es, xs = [], []
        for _ in range(REPEATS):
            e1 = _bench(lambda: gated_residual_eager(h, a, gt))
            x1 = _bench(lambda: gated_residual(h, a, gt))
            x2 = _bench(lambda: gated_residual(h, a, gt))
            e2 = _bench(lambda: gated_residual_eager(h, a, gt))
            es += [e1, e2]
            xs += [x1, x2]
        te, tx = sum(es) / len(es), sum(xs) / len(xs)
        spread = max(_spread(es), _spread(xs))
        tf = _bench(lambda: gated_residual(h, a, gt, allow_fma=True))

        if spread > MAX_SPREAD:
            # A noisy sample cannot support either verdict. Saying so beats reporting a ratio drawn
            # from a distribution this wide.
            print(f"  NOT EVAL {str(shape):>16}  spread {spread*100:.1f}% > "
                  f"{MAX_SPREAD*100:.0f}%: too noisy to judge (eager {te:.2f} exact {tx:.2f} us)")
            continue
        verdict = "OK  " if te > tx else "FAIL"
        ok &= te > tx
        print(f"  {verdict} {str(shape):>16}  eager {te:6.2f} us  exact {tx:6.2f} us  "
              f"-> {te/tx:.2f}x  (spread {spread*100:.1f}%, ABBA x{REPEATS})   "
              f"(FMA mode {tf:6.2f} us: disabling contraction costs {(tx/tf-1)*100:+.1f}%)")
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        sys.exit(0)
    results = [t() for t in (test_bit_exact, test_scales, test_fma_negative_control,
                             test_ptx, test_faster_than_eager)]
    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} groups")
    sys.exit(0 if all(results) else 1)
