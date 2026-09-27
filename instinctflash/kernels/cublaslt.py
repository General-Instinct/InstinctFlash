"""Direct cuBLASLt FP8 GEMM with measured algorithm selection.

``torch._scaled_mm`` exposes neither the weight layout nor the algorithm
choice, and cuBLASLt's first heuristic is often not the fastest kernel for the
skinny (M ~ 50) decode shapes of action experts. This module drives cuBLASLt
directly through ctypes (the library PyTorch already loaded; nothing is
compiled), enumerates the heuristic candidates for a shape, times each on the
actual device, and keeps the fastest.

Computes row-major ``D[M, N] = A[M, K] @ W^T`` for E4M3 inputs with FP32
accumulation, unit scales and a 16-bit output. ``W`` is either ``[N, K]``
("tn") or ``[K, N]`` ("nn") row-major.
"""
from __future__ import annotations

import ctypes

import torch

_TRANSA, _TRANSB = 3, 4
_PREF_MAX_WORKSPACE = 1
_OP_N, _OP_T = 0, 1
_COMPUTE_32F = 68
_R_32F, _R_16F, _R_16BF, _R_8F_E4M3 = 0, 2, 14, 28
_DTYPE = {torch.float16: _R_16F, torch.bfloat16: _R_16BF, torch.float8_e4m3fn: _R_8F_E4M3,
          torch.float32: _R_32F}
WORKSPACE_BYTES = 32 << 20


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heuristic(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


def _load_library():
    # Prefer the exact cuBLASLt PyTorch mapped, so both share one version.
    torch.zeros(1, device="cuda") @ torch.zeros(1, 1, device="cuda").view(1, 1).expand(1, 1)
    with open("/proc/self/maps") as f:
        for line in f:
            if "libcublasLt.so" in line:
                return ctypes.CDLL(line.split()[-1])
    return ctypes.CDLL("libcublasLt.so")


class _Lt:
    _instance = None

    def __init__(self):
        lib = self.lib = _load_library()
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        lib.cublasLtCreate.argtypes = [ctypes.POINTER(vp)]
        lib.cublasLtMatmulDescCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_int]
        lib.cublasLtMatmulDescSetAttribute.argtypes = [vp, ctypes.c_int, vp, sz]
        lib.cublasLtMatrixLayoutCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_uint64,
                                                   ctypes.c_uint64, ctypes.c_int64]
        lib.cublasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(vp)]
        lib.cublasLtMatmulPreferenceSetAttribute.argtypes = [vp, ctypes.c_int, vp, sz]
        lib.cublasLtMatmulAlgoGetHeuristic.argtypes = [vp, vp, vp, vp, vp, vp, vp, ctypes.c_int,
                                                       ctypes.POINTER(_Heuristic), ctypes.POINTER(ctypes.c_int)]
        lib.cublasLtMatmul.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp,
                                       ctypes.POINTER(_Algo), vp, sz, vp]
        self.handle = vp()
        self._check(lib.cublasLtCreate(ctypes.byref(self.handle)), "cublasLtCreate")
        self.workspaces = {}
        self.alpha = ctypes.c_float(1.0)
        self.beta = ctypes.c_float(0.0)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _check(status, what):
        if status != 0:
            raise RuntimeError(f"{what} failed with cuBLAS status {status}")

    def workspace(self, device):
        ws = self.workspaces.get(device)
        if ws is None:
            ws = self.workspaces[device] = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        return ws

    def layout(self, dtype, rows, cols, ld):
        out = ctypes.c_void_p()
        self._check(self.lib.cublasLtMatrixLayoutCreate(ctypes.byref(out), dtype, rows, cols, ld), "layout")
        return out


class Fp8Gemm:
    """A prepared FP8 GEMM for one (M, N, K, layout, output dtype)."""

    _cache: dict = {}

    def __init__(self, M, N, K, *, layout="tn", out_dtype=torch.bfloat16, device="cuda"):
        lt = self.lt = _Lt.get()
        self.M, self.N, self.K, self.layout_name, self.out_dtype = M, N, K, layout, out_dtype
        self.device = torch.device(device)
        self.desc = ctypes.c_void_p()
        lt._check(lt.lib.cublasLtMatmulDescCreate(ctypes.byref(self.desc), _COMPUTE_32F, _R_32F), "desc")
        ta = ctypes.c_int(_OP_T if layout == "tn" else _OP_N)
        tb = ctypes.c_int(_OP_N)
        lt._check(lt.lib.cublasLtMatmulDescSetAttribute(self.desc, _TRANSA, ctypes.byref(ta), 4), "transa")
        lt._check(lt.lib.cublasLtMatmulDescSetAttribute(self.desc, _TRANSB, ctypes.byref(tb), 4), "transb")
        # Column-major view: D^T[N, M] = op(W) @ A^T
        if layout == "tn":
            self.a_desc = lt.layout(_R_8F_E4M3, K, N, K)
        else:
            self.a_desc = lt.layout(_R_8F_E4M3, N, K, N)
        self.b_desc = lt.layout(_R_8F_E4M3, K, M, K)
        self.d_desc = lt.layout(_DTYPE[out_dtype], N, M, N)
        pref = ctypes.c_void_p()
        lt._check(lt.lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "pref")
        ws = ctypes.c_size_t(WORKSPACE_BYTES)
        lt._check(lt.lib.cublasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WORKSPACE, ctypes.byref(ws), 8), "pref")
        results = (_Heuristic * 32)()
        count = ctypes.c_int(0)
        lt._check(lt.lib.cublasLtMatmulAlgoGetHeuristic(lt.handle, self.desc, self.a_desc, self.b_desc, self.d_desc,
                                                         self.d_desc, pref, 32, results, ctypes.byref(count)),
                  "heuristic")
        self.candidates = [results[i] for i in range(count.value) if results[i].state == 0]
        if not self.candidates:
            raise RuntimeError(f"cuBLASLt has no FP8 algorithm for {M}x{N}x{K} ({layout})")
        self.choice = self.candidates[0]

    def __call__(self, a, w, out, stream=None):
        lt = self.lt
        ws = lt.workspace(self.device)
        s = stream if stream is not None else torch.cuda.current_stream(self.device).cuda_stream
        status = lt.lib.cublasLtMatmul(
            lt.handle, self.desc, ctypes.byref(lt.alpha), ctypes.c_void_p(w.data_ptr()), self.a_desc,
            ctypes.c_void_p(a.data_ptr()), self.b_desc, ctypes.byref(lt.beta), ctypes.c_void_p(out.data_ptr()),
            self.d_desc, ctypes.c_void_p(out.data_ptr()), self.d_desc, ctypes.byref(self.choice.algo),
            ctypes.c_void_p(ws.data_ptr()), WORKSPACE_BYTES, ctypes.c_void_p(s))
        if status != 0:
            raise RuntimeError(f"cublasLtMatmul failed with status {status}")
        return out

    def tune(self, a, weights, out, *, reps=10, rounds=3):
        """Time every candidate inside a CUDA graph over ``weights`` (a list rotated to defeat
        L2) and keep the fastest. Returns its per-call time in milliseconds."""
        best = None
        stream = torch.cuda.Stream(self.device)
        for cand in self.candidates:
            self.choice = cand
            try:
                self(a, weights[0], out)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(reps):
                        for w in weights:
                            self(a, w, out)
                graph.replay()
                torch.cuda.synchronize()
            except RuntimeError:
                continue
            times = []
            for _ in range(rounds):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                graph.replay()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e) / (reps * len(weights)))
            del graph
            t = min(times)
            if best is None or t < best[0]:
                best = (t, cand)
        if best is None:
            raise RuntimeError(f"no cuBLASLt candidate ran for {self.M}x{self.N}x{self.K}")
        self.choice = best[1]
        return best[0]
