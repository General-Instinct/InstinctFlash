"""Optional prebuilt native kernels.

The core package stays CUDA-free.  GPU-specific shared libraries are built explicitly and loaded
only when a hardware-gated pass selects them.
"""
