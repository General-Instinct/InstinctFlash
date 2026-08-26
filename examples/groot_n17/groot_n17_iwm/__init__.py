"""GR00T N1.7 optimization package: the bitexact fastpaths.

`fast_decode` and `backbone_fastpath` are pure-torch/numpy rewrites gated bitexact against
stock under the 6-case H100 protocol (see ../verify_fastpaths.py and its committed results).
Nothing imports torch at package import time; each module is imported by the caller that
installs it.
"""
