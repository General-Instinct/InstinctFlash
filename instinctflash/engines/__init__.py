"""InstinctFlash's own accelerated inference engines.

Each engine prepares a checkpoint's weights once, executes the model with the
kernels in :mod:`instinctflash.kernels`, and captures its whole request path
in CUDA graphs keyed only by buffer geometry. Engines never calibrate on live
observations: activation scales are computed per token as the data flows.
"""
