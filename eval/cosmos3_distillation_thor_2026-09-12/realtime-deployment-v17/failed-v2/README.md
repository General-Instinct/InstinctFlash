V2 fixed the 1000 clock but incorrectly set prefix NFE to2. The real Runtime
requires one prefix phase regardless of denoiser steps. Thor rejected the package
before native model loading/inference. Source weights and failed receipts are
preserved; -v2 packages are excluded. V3 fixes prefix=1/action=2 and additionally
parses the actual package through Runtime and its adapter before returning success.
