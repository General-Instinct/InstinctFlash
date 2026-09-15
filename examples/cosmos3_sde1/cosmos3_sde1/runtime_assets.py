"""Validate the actual external WanVAE load in an isolated qualification worker."""
from contextlib import contextmanager
import functools
import hashlib
from pathlib import Path

WAN_BYTES = 2818839170
WAN_SHA256 = '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'


def identify(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return dict(path=str(path), resolved_path=str(path.resolve()),
        bytes=path.stat().st_size, sha256=digest.hexdigest())


@contextmanager
def audit_vae_load(io=None):
    """Hash before deserialization, verify after load, restore loader binding."""
    if io is None:
        from cosmos_framework.utils.easy_io import easy_io
        io = easy_io
    original = io.load
    owned = 'load' in vars(io)
    records = []

    @functools.wraps(original)
    def load(file, *args, **kwargs):
        path = file
        if not isinstance(path, (str, Path)) or Path(path).name != 'Wan2.2_VAE.pth':
            return original(path, *args, **kwargs)
        before = identify(path)
        if before['bytes'] != WAN_BYTES or before['sha256'] != WAN_SHA256:
            raise ValueError('Actual external WanVAE differs from the declared runtime anchor')
        result = original(path, *args, **kwargs)
        if identify(path) != before:
            raise ValueError('External WanVAE changed while loading')
        records.append(before)
        return result

    io.load = load
    try:
        yield records
        if len(records) != 1:
            raise RuntimeError('Expected exactly one observed native external WanVAE load')
    finally:
        intact = io.load is load
        if owned:
            io.load = original
        else:
            del io.load
        if not intact:
            raise RuntimeError('Runtime asset observer was replaced during model construction')
