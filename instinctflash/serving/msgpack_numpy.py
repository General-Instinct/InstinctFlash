"""msgpack with NumPy arrays — the openpi wire encoding, vendored byte-for-byte.

This is the serialization `openpi-client` (pypi, 0.1.1) puts on the wire, adapted upstream from
https://github.com/lebedov/msgpack-numpy with the pickle fallback removed (pickle on a network
socket is arbitrary code execution). Vendored from openpi (Apache-2.0,
src `packages/openpi-client/src/openpi_client/msgpack_numpy.py`) rather than depended on, for two
reasons that are both about the server side:

  * the `openpi-client` wheel pins `numpy<2.0.0`, which a serving box has no reason to inherit;
  * a server importing a client package to get 50 lines of packing is a dependency pointing the
    wrong way.

BYTE-COMPAT IS THE CONTRACT. An ndarray packs as a map with **byte** keys
(`b"__ndarray__"`, `b"data"`, `b"dtype"`, `b"shape"`), a NumPy scalar as `b"__npgeneric__"`, and
dtype kinds "V"/"O"/"c" are refused at pack time. Any deviation here silently breaks every robot
client in the pi0/openpi ecosystem, so `tests/test_ws_server.py` compares our bytes against the
installed `openpi_client` wheel's own packer rather than trusting this comment.
"""

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
