"""Loopback-only Cosmos task-evaluation endpoint; no task verdict at startup.

The official RoboLab image/action wire format is retained. Evaluation reset,
identity, and request-sequence fields are required by CosmosQualityPolicy.
Use an SSH tunnel from the simulator host to this endpoint on Thor.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .util import ConfigurationError, write_json_atomic


class QualitySession:
    """A connection cannot continue a previous connection's random stream."""

    def __init__(self, policy):
        self.policy = policy
        self.reset_seen = False
        self.failed = False

    def infer(self, request):
        if self.failed:
            raise ConfigurationError("failed evaluation connection cannot resume")
        try:
            if not isinstance(request, dict):
                raise ConfigurationError("evaluation request must be a mapping")
            reset = request.get("reset") is True
            if not self.reset_seen and not reset:
                raise ConfigurationError("connection requires an explicit episode reset first")
            response = self.policy.infer(request)
            self.reset_seen |= reset
            return response
        except Exception:
            self.failed = True
            raise


async def serve_policy(policy, *, port, ready_receipt):
    from instinctflash.serving.msgpack_numpy import Packer, unpackb
    from websockets.asyncio.server import serve

    ready_receipt = Path(ready_receipt)
    if ready_receipt.exists():
        raise ConfigurationError("refusing to replace an existing ready receipt")
    busy = False
    connection_count = 0

    async def handle(ws):
        nonlocal busy, connection_count
        if busy:
            await ws.close(code=1013, reason="evaluation policy already has a client")
            return
        busy = True
        connection_count += 1
        session = QualitySession(policy)
        packer = Packer()
        try:
            await ws.send(packer.pack(policy.metadata))
            async for frame in ws:
                if not isinstance(frame, bytes):
                    raise ConfigurationError("evaluation transport requires binary msgpack")
                # Deliberately serial: a policy request owns its complete RNG scope.
                response = session.infer(unpackb(frame))
                await ws.send(packer.pack(response))
        except Exception as error:
            write_json_atomic(
                ready_receipt.parent / f"connection-{connection_count:06d}-failure.json",
                {"error_type": type(error).__name__, "error": str(error),
                 "task_quality_validated": False, "retry_permitted": False},
            )
            try:
                await ws.send(f"{type(error).__name__}: {error}")
                await ws.close(code=1011, reason="evaluation request failed; no automatic retry")
            except Exception:
                pass
        finally:
            busy = False

    async with serve(handle, "127.0.0.1", port, compression=None, max_size=None,
                     ping_interval=None):
        write_json_atomic(ready_receipt, {
            "status": "listening", "host": "127.0.0.1", "port": port,
            "metadata": policy.metadata, "task_quality_validated": False,
            "note": "Socket readiness is not a completed simulator episode or certificate.",
        })
        print(f"ready: {ready_receipt}", flush=True)
        await asyncio.Future()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19071)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    if args.ready_receipt.exists() or args.output_dir.exists():
        parser.error("use fresh output and ready-receipt paths")

    from .cosmos_quality_policy import CosmosQualityPolicy

    policy = CosmosQualityPolicy.from_matrix(
        args.matrix, args.cell_id, matrix_sha256=args.matrix_sha256,
        output_dir=args.output_dir,
    )
    try:
        asyncio.run(serve_policy(policy, port=args.port, ready_receipt=args.ready_receipt))
    finally:
        policy.close()


if __name__ == "__main__":
    main()
