"""Benchmark endpoints stop on SIGTERM through their own cleanup path."""
import asyncio
import os
import signal
import threading

from benchmarks.vla.util import wait_for_stop_signal


def test_sigterm_returns_and_restores_the_default_handler():
    cleaned = []

    async def endpoint():
        try:
            await wait_for_stop_signal()
        finally:
            cleaned.append(True)

    previous = signal.getsignal(signal.SIGTERM)
    timer = threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGTERM))
    timer.start()
    asyncio.run(asyncio.wait_for(endpoint(), timeout=10))
    timer.join()
    assert cleaned == [True]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_every_benchmark_endpoint_waits_for_sigterm():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "vla"
    for name in ("joint_policy_server.py", "groot_policy_server.py", "wan_va_runtime_server.py",
                 "wan_va_benchmark_server.py", "cosmos_quality_server.py"):
        source = (root / name).read_text()
        assert "await wait_for_stop_signal()" in source, name
        assert "await asyncio.Future()" not in source, name
