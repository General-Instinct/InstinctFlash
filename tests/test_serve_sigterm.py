"""`instinctflash serve` stops cleanly on SIGTERM even when SIGINT was inherited as ignored."""
import os
import signal
import threading

from instinctflash.cli import _serve_until_stopped


class FakeServer:
    def __init__(self):
        self.stopped = threading.Event()
        self.shutdowns = 0

    def serve_forever(self):
        # Wake periodically so the main thread can run Python signal handlers, as the real
        # asyncio loop does when select() is interrupted.
        while not self.stopped.wait(0.05):
            pass

    def shutdown(self):
        self.shutdowns += 1
        self.stopped.set()


def test_sigterm_shuts_the_server_down_and_restores_the_handler():
    before = signal.getsignal(signal.SIGTERM)
    previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)   # the inherited background state
    try:
        server = FakeServer()
        timer = threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGTERM))
        timer.start()
        _serve_until_stopped(server)
        timer.join()
        assert server.shutdowns == 1
        assert signal.getsignal(signal.SIGTERM) is before
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN, "SIGINT disposition is left alone"
    finally:
        signal.signal(signal.SIGINT, previous_int)


def test_keyboard_interrupt_still_returns_normally():
    class Interrupted(FakeServer):
        def serve_forever(self):
            raise KeyboardInterrupt

    _serve_until_stopped(Interrupted())
