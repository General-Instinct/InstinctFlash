"""A verbose worker must not deadlock before its socket is bound."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from instinctflash.runtime.execution import WorkerBackend


class WorkerOutput(unittest.TestCase):
    def test_verbose_startup_is_drained_and_failure_tail_is_retained(self):
        class Adapter:
            def worker_command(self, checkpoint, plan, **kwargs):
                script = "import sys; print('progress\\n'*300000); print('specific loader failure'); sys.exit(3)"
                return [sys.executable, '-u', '-c', script], {}
        backend = WorkerBackend(Adapter(), SimpleNamespace(), None, startup_timeout_s=5)
        try:
            with self.assertRaisesRegex(RuntimeError, 'specific loader failure'):
                backend._spawn()
            self.assertLessEqual(len(backend._log_chunks), 16)
        finally:
            backend.close()

    def test_shipped_client_roundtrips_arrays_and_closes_without_native_imports(self):
        try:
            import numpy as np
            import websockets.sync.client
            import msgpack
        except ImportError:
            self.skipTest('worker wire extras are not installed')
        from instinctflash.runtime.execution import _connect_client
        from instinctflash.serving.msgpack_numpy import Packer, unpackb
        packer = Packer()
        replies = iter([packer.pack({}), packer.pack({'action': np.arange(3)}), 'native error'])
        class Connection:
            closed = False
            def recv(self): return next(replies)
            def send(self, value): self.sent = value
            def close(self): self.closed = True
        connection = Connection()
        with patch.dict('os.environ', {'LINGBOT_ROOT': '/no/native/client'}), patch(
                'websockets.sync.client.connect', return_value=connection):
            client = _connect_client(29990)
        np.testing.assert_array_equal(client.infer({'obs': np.ones(2)})['action'], np.arange(3))
        np.testing.assert_array_equal(unpackb(connection.sent)['obs'], np.ones(2))
        with self.assertRaisesRegex(RuntimeError, 'native error'):
            client.infer({})
        client.close()
        self.assertTrue(connection.closed)


if __name__ == '__main__':
    unittest.main()
