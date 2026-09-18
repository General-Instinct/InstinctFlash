"""FlashRT — Framework-agnostic CUDA Graph capture/replay.

Uses CUDA Runtime API directly via ctypes. Works with any framework
(PyTorch, JAX, or raw CUDA) because it operates at the stream level.

Usage:
    graph = CUDAGraph()
    stream = graph.create_stream()

    # Warmup
    my_kernel(args..., stream)

    # Capture
    graph.begin_capture(stream)
    my_kernel(args..., stream)
    graph.end_capture(stream)

    # Replay (zero dispatch overhead)
    graph.replay(stream)
    graph.sync(stream)
    graph.close()
"""

import ctypes
import logging

logger = logging.getLogger(__name__)

# Load CUDA runtime
_cudart = ctypes.CDLL("libcudart.so")


def _check(status, msg=""):
    if status != 0:
        raise RuntimeError(f"CUDA error {status}: {msg}")


class CUDAGraph:
    """Own one capture and its executable, plus streams created by this object.

    Borrowed streams and captured buffers remain the caller's responsibility.
    Like CUDA graph objects, this wrapper is not thread-safe. Capture/replay use
    the caller's current device; close restores the resource device temporarily.
    Use close (or a with block) for deterministic release, including exceptions.
    """

    def __init__(self):
        self._graph = ctypes.c_void_p()
        self._graph_exec = ctypes.c_void_p()
        self._captured = False
        self._runtime = _cudart  # Keep the library alive through finalization.
        self._owned_streams = []
        self._capture_stream = None
        self._device = None
        self._closed = False

    def _require_open(self):
        if self._closed:
            raise RuntimeError("CUDA graph is closed")

    def _bind_device(self):
        device = ctypes.c_int()
        _check(self._runtime.cudaGetDevice(ctypes.byref(device)), "cudaGetDevice")
        if self._device is not None and self._device != device.value:
            raise RuntimeError("CUDA graph resources belong to a different device")
        self._device = device.value

    def create_stream(self) -> ctypes.c_void_p:
        """Create a new CUDA stream for capture."""
        self._require_open()
        self._bind_device()
        stream = ctypes.c_void_p()
        _check(self._runtime.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        self._owned_streams.append(ctypes.c_void_p(stream.value))
        return stream

    def begin_capture(self, stream: ctypes.c_void_p):
        """Begin CUDA Graph capture on the given stream.

        All CUDA operations on this stream after this call will be
        recorded into the graph instead of being executed.
        """
        self._require_open()
        if self._capture_stream is not None or self._captured:
            raise RuntimeError("Use a new CUDA graph for another capture")
        self._bind_device()
        # cudaStreamCaptureModeRelaxed=2: only capture ops on THIS stream.
        # Global mode (0) blocks ALL streams, conflicting with XLA's background ops.
        _check(self._runtime.cudaStreamBeginCapture(stream, 2), "cudaStreamBeginCapture")
        self._capture_stream = ctypes.c_void_p(stream.value)

    def end_capture(self, stream: ctypes.c_void_p):
        """End capture and instantiate the graph for replay."""
        self._require_open()
        if self._capture_stream is None or stream.value != self._capture_stream.value:
            raise RuntimeError("End capture on the stream that began it")
        self._bind_device()
        try:
            _check(self._end_capture(), "cudaStreamEndCapture")
            _check(self._runtime.cudaGraphInstantiate(
                ctypes.byref(self._graph_exec), self._graph, 0),
                   "cudaGraphInstantiate")
        except Exception:
            self._close_safely()
            raise
        self._captured = True

    def _end_capture(self):
        status = self._runtime.cudaStreamEndCapture(
            self._capture_stream, ctypes.byref(self._graph))
        if status in (0, 901):  # cudaSuccess, cudaErrorStreamCaptureInvalidated
            self._capture_stream = None
        else:
            # Other failures (e.g. unjoined capture branches) may also terminate
            # capture. Query before deciding whether the stream can be released.
            capturing = ctypes.c_int()
            query = self._runtime.cudaStreamIsCapturing(
                self._capture_stream, ctypes.byref(capturing))
            if query == 0 and capturing.value == 0:  # cudaStreamCaptureStatusNone
                self._capture_stream = None
        return status

    def replay(self, stream: ctypes.c_void_p):
        """Replay the captured graph (single CPU instruction → GPU replay)."""
        self._require_open()
        if not self._captured:
            raise RuntimeError("No graph captured")
        _check(self._runtime.cudaGraphLaunch(self._graph_exec, stream), "cudaGraphLaunch")

    def sync(self, stream: ctypes.c_void_p):
        """Synchronize stream."""
        self._require_open()
        _check(self._runtime.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def close(self):
        """Release owned resources; repeat calls are harmless after success.

        End an abandoned capture without instantiating it. CUDA defers release
        of in-flight executable graphs/streams until their work completes; this
        does not synchronize or extend the lifetime of caller-owned buffers.
        Failed releases remain tracked for an explicit retry, while other
        resources are still cleaned up. The object cannot be reused after close.
        """
        self._closed = True
        self._captured = False
        if not (self._capture_stream is not None or self._graph.value
                or self._graph_exec.value or self._owned_streams):
            return
        runtime = self._runtime
        previous = ctypes.c_int()
        _check(runtime.cudaGetDevice(ctypes.byref(previous)), "cudaGetDevice")
        changed = previous.value != self._device
        if changed:
            _check(runtime.cudaSetDevice(self._device), "cudaSetDevice")
        errors = []
        try:
            if self._capture_stream is not None:
                status = self._end_capture()
                if status not in (0, 901):
                    errors.append((status, "cudaStreamEndCapture"))
            for name, handle in (("cudaGraphExecDestroy", self._graph_exec),
                                 ("cudaGraphDestroy", self._graph)):
                if handle.value:
                    status = getattr(runtime, name)(handle)
                    if status == 0:
                        handle.value = None
                    else:
                        errors.append((status, name))
            for stream in self._owned_streams[:]:
                # Never destroy a stream whose capture could not be terminated.
                if (self._capture_stream is not None
                        and stream.value == self._capture_stream.value):
                    continue
                status = runtime.cudaStreamDestroy(stream)
                if status == 0:
                    self._owned_streams.remove(stream)
                else:
                    errors.append((status, "cudaStreamDestroy"))
        finally:
            if changed:
                status = runtime.cudaSetDevice(previous.value)
                if status:
                    errors.append((status, "cudaSetDevice (restore)"))
        if errors:
            _check(*errors[0])

    def _close_safely(self):
        try:
            self.close()
        except Exception:
            # Destruction/runtime shutdown must not mask a capture exception.
            logger.debug("CUDA graph cleanup failed", exc_info=True)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self._close_safely()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Includes partial initialization and interpreter/CUDA shutdown.

    @property
    def captured(self) -> bool:
        return self._captured
