"""Optional host heap-cache release during loading; never active in timed inference."""
import ctypes
import json
import threading
import time
from pathlib import Path


class LoadingHeapTrim:
    def __init__(self, trace_path, interval=0.25):
        self.trace_path = Path(trace_path)
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._trace = None
        self.samples = []
        self._trim = ctypes.CDLL(None).malloc_trim
        self._trim.argtypes = [ctypes.c_size_t]
        self._trim.restype = ctypes.c_int

    @staticmethod
    def memory():
        available = next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))
        rss = next(int(s.split()[1]) for s in Path('/proc/self/status').read_text().splitlines()
                   if s.startswith('VmRSS:'))
        return {'available_kib': available, 'rss_kib': rss}

    def _sample(self):
        before = self.memory()
        released = bool(self._trim(0))
        row = {'time': time.time(), 'before': before, 'after': self.memory(), 'released': released}
        self.samples.append(row)
        self._trace.write(json.dumps(row) + '\n')
        self._trace.flush()

    def start(self):
        self._trace = self.trace_path.open('x')
        def run():
            while not self._stop.wait(self.interval):
                self._sample()
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def close(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self._sample()
            self._trace.close()
        return {'scope': 'malloc_trim(0) during loading only; thread joined before warmup/timing',
                'samples': len(self.samples),
                'release_calls': sum(x['released'] for x in self.samples),
                'min_available_kib': min((x['after']['available_kib'] for x in self.samples), default=None),
                'max_rss_kib': max((x['before']['rss_kib'] for x in self.samples), default=None),
                'trace_file': self.trace_path.name}
