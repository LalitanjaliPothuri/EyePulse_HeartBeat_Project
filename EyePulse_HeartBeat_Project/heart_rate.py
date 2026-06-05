import time
import numpy as np

class HeartRateMonitor:

    def __init__(self, max_len=300):
        self.buffer = []
        self.timestamps = []
        self.max_len = max_len

    def add_signal(self, val, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        self.buffer.append(val)
        self.timestamps.append(timestamp)

        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)
            self.timestamps.pop(0)

    def get_signal(self):
        return np.array(self.buffer, dtype=float)

    def buffer_length(self):
        return len(self.buffer)

    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        duration = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / max(duration, 1e-6)
