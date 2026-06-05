import numpy as np
from scipy.signal import butter, filtfilt, detrend

class SignalProcessor:

    def __init__(self, fps):
        self.fps = fps

    def process(self, signal):
        signal = np.asarray(signal, dtype=float)
        if len(signal) < 3:
            return signal

        signal = detrend(signal)
        signal = self._normalize(signal)
        return self.bandpass(signal)

    def bandpass(self, signal):
        low, high = 0.75, 3.5
        nyq = 0.5 * self.fps
        low_cut = low / nyq
        high_cut = min(high / nyq, 0.99)

        b, a = butter(3, [low_cut, high_cut], btype='band')
        return filtfilt(b, a, signal)

    def _normalize(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-6:
            return signal - mean
        return (signal - mean) / std