import numpy as np

class BPMCalculator:

    def calculate(self, signal, fps):
        signal = np.asarray(signal, dtype=float)
        n = len(signal)
        if n < 30:
            return 0, 0, 0.0

        window = np.hamming(n)
        signal = signal * window

        fft = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(n, d=1.0/fps)
        mag = np.abs(fft)

        idx = np.where((freq >= 0.75) & (freq <= 3.5))
        if idx[0].size == 0:
            return 0, 0, 0.0

        freq = freq[idx]
        mag = mag[idx]
        peak_idx = np.argmax(mag)
        peak_freq = freq[peak_idx]

        noise_floor = np.median(mag) if mag.size > 0 else 0.0
        confidence = float(mag[peak_idx] / max(noise_floor, 1e-6))

        bpm = int(round(peak_freq * 60))
        return bpm, round(peak_freq, 2), round(confidence, 2)