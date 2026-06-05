import numpy as np

class ROIExtractor:

    def get_eye_roi(self, frame, landmarks):
        h, w, _ = frame.shape

        eye_ids = [33, 133, 160, 159, 158, 157, 173,
                   362, 263, 387, 386, 385, 384, 398]

        pts = []
        for i in eye_ids:
            lm = landmarks.landmark[i]
            x, y = int(lm.x * w), int(lm.y * h)
            pts.append((x, y))

        pts = np.array(pts)
        x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
        x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])

        x1, y1 = max(0, x1 - 20), max(0, y1 - 20)
        x2, y2 = min(w, x2 + 20), min(h, y2 + 20)

        if x2 - x1 < 40 or y2 - y1 < 40:
            return None, None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None

        return roi, (x1, y1, x2, y2)