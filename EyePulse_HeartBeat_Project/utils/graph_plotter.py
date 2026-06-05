import cv2
import numpy as np

class GraphPlotter:

    def __init__(self, width=400, height=300):
        self.width = width
        self.height = height
        self.data = []

    def update(self, value):

        self.data.append(value)

        if len(self.data) > 200:
            self.data.pop(0)

        graph = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if len(self.data) > 10:

            max_val = max(self.data)
            min_val = min(self.data)

            for i in range(1, len(self.data)):

                x1 = (i - 1) * (self.width // 200)
                x2 = i * (self.width // 200)

                y1 = int(self.height - ((self.data[i - 1] - min_val) /
                     (max_val - min_val + 1e-5)) * self.height)

                y2 = int(self.height - ((self.data[i] - min_val) /
                     (max_val - min_val + 1e-5)) * self.height)

                cv2.line(graph, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return graph