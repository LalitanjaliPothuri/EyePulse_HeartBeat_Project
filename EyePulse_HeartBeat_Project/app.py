import time
import cv2
import numpy as np

from utils.face_detection import FaceDetector
from utils.roi_extraction import ROIExtractor
from utils.signal_processing import SignalProcessor
from utils.bpm_calculator import BPMCalculator
from utils.graph_plotter import GraphPlotter
from heart_rate import HeartRateMonitor

face_detector = FaceDetector()
roi_extractor = ROIExtractor()
signal_processor = SignalProcessor(30)
bpm_calculator = BPMCalculator()
monitor = HeartRateMonitor(max_len=360)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
if fps < 5:
    fps = 30
signal_processor.fps = fps

graph_plotter = None
bpm_history = []
frequency_history = []
confidence_history = []

bpm = 0
frequency = 0
confidence = 0.0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_time = time.time()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if graph_plotter is None:
        graph_plotter = GraphPlotter(width=300, height=frame.shape[0])

    results = face_detector.detect_face(rgb)
    graph = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        roi, box = roi_extractor.get_eye_roi(frame, landmarks)

        if roi is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            green = float(np.mean(roi[:, :, 1]))
            monitor.add_signal(green, timestamp=frame_time)
            graph = graph_plotter.update(green)

            if monitor.buffer_length() >= 180 and frame_count % 4 == 0:
                tracking_fps = monitor.get_fps() or fps
                processing_fps = tracking_fps if tracking_fps >= 15 else fps
                signal_processor.fps = processing_fps

                signal = monitor.get_signal()
                filtered = signal_processor.process(signal)
                bpm_candidate, frequency_candidate, confidence_candidate = bpm_calculator.calculate(
                    filtered, processing_fps)

                if bpm_candidate > 0 and confidence_candidate >= 2.5:
                    bpm_history.append(bpm_candidate)
                    frequency_history.append(frequency_candidate)
                    confidence_history.append(confidence_candidate)
                elif not bpm_history and bpm_candidate > 0:
                    bpm_history.append(bpm_candidate)
                    frequency_history.append(frequency_candidate)
                    confidence_history.append(confidence_candidate)

                if len(bpm_history) > 8:
                    bpm_history.pop(0)
                    frequency_history.pop(0)
                    confidence_history.pop(0)

                if bpm_history:
                    bpm = int(np.median(bpm_history))
                    frequency = round(np.median(frequency_history), 2)
                    confidence = round(np.median(confidence_history), 2)

    cv2.putText(frame, f"Heart Rate: {bpm} BPM", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Frequency: {frequency} Hz", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"Quality: {confidence:.2f}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
    cv2.putText(frame, "Keep face steady and well lit", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    combined = np.hstack((frame, graph))
    cv2.imshow("Heart Rate Monitor Dashboard", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()