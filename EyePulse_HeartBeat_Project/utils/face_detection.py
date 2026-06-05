import mediapipe as mp

class FaceDetector:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def detect_face(self, frame_rgb):
        return self.face_mesh.process(frame_rgb)