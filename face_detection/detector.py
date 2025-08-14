from .eyes import EyesDetector
from .ears import EarsDetector
from .nose import NoseDetector
from .mouth import MouthDetector

class FaceDetector:
    def __init__(self):
        self.eyes_detector = EyesDetector()
        self.ears_detector = EarsDetector()
        self.nose_detector = NoseDetector()
        self.mouth_detector = MouthDetector()

    def detect(self, face_landmarks, image_width=None, image_height=None):
        """
        Detects facial features using the respective detectors.
        Args:
            face_landmarks: The facial landmarks data (from mediapipe FaceMesh).
            image_width: Width of the image (for pixel coordinates).
            image_height: Height of the image (for pixel coordinates).
        Returns:
            dict: {
                'eyes': ...,
                'ears': ...,
                'nose': ...,
                'mouth': ...
            } or None if not found.
        """
        if not face_landmarks:
            return None
        return {
            'eyes': self.eyes_detector.detect(face_landmarks, image_width, image_height),
            'ears': self.ears_detector.detect(face_landmarks, image_width, image_height),
            'nose': self.nose_detector.detect(face_landmarks, image_width, image_height),
            'mouth': self.mouth_detector.detect(face_landmarks, image_width, image_height)
        }

