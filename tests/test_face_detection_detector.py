import pytest
from face_detection.detector import FaceDetector

class DummyDetector:
    def detect(self, face_landmarks, image_width=None, image_height=None):
        return 'dummy_result'

class DummyLandmarks:
    pass

def test_detect_none_landmarks(monkeypatch):
    fd = FaceDetector()
    # Patch sub-detectors to dummy
    fd.eyes_detector = DummyDetector()
    fd.ears_detector = DummyDetector()
    fd.nose_detector = DummyDetector()
    fd.mouth_detector = DummyDetector()
    assert fd.detect(None) is None

def test_detect_calls_sub_detectors(monkeypatch):
    fd = FaceDetector()
    fd.eyes_detector = DummyDetector()
    fd.ears_detector = DummyDetector()
    fd.nose_detector = DummyDetector()
    fd.mouth_detector = DummyDetector()
    result = fd.detect(DummyLandmarks(), 100, 100)
    assert result == {
        'eyes': 'dummy_result',
        'ears': 'dummy_result',
        'nose': 'dummy_result',
        'mouth': 'dummy_result'
    }

