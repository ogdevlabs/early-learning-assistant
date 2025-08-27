import pytest
from app_flow.system import FaceHandInteractionSystem

class DummyScorer:
    pass

@pytest.fixture
def system():
    return FaceHandInteractionSystem(DummyScorer())

def test_select_new_target_changes_target(system):
    original = system.current_target
    system._select_new_target()
    assert system.current_target != original

def test_facial_labels_are_valid(system):
    assert set(system.facial_labels) == {"Pointing Nose", "Pointing Mouth", "Pointing Eyes", "Pointing Ears"}

def test_face_proximity_threshold_default(system):
    assert system.face_proximity_threshold == 110

def test_threshold_adjustment_step_default(system):
    assert system.threshold_adjustment_step == 10

# For methods requiring frame/landmarks, you would use mocks or synthetic data in more advanced tests.

