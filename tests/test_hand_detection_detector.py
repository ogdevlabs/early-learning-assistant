import pytest
from hand_detection.detector import HandDetectorThread

def dummy_frame_provider():
    return None

def dummy_facial_points_provider():
    return {}

def dummy_audio_feedback(msg):
    return f"audio: {msg}"

def test_hand_detector_thread_init():
    thread = HandDetectorThread(
        frame_provider=dummy_frame_provider,
        facial_points_provider=dummy_facial_points_provider,
        log_pointing_callback=None,
        show_hand_markers=True,
        hand_marker_color=(255,255,255),
        index_marker_color=(255,0,0),
        audio_feedback=dummy_audio_feedback
    )
    assert thread.frame_provider == dummy_frame_provider
    assert thread.facial_points_provider == dummy_facial_points_provider
    assert thread.show_hand_markers is True
    assert thread.hand_marker_color == (255,255,255)
    assert thread.index_marker_color == (255,0,0)
    assert thread.audio_feedback == dummy_audio_feedback
    assert thread.running is True

def test_hand_detector_thread_stop():
    thread = HandDetectorThread(dummy_frame_provider, dummy_facial_points_provider)
    thread.running = False
    assert thread.running is False

