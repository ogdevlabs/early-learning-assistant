import pytest
import unittest.mock as mock
import time
import threading
from audio.audio_feedback import AudioFeedback

class TestAudioFeedback:
    """Unit tests for AudioFeedback class."""

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_initialization_success(self, mock_pyttsx3_init):
        """Test successful AudioFeedback initialization."""
        mock_engine = mock.Mock()
        # Mock the voices property to return an empty list to avoid len() error
        mock_engine.getProperty.return_value = []
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback(voice_rate=200, voice_volume=0.9, voice_id=1)

        assert audio_feedback.voice_rate == 200
        assert audio_feedback.voice_volume == 0.9
        assert audio_feedback.voice_id == 1
        assert audio_feedback.running == False
        assert audio_feedback.speaking == False
        assert audio_feedback.muted == False
        assert audio_feedback.tts_engine == mock_engine

        # Verify TTS engine was configured
        mock_engine.setProperty.assert_any_call('rate', 200)
        mock_engine.setProperty.assert_any_call('volume', 0.9)

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_initialization_failure(self, mock_pyttsx3_init):
        """Test AudioFeedback initialization failure."""
        mock_pyttsx3_init.side_effect = Exception("TTS init failed")

        audio_feedback = AudioFeedback()

        assert audio_feedback.tts_engine is None

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_start_success(self, mock_pyttsx3_init):
        """Test successful audio feedback start."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        result = audio_feedback.start()

        assert result == True
        assert audio_feedback.running == True
        assert audio_feedback.tts_thread is not None
        assert audio_feedback.tts_thread.daemon == True

    def test_start_without_engine(self):
        """Test start failure when TTS engine not initialized."""
        with mock.patch('audio.audio_feedback.pyttsx3.init', side_effect=Exception("TTS failed")):
            audio_feedback = AudioFeedback()
            result = audio_feedback.start()

            assert result == False
            assert audio_feedback.running == False

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_stop(self, mock_pyttsx3_init):
        """Test audio feedback stop."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        # Add something to queue
        audio_feedback.tts_queue.put("test")

        audio_feedback.stop()

        assert audio_feedback.running == False
        assert audio_feedback.tts_queue.empty() == True

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_speak_success(self, mock_pyttsx3_init):
        """Test successful speak operation."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        result = audio_feedback.speak("Hello world")

        assert result == True
        assert audio_feedback.tts_queue.qsize() == 1

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_speak_not_running(self, mock_pyttsx3_init):
        """Test speak when not running."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        # Don't start

        result = audio_feedback.speak("Hello world")

        assert result == False

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_speak_empty_text(self, mock_pyttsx3_init):
        """Test speak with empty text."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        result = audio_feedback.speak("")

        assert result == False
        assert audio_feedback.tts_queue.qsize() == 0

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_speak_priority(self, mock_pyttsx3_init):
        """Test priority speak clears queue."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        # Add normal message
        audio_feedback.speak("Normal message")
        assert audio_feedback.tts_queue.qsize() == 1

        # Add priority message
        result = audio_feedback.speak("Priority message", priority=True)

        assert result == True
        assert audio_feedback.tts_queue.qsize() == 1  # Queue was cleared and new message added

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_speak_response_success(self, mock_pyttsx3_init):
        """Test successful speak_response."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        result = audio_feedback.speak_response("hello")

        assert result == True
        assert audio_feedback.tts_queue.qsize() == 1

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_speak_response_unknown_key(self, mock_pyttsx3_init):
        """Test speak_response with unknown key."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        result = audio_feedback.speak_response("unknown_key")

        assert result == False
        assert audio_feedback.tts_queue.qsize() == 0

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_announce_detection(self, mock_pyttsx3_init):
        """Test announce_detection method."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        # Test face detection
        audio_feedback.announce_detection("face", True)
        assert audio_feedback.tts_queue.qsize() == 1

        # Test hands detection
        audio_feedback.announce_detection("hands", False)
        assert audio_feedback.tts_queue.qsize() == 2

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_announce_pointing(self, mock_pyttsx3_init):
        """Test announce_pointing method."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        audio_feedback.announce_pointing("nose")
        assert audio_feedback.tts_queue.qsize() == 1

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_announce_pointing_held(self, mock_pyttsx3_init):
        """Test announce_pointing_held method."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        audio_feedback.announce_pointing_held("nose", 3.5)
        assert audio_feedback.tts_queue.qsize() == 1

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_set_muted(self, mock_pyttsx3_init):
        """Test mute functionality."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()

        audio_feedback.set_muted(True)
        assert audio_feedback.muted == True

        audio_feedback.set_muted(False)
        assert audio_feedback.muted == False

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_is_speaking(self, mock_pyttsx3_init):
        """Test is_speaking status."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()

        assert audio_feedback.is_speaking() == False

        audio_feedback.speaking = True
        assert audio_feedback.is_speaking() == True

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_is_running(self, mock_pyttsx3_init):
        """Test is_running status."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()

        assert audio_feedback.is_running() == False

        audio_feedback.start()
        assert audio_feedback.is_running() == True

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_get_available_voices(self, mock_pyttsx3_init):
        """Test get_available_voices method."""
        mock_engine = mock.Mock()
        mock_voice1 = mock.Mock()
        mock_voice1.name = "Voice 1"
        mock_voice1.id = "voice1"
        mock_voice2 = mock.Mock()
        mock_voice2.name = "Voice 2"
        mock_voice2.id = "voice2"

        mock_engine.getProperty.return_value = [mock_voice1, mock_voice2]
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        voices = audio_feedback.get_available_voices()

        assert len(voices) == 2
        assert voices[0] == (0, "Voice 1", "voice1")
        assert voices[1] == (1, "Voice 2", "voice2")

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_set_voice_properties(self, mock_pyttsx3_init):
        """Test set_voice_properties method."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.set_voice_properties(rate=180, volume=0.7)

        assert audio_feedback.voice_rate == 180
        assert audio_feedback.voice_volume == 0.7
        mock_engine.setProperty.assert_any_call('rate', 180)
        mock_engine.setProperty.assert_any_call('volume', 0.7)

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_add_custom_response(self, mock_pyttsx3_init):
        """Test add_custom_response method."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.add_custom_response("custom_key", "Custom response text")

        assert "custom_key" in audio_feedback.responses
        assert audio_feedback.responses["custom_key"] == "Custom response text"

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_get_queue_size(self, mock_pyttsx3_init):
        """Test get_queue_size method."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        assert audio_feedback.get_queue_size() == 0

        audio_feedback.speak("Test message")
        assert audio_feedback.get_queue_size() == 1

    @mock.patch('audio.audio_feedback.pyttsx3.init')
    def test_clear_queue(self, mock_pyttsx3_init):
        """Test clear_queue method."""
        mock_engine = mock.Mock()
        mock_pyttsx3_init.return_value = mock_engine

        audio_feedback = AudioFeedback()
        audio_feedback.start()

        # Add messages to queue
        audio_feedback.speak("Message 1")
        audio_feedback.speak("Message 2")
        assert audio_feedback.get_queue_size() == 2

        audio_feedback.clear_queue()
        assert audio_feedback.get_queue_size() == 0

if __name__ == '__main__':
    pytest.main([__file__])
