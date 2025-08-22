import pytest
import unittest.mock as mock
import numpy as np
import time
from audio.voice_recognition import VoiceRecognitionThread
from audio.audio_manager import AudioManager

class TestVoiceRecognitionThread:
    """Unit tests for VoiceRecognitionThread class."""

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_initialization(self, mock_recognizer):
        """Test VoiceRecognitionThread initialization."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_audio_manager.sample_rate = 16000
        mock_callback = mock.Mock()

        voice_recognition = VoiceRecognitionThread(
            audio_manager=mock_audio_manager,
            command_callback=mock_callback,
            recognition_language='en-US',
            confidence_threshold=0.8
        )

        assert voice_recognition.audio_manager == mock_audio_manager
        assert voice_recognition.command_callback == mock_callback
        assert voice_recognition.recognition_language == 'en-US'
        assert voice_recognition.confidence_threshold == 0.8
        assert voice_recognition.running == False
        assert voice_recognition.daemon == True
        assert voice_recognition.total_recognitions == 0
        assert voice_recognition.successful_recognitions == 0
        assert voice_recognition.failed_recognitions == 0

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_process_audio_chunk_voice_activity(self, mock_recognizer):
        """Test audio chunk processing with voice activity."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_audio_manager.sample_rate = 16000

        voice_recognition = VoiceRecognitionThread(mock_audio_manager)

        # Create audio data with sufficient level
        audio_data = np.array([0.1, 0.2, 0.3, 0.2, 0.1])

        voice_recognition._process_audio_chunk(audio_data)

        assert len(voice_recognition.audio_buffer) == 1
        assert voice_recognition.buffer_duration > 0

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_process_audio_chunk_silence(self, mock_recognizer):
        """Test audio chunk processing with silence."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_audio_manager.sample_rate = 16000

        voice_recognition = VoiceRecognitionThread(mock_audio_manager)

        # Create audio data with low level (silence)
        audio_data = np.array([0.001, 0.002, 0.001, 0.002, 0.001])

        voice_recognition._process_audio_chunk(audio_data)

        assert len(voice_recognition.audio_buffer) == 0
        assert voice_recognition.buffer_duration == 0

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_clear_buffer(self, mock_recognizer):
        """Test buffer clearing."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        voice_recognition = VoiceRecognitionThread(mock_audio_manager)

        # Add some data to buffer
        voice_recognition.audio_buffer = [np.array([1, 2, 3])]
        voice_recognition.buffer_duration = 1.0

        voice_recognition._clear_buffer()

        assert voice_recognition.audio_buffer == []
        assert voice_recognition.buffer_duration == 0.0

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_process_voice_command_exact_match(self, mock_recognizer):
        """Test voice command processing with exact match."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_callback = mock.Mock()

        voice_recognition = VoiceRecognitionThread(mock_audio_manager, mock_callback)

        voice_recognition._process_voice_command("hello")

        mock_callback.assert_called_once_with("hello")

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_process_voice_command_alias_match(self, mock_recognizer):
        """Test voice command processing with alias match."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_callback = mock.Mock()

        voice_recognition = VoiceRecognitionThread(mock_audio_manager, mock_callback)

        voice_recognition._process_voice_command("hi")  # Alias for "hello"

        mock_callback.assert_called_once_with("hello")

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_process_voice_command_partial_match(self, mock_recognizer):
        """Test voice command processing with partial match."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_callback = mock.Mock()

        voice_recognition = VoiceRecognitionThread(mock_audio_manager, mock_callback)

        voice_recognition._process_voice_command("start detection now please")

        mock_callback.assert_called_once_with("start_detection")

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_process_voice_command_no_match(self, mock_recognizer):
        """Test voice command processing with no match."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_callback = mock.Mock()

        voice_recognition = VoiceRecognitionThread(mock_audio_manager, mock_callback)

        voice_recognition._process_voice_command("unknown command")

        mock_callback.assert_called_once_with("unknown command")

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    @mock.patch('audio.voice_recognition.sr.AudioData')
    def test_recognize_speech_success(self, mock_audio_data, mock_recognizer):
        """Test successful speech recognition."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_callback = mock.Mock()

        # Mock recognizer instance
        mock_recognizer_instance = mock.Mock()
        mock_recognizer_instance.recognize_google.return_value = "Hello World"
        mock_recognizer.return_value = mock_recognizer_instance

        voice_recognition = VoiceRecognitionThread(mock_audio_manager, mock_callback)
        voice_recognition.recognizer = mock_recognizer_instance

        mock_audio = mock.Mock()
        voice_recognition._recognize_speech(mock_audio)

        assert voice_recognition.total_recognitions == 1
        assert voice_recognition.successful_recognitions == 1
        assert voice_recognition.failed_recognitions == 0

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_recognize_speech_unknown_value(self, mock_recognizer):
        """Test speech recognition with unknown value error."""
        import speech_recognition as sr
        mock_audio_manager = mock.Mock(spec=AudioManager)

        # Mock recognizer instance
        mock_recognizer_instance = mock.Mock()
        mock_recognizer_instance.recognize_google.side_effect = sr.UnknownValueError()
        mock_recognizer.return_value = mock_recognizer_instance

        voice_recognition = VoiceRecognitionThread(mock_audio_manager)
        voice_recognition.recognizer = mock_recognizer_instance

        mock_audio = mock.Mock()
        voice_recognition._recognize_speech(mock_audio)

        assert voice_recognition.total_recognitions == 1
        assert voice_recognition.successful_recognitions == 0
        assert voice_recognition.failed_recognitions == 1

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_get_statistics(self, mock_recognizer):
        """Test getting recognition statistics."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        voice_recognition = VoiceRecognitionThread(mock_audio_manager)

        # Set some statistics
        voice_recognition.total_recognitions = 10
        voice_recognition.successful_recognitions = 8
        voice_recognition.failed_recognitions = 2

        stats = voice_recognition.get_statistics()

        assert stats['total_recognitions'] == 10
        assert stats['successful_recognitions'] == 8
        assert stats['failed_recognitions'] == 2
        assert stats['success_rate'] == 80.0

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_update_sensitivity(self, mock_recognizer):
        """Test updating recognition sensitivity."""
        mock_audio_manager = mock.Mock(spec=AudioManager)

        mock_recognizer_instance = mock.Mock()
        mock_recognizer.return_value = mock_recognizer_instance

        voice_recognition = VoiceRecognitionThread(mock_audio_manager)
        voice_recognition.recognizer = mock_recognizer_instance

        voice_recognition.update_sensitivity(500)

        assert voice_recognition.recognizer.energy_threshold == 500

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_add_custom_command(self, mock_recognizer):
        """Test adding custom voice command."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        voice_recognition = VoiceRecognitionThread(mock_audio_manager)

        custom_handler = mock.Mock()
        voice_recognition.add_custom_command("custom command", custom_handler)

        assert "custom command" in voice_recognition.voice_commands
        assert voice_recognition.voice_commands["custom command"] == custom_handler

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_remove_command(self, mock_recognizer):
        """Test removing voice command."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        voice_recognition = VoiceRecognitionThread(mock_audio_manager)

        # Add a command first
        custom_handler = mock.Mock()
        voice_recognition.add_custom_command("test command", custom_handler)

        # Remove it
        voice_recognition.remove_command("test command")

        assert "test command" not in voice_recognition.voice_commands

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_stop(self, mock_recognizer):
        """Test stopping voice recognition."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        voice_recognition = VoiceRecognitionThread(mock_audio_manager)

        voice_recognition.running = True
        voice_recognition.stop()

        assert voice_recognition.running == False

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_is_running(self, mock_recognizer):
        """Test checking if voice recognition is running."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        voice_recognition = VoiceRecognitionThread(mock_audio_manager)

        assert voice_recognition.is_running() == False

        voice_recognition.running = True
        assert voice_recognition.is_running() == True

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_handle_speech_timeout(self, mock_recognizer):
        """Test speech timeout handling."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        voice_recognition = VoiceRecognitionThread(mock_audio_manager)

        # Add some data to buffer
        voice_recognition.audio_buffer = [np.array([1, 2, 3])]

        voice_recognition._handle_speech_timeout()

        assert voice_recognition.audio_buffer == []

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_command_handlers(self, mock_recognizer):
        """Test various command handlers."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_callback = mock.Mock()

        voice_recognition = VoiceRecognitionThread(mock_audio_manager, mock_callback)

        # Test start detection handler
        voice_recognition._handle_start_detection()
        mock_callback.assert_called_with("start_detection")

        # Test stop detection handler
        voice_recognition._handle_stop_detection()
        mock_callback.assert_called_with("stop_detection")

        # Test show hands handler
        voice_recognition._handle_show_hands()
        mock_callback.assert_called_with("show_hands")

    @mock.patch('audio.voice_recognition.sr.Recognizer')
    def test_statistics_handler(self, mock_recognizer):
        """Test statistics command handler."""
        mock_audio_manager = mock.Mock(spec=AudioManager)
        mock_callback = mock.Mock()

        voice_recognition = VoiceRecognitionThread(mock_audio_manager, mock_callback)

        # Set some statistics
        voice_recognition.total_recognitions = 5
        voice_recognition.successful_recognitions = 4

        voice_recognition._handle_statistics()

        # Check that callback was called with stats message
        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0][0]
        assert call_args.startswith("stats:")
        assert "80.0%" in call_args

if __name__ == '__main__':
    pytest.main([__file__])
