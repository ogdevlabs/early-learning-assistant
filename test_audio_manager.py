import pytest
import unittest.mock as mock
import numpy as np
from audio.audio_manager import AudioManager
import time
import threading

class TestAudioManager:
    """Unit tests for AudioManager class."""

    def test_initialization(self):
        """Test AudioManager initialization with default parameters."""
        audio_manager = AudioManager()

        assert audio_manager.sample_rate == 16000
        assert audio_manager.channels == 1
        assert audio_manager.chunk_size == 1024
        assert audio_manager.recording == False
        assert audio_manager.playing == False
        assert audio_manager.microphone_enabled == False
        assert audio_manager.speakers_enabled == False
        assert audio_manager.current_input_level == 0.0
        assert audio_manager.current_output_level == 0.0

    def test_initialization_custom_params(self):
        """Test AudioManager initialization with custom parameters."""
        audio_manager = AudioManager(sample_rate=44100, channels=2, chunk_size=512)

        assert audio_manager.sample_rate == 44100
        assert audio_manager.channels == 2
        assert audio_manager.chunk_size == 512

    @mock.patch('audio.audio_manager.sd.query_devices')
    def test_list_devices_success(self, mock_query_devices):
        """Test successful device listing."""
        mock_devices = [
            {'name': 'Built-in Microphone', 'index': 0},
            {'name': 'Built-in Speakers', 'index': 1}
        ]
        mock_query_devices.return_value = mock_devices

        audio_manager = AudioManager()
        devices = audio_manager.list_devices()

        assert devices == mock_devices
        mock_query_devices.assert_called_once()

    @mock.patch('audio.audio_manager.sd.query_devices')
    def test_list_devices_failure(self, mock_query_devices):
        """Test device listing failure."""
        mock_query_devices.side_effect = Exception("Device error")

        audio_manager = AudioManager()
        devices = audio_manager.list_devices()

        assert devices == []

    @mock.patch('audio.audio_manager.sd.query_devices')
    def test_enable_microphone_success(self, mock_query_devices):
        """Test successful microphone enabling."""
        mock_query_devices.return_value = {'name': 'Test Microphone'}

        audio_manager = AudioManager()
        result = audio_manager.enable_microphone()

        assert result == True
        assert audio_manager.microphone_enabled == True

    @mock.patch('audio.audio_manager.sd.query_devices')
    def test_enable_microphone_failure(self, mock_query_devices):
        """Test microphone enabling failure."""
        mock_query_devices.side_effect = Exception("Microphone error")

        audio_manager = AudioManager()
        result = audio_manager.enable_microphone()

        assert result == False
        assert audio_manager.microphone_enabled == False

    @mock.patch('audio.audio_manager.sd.query_devices')
    def test_enable_speakers_success(self, mock_query_devices):
        """Test successful speakers enabling."""
        mock_query_devices.return_value = {'name': 'Test Speakers'}

        audio_manager = AudioManager()
        result = audio_manager.enable_speakers()

        assert result == True
        assert audio_manager.speakers_enabled == True

    def test_start_recording_without_microphone(self):
        """Test recording start failure when microphone not enabled."""
        audio_manager = AudioManager()
        result = audio_manager.start_recording()

        assert result == False
        assert audio_manager.recording == False

    @mock.patch('audio.audio_manager.sd.query_devices')
    @mock.patch('audio.audio_manager.threading.Thread')
    def test_start_recording_success(self, mock_thread, mock_query_devices):
        """Test successful recording start."""
        mock_query_devices.return_value = {'name': 'Test Microphone'}
        mock_thread_instance = mock.Mock()
        mock_thread.return_value = mock_thread_instance

        audio_manager = AudioManager()
        audio_manager.enable_microphone()
        result = audio_manager.start_recording()

        assert result == True
        assert audio_manager.recording == True
        assert audio_manager.running == True
        mock_thread_instance.start.assert_called_once()

    def test_get_audio_data_empty_queue(self):
        """Test getting audio data from empty queue."""
        audio_manager = AudioManager()
        data = audio_manager.get_audio_data()

        assert data is None

    def test_get_audio_data_with_data(self):
        """Test getting audio data from queue with data."""
        audio_manager = AudioManager()
        test_data = np.array([0.1, 0.2, 0.3])
        audio_manager.input_queue.put(test_data)

        data = audio_manager.get_audio_data()
        np.testing.assert_array_equal(data, test_data)

    def test_is_recording(self):
        """Test recording status check."""
        audio_manager = AudioManager()

        assert audio_manager.is_recording() == False

        audio_manager.recording = True
        assert audio_manager.is_recording() == True

    def test_get_input_level(self):
        """Test input level getter."""
        audio_manager = AudioManager()

        assert audio_manager.get_input_level() == 0.0

        audio_manager.current_input_level = 0.5
        assert audio_manager.get_input_level() == 0.5

    def test_set_level_callback(self):
        """Test setting level callback."""
        audio_manager = AudioManager()
        callback = lambda x: x

        audio_manager.set_level_callback(callback)
        assert audio_manager.level_callback == callback

    def test_stop_recording(self):
        """Test stopping recording."""
        audio_manager = AudioManager()
        audio_manager.recording = True

        # Mock stream
        mock_stream = mock.Mock()
        audio_manager.input_stream = mock_stream

        # Mock thread
        mock_thread = mock.Mock()
        mock_thread.is_alive.return_value = True
        audio_manager.input_thread = mock_thread

        audio_manager.stop_recording()

        assert audio_manager.recording == False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_thread.join.assert_called_once_with(timeout=2.0)
        assert audio_manager.input_stream is None

    def test_shutdown(self):
        """Test AudioManager shutdown."""
        audio_manager = AudioManager()

        # Add some test data to queue
        audio_manager.input_queue.put(np.array([1, 2, 3]))

        audio_manager.shutdown()

        assert audio_manager.running == False
        assert audio_manager.input_queue.empty() == True

if __name__ == '__main__':
    pytest.main([__file__])
