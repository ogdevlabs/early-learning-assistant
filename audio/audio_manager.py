import threading
import logging
import sounddevice as sd
import numpy as np
import queue
import time
from typing import Optional, Callable

class AudioManager:
    """
    Central audio management system handling both input (microphone) and output (speakers).
    Provides thread-safe audio capture and playback capabilities.
    """

    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        """
        Initialize AudioManager with audio parameters.

        Args:
            sample_rate (int): Audio sample rate in Hz
            channels (int): Number of audio channels
            chunk_size (int): Audio buffer size
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        # Audio queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # State management
        self.recording = False
        self.playing = False
        self.microphone_enabled = False
        self.speakers_enabled = False

        # Audio streams
        self.input_stream = None
        self.output_stream = None

        # Threading
        self.input_thread = None
        self.output_thread = None
        self.running = False

        # Callbacks
        self.audio_callback: Optional[Callable] = None
        self.level_callback: Optional[Callable] = None

        # Audio level monitoring
        self.current_input_level = 0.0
        self.current_output_level = 0.0

        logging.info(f"AudioManager initialized: {sample_rate}Hz, {channels}ch, {chunk_size} buffer")

    def list_devices(self):
        """List all available audio devices."""
        try:
            devices = sd.query_devices()
            logging.info(f"Found {len(devices)} audio devices")
            return devices
        except Exception as e:
            logging.error(f"Failed to list audio devices: {e}")
            return []

    def enable_microphone(self, device_id=None):
        """
        Enable microphone with optional device selection.

        Args:
            device_id (int, optional): Specific device ID to use

        Returns:
            bool: True if successful
        """
        try:
            # Verify device availability
            if device_id is not None:
                device_info = sd.query_devices(device_id, 'input')
                logging.info(f"Selected input device: {device_info['name']}")
            else:
                device_info = sd.query_devices(kind='input')
                logging.info(f"Using default input device: {device_info['name']}")

            self.microphone_enabled = True
            logging.info("Microphone enabled successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to enable microphone: {e}")
            return False

    def enable_speakers(self, device_id=None):
        """
        Enable speakers with optional device selection.

        Args:
            device_id (int, optional): Specific device ID to use

        Returns:
            bool: True if successful
        """
        try:
            # Verify device availability
            if device_id is not None:
                device_info = sd.query_devices(device_id, 'output')
                logging.info(f"Selected output device: {device_info['name']}")
            else:
                device_info = sd.query_devices(kind='output')
                logging.info(f"Using default output device: {device_info['name']}")

            self.speakers_enabled = True
            logging.info("Speakers enabled successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to enable speakers: {e}")
            return False

    def start_recording(self, callback=None):
        """
        Start audio recording.

        Args:
            callback (callable, optional): Function to call with audio data

        Returns:
            bool: True if successful
        """
        if not self.microphone_enabled:
            logging.error("Microphone not enabled. Call enable_microphone() first.")
            return False

        if self.recording:
            logging.warning("Recording already in progress")
            return True

        try:
            self.audio_callback = callback
            self.running = True
            self.recording = True

            self.input_thread = threading.Thread(target=self._input_capture_loop, daemon=True)
            self.input_thread.start()

            logging.info("Audio recording started")
            return True

        except Exception as e:
            logging.error(f"Failed to start recording: {e}")
            self.recording = False
            self.running = False
            return False

    def stop_recording(self):
        """Stop audio recording."""
        self.recording = False

        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=2.0)

        logging.info("Audio recording stopped")

    def _input_capture_loop(self):
        """Main audio input capture loop."""
        def input_callback(indata, frames, time, status):
            if status:
                logging.warning(f"Input callback status: {status}")

            # Calculate input level
            self.current_input_level = float(np.abs(indata).mean())

            # Put audio data in queue
            audio_data = indata.copy()
            try:
                self.input_queue.put_nowait(audio_data)

                # Call external callback if provided
                if self.audio_callback:
                    self.audio_callback(audio_data)

            except queue.Full:
                logging.warning("Input queue full, dropping frames")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=input_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            ) as stream:
                self.input_stream = stream
                logging.info(f"Input stream started: {self.sample_rate}Hz, {self.channels}ch")

                while self.running and self.recording:
                    time.sleep(0.1)

        except Exception as e:
            logging.error(f"Input capture error: {e}")
        finally:
            self.recording = False
            logging.info("Input capture loop ended")

    def get_audio_data(self):
        """
        Get audio data from input queue (non-blocking).

        Returns:
            numpy.ndarray or None: Audio data or None if queue empty
        """
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

    def is_recording(self):
        """Check if currently recording."""
        return self.recording

    def get_input_level(self):
        """Get current audio input level (0.0 to 1.0)."""
        return self.current_input_level

    def set_level_callback(self, callback):
        """Set callback for audio level monitoring."""
        self.level_callback = callback

    def shutdown(self):
        """Shutdown audio manager and cleanup resources."""
        self.running = False
        self.stop_recording()

        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

        logging.info("AudioManager shutdown complete")
