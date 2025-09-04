import threading
import logging
import sounddevice as sd
import numpy as np
import queue
import time
import json
import os
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
        """List all available audio devices and return list (for tests)."""
        try:
            devices = sd.query_devices()
            if os.getenv('ELA_AUDIO_DIAG'):
                print("Available audio devices:")
                for idx, dev in enumerate(devices):
                    dev_type = []
                    if dev['max_input_channels'] > 0:
                        dev_type.append('Input')
                    if dev['max_output_channels'] > 0:
                        dev_type.append('Output')
                    print(f"[{idx}] {dev['name']} - {', '.join(dev_type)}")
            logging.debug(f"Found {len(devices)} audio devices")
            return devices
        except Exception as e:
            logging.error(f"Error listing audio devices: {e}")
            return []

    def select_input_device(self, device_index: int):
        """Select the input device by index."""
        try:
            devices = sd.query_devices()
            if 0 <= device_index < len(devices):
                if devices[device_index]['max_input_channels'] > 0:
                    self.input_device = device_index
                    logging.info(f"Selected input device: [{device_index}] {devices[device_index]['name']}")
                else:
                    logging.warning(f"Device [{device_index}] is not an input device.")
            else:
                logging.warning(f"Device index {device_index} out of range.")
        except Exception as e:
            logging.error(f"Error selecting input device: {e}")

    def get_current_input_device(self):
        """Return the currently selected input device info."""
        try:
            if hasattr(self, 'input_device'):
                dev = sd.query_devices()[self.input_device]
                return {'index': self.input_device, 'name': dev['name'], 'max_input_channels': dev['max_input_channels']}
            else:
                return None
        except Exception as e:
            logging.error(f"Error getting current input device: {e}")
            return None

    def _get_admin_config(self):
        """Read device selection from audio_device_config.json if available."""
        config_path = os.path.join(os.path.dirname(__file__), 'audio_device_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config
            except Exception as e:
                logging.error(f"Error reading admin config: {e}")
        return None

    def enable_microphone(self, device_id=None):
        """
        Enable microphone with optional device selection. Uses admin config if available.

        Args:
            device_id (int, optional): Specific device ID to use

        Returns:
            bool: True if successful
        """
        try:
            admin_config = self._get_admin_config()
            if admin_config and 'input_device' in admin_config:
                device_id = admin_config['input_device']
                logging.info(f"Using admin-selected input device: {device_id}")

            devices = sd.query_devices()
            selected_info = None
            if device_id is not None:
                # Validate index
                if 0 <= device_id < len(devices) and devices[device_id]['max_input_channels'] > 0:
                    selected_info = sd.query_devices(device_id, 'input')
                else:
                    raise Exception(f"Provided device_id {device_id} invalid or not input-capable")
            else:
                # Default input device index from sounddevice defaults
                default_input_index = sd.default.device[0] if sd.default.device else None
                if default_input_index is not None and default_input_index >= 0:
                    selected_info = sd.query_devices(default_input_index, 'input')
                    device_id = default_input_index
                else:
                    # Fallback: first device with input channels
                    for idx, dev in enumerate(devices):
                        if dev['max_input_channels'] > 0:
                            selected_info = sd.query_devices(idx, 'input')
                            device_id = idx
                            break
            if not selected_info:
                raise Exception("No suitable input device found")

            default_sr = selected_info.get('default_samplerate')
            if default_sr and abs(default_sr - self.sample_rate) > 1:
                logging.warning(f"Device default SR {default_sr} differs from requested {self.sample_rate}")

            self.input_device = device_id
            self.microphone_enabled = True
            logging.info(f"Microphone enabled on device {device_id} '{selected_info['name']}' sr={self.sample_rate} ch={self.channels}")
            return True
        except Exception as e:
            logging.error(f"Failed to enable microphone: {e}")
            return False

    def enable_speakers(self, device_id=None):
        """
        Enable speakers with optional device selection. Uses admin config if available.

        Args:
            device_id (int, optional): Specific device ID to use

        Returns:
            bool: True if successful
        """
        try:
            admin_config = self._get_admin_config()
            if admin_config and 'output_device' in admin_config:
                device_id = admin_config['output_device']
                logging.info(f"Using admin-selected output device: {device_id}")

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
        """Main audio input capture loop with diagnostics."""
        inactivity_logged = False
        start_time = time.time()
        def input_callback(indata, frames, time_info, status):
            if status:
                logging.warning(f"Input callback status: {status}")
            self.current_input_level = float(np.abs(indata).mean())
            if os.getenv('ELA_AUDIO_DIAG'):
                print(f"[AudioManager] lvl={self.current_input_level:.5f}")
            audio_data = indata.copy()
            try:
                self.input_queue.put_nowait(audio_data)
                if self.audio_callback:
                    self.audio_callback(audio_data)
            except queue.Full:
                logging.warning("Input queue full, dropping frames")
        try:
            if os.getenv('ELA_AUDIO_DIAG'):
                print(f"[AudioManager] Opening InputStream dev={getattr(self,'input_device',None)} sr={self.sample_rate} ch={self.channels} block={self.chunk_size}")
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=input_callback,
                blocksize=self.chunk_size,
                dtype=np.float32,
                device=getattr(self, 'input_device', None)
            ) as stream:
                self.input_stream = stream
                logging.info(f"Input stream started on device {getattr(self,'input_device',None)}")
                while self.running and self.recording:
                    # Inactivity detection: after 3s if level remains extremely low
                    if (time.time() - start_time) > 3 and self.current_input_level < 1e-5 and not inactivity_logged:
                        logging.error("No measurable audio input detected from selected device (level < 1e-5). Check mic selection & permissions.")
                        inactivity_logged = True
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
