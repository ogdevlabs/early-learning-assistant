import threading
import logging
import sounddevice as sd
import numpy as np
import queue
import time

class AudioManager:
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.recording = False
        self.microphone_enabled = False

        # Audio stream
        self.stream = None

        # Threading
        self.audio_thread = None
        self.running = False

        logging.info("AudioManager initialized")

    def enable_microphone(self):
        """Enable microphone and start audio capture"""
        try:
            # Check available audio devices
            devices = sd.query_devices()
            logging.info(f"Available audio devices: {len(devices)}")

            # Get default input device
            default_device = sd.query_devices(kind='input')
            logging.info(f"Default input device: {default_device['name']}")

            self.microphone_enabled = True
            logging.info("Microphone enabled successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to enable microphone: {e}")
            return False

    def start_recording(self):
        """Start audio recording in a separate thread"""
        if not self.microphone_enabled:
            logging.error("Microphone not enabled. Call enable_microphone() first.")
            return False

        if self.recording:
            logging.warning("Recording already in progress")
            return True

        try:
            self.running = True
            self.recording = True
            self.audio_thread = threading.Thread(target=self._audio_capture_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            logging.info("Audio recording started")
            return True

        except Exception as e:
            logging.error(f"Failed to start recording: {e}")
            self.recording = False
            self.running = False
            return False

    def stop_recording(self):
        """Stop audio recording"""
        self.running = False
        self.recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)

        logging.info("Audio recording stopped")

    def _audio_capture_loop(self):
        """Main audio capture loop running in separate thread"""
        def callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio callback status: {status}")

            # Put audio data in queue for processing
            audio_data = indata.copy()
            try:
                self.audio_queue.put_nowait(audio_data)
            except queue.Full:
                logging.warning("Audio queue full, dropping frames")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            ) as stream:
                self.stream = stream
                logging.info(f"Audio stream started: {self.sample_rate}Hz, {self.channels} channel(s)")

                while self.running:
                    time.sleep(0.1)

        except Exception as e:
            logging.error(f"Audio capture error: {e}")
        finally:
            self.recording = False
            logging.info("Audio capture loop ended")

    def get_audio_data(self):
        """Get audio data from queue (non-blocking)"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def is_recording(self):
        """Check if currently recording"""
        return self.recording

    def get_audio_level(self):
        """Get current audio input level (0.0 to 1.0)"""
        audio_data = self.get_audio_data()
        if audio_data is not None:
            return float(np.abs(audio_data).mean())
        return 0.0
