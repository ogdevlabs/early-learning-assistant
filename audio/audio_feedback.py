import threading
import logging
import queue
import time
import subprocess
from typing import Optional, Dict, List, Tuple

class AudioFeedback:
    """
    Advanced text-to-speech system with improved threading and audio integration.
    Provides fluent speech output independent of main application threads.
    """

    def __init__(self, voice_rate: int = 150, voice_volume: float = 0.8, voice_id: Optional[int] = None):
        """
        Initialize AudioFeedback system.

        Args:
            voice_rate (int): Speech rate (words per minute)
            voice_volume (float): Speech volume (0.0 to 1.0)
            voice_id (int, optional): Voice ID to use
        """
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume
        self.voice_id = voice_id
        self.tts_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.tts_thread: Optional[threading.Thread] = None
        self.running = False

        # Speech control
        self.speaking = False
        self.muted = False

        # Enhanced predefined responses for facial detection and hand tracking
        self.responses = {
            # Greeting and help
            'hello': "Hello! I'm your face and hand detection assistant.",
            'help': "Available commands: start detection, stop detection, show hands, hide hands, point at nose, eyes, mouth, or ears.",
            'status': "Face detection is active. Hand tracking is enabled.",

            # Detection control
            'start_detection': "Starting detection systems.",
            'stop_detection': "Stopping detection systems.",
            'show_hands': "Hand markers are now visible.",
            'hide_hands': "Hand markers are now hidden.",

            # Facial feature pointing instructions
            'point_nose': "Point your index finger at your nose and hold for 3 seconds.",
            'point_eyes': "Point your index finger at your eyes and hold for 3 seconds.",
            'point_mouth': "Point your index finger at your mouth and hold for 3 seconds.",
            'point_ears': "Point your index finger at your ears and hold for 3 seconds.",

            # Detection feedback
            'finger_detected': "Index finger detected at facial feature.",
            'finger_held': "Excellent! Finger held for required duration.",
            'face_detected': "Face detected successfully.",
            'no_face': "No face detected. Please position yourself in front of the camera.",
            'hands_detected': "Hands detected successfully.",
            'no_hands': "No hands detected. Please show your hands to the camera.",

            # System feedback
            'audio_enabled': "Audio feedback is now enabled.",
            'audio_disabled': "Audio feedback is now disabled.",
            'system_ready': "Face and hand detection system is ready.",
            'error': "An error occurred. Please try again."
        }

        logging.info("AudioFeedback initialized")

    def start(self) -> bool:
        """
        Start audio feedback thread.

        Returns:
            bool: True if started successfully
        """
        if self.running:
            logging.warning("Audio feedback already running")
            return True

        try:
            self.running = True
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            logging.info("Audio feedback thread started")
            return True

        except Exception as e:
            logging.error(f"Failed to start audio feedback: {e}")
            self.running = False
            return False

    def stop(self):
        """Stop audio feedback thread and cleanup."""
        self.running = False

        # Clear queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break

        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)

        logging.info("Audio feedback thread stopped")

    def _tts_worker(self):
        """Background thread for processing TTS queue and producing speech using macOS 'say' command."""
        while self.running:
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text:
                    logging.info(f"Speaking: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    try:
                        subprocess.run(["say", text], check=True)
                    except Exception as e:
                        logging.error(f"Failed to speak using 'say': {e}")
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS worker error: {e}")
                self.speaking = False
                time.sleep(0.1)
        logging.info("TTS worker thread ended")

    def speak(self, text: str, priority: bool = False) -> bool:
        """
        Add text to speech queue.

        Args:
            text (str): Text to speak
            priority (bool): If True, clear queue and speak immediately

        Returns:
            bool: True if queued successfully
        """
        if not self.running:
            logging.warning("Audio feedback not started")
            return False

        if not text or not text.strip():
            return False

        try:
            if priority:
                # Clear queue for priority messages
                while not self.tts_queue.empty():
                    try:
                        self.tts_queue.get_nowait()
                    except queue.Empty:
                        break

            self.tts_queue.put_nowait(text.strip())
            logging.info(f"Queued for speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            return True

        except queue.Full:
            logging.warning("TTS queue full, dropping message")
            return False

    def speak_response(self, response_key: str, priority: bool = False) -> bool:
        """
        Speak a predefined response.

        Args:
            response_key (str): Key for predefined response
            priority (bool): If True, speak with priority

        Returns:
            bool: True if response found and queued
        """
        if response_key in self.responses:
            return self.speak(self.responses[response_key], priority)
        else:
            logging.warning(f"Unknown response key: {response_key}")
            return False

    def announce_detection(self, detection_type: str, status: bool):
        """
        Announce detection status changes.

        Args:
            detection_type (str): Type of detection ('face', 'hands')
            status (bool): Detection status
        """
        if detection_type == "face":
            self.speak_response("face_detected" if status else "no_face")
        elif detection_type == "hands":
            self.speak_response("hands_detected" if status else "no_hands")

    def announce_pointing(self, facial_feature: str):
        """
        Announce finger pointing detection.

        Args:
            facial_feature (str): Name of facial feature
        """
        self.speak(f"Index finger pointing at {facial_feature}")

    def announce_pointing_held(self, facial_feature: str, duration: float):
        """
        Announce finger held at feature.

        Args:
            facial_feature (str): Name of facial feature
            duration (float): Duration in seconds
        """
        self.speak(f"Excellent! Finger held at {facial_feature} for {duration:.1f} seconds")

    def set_muted(self, muted: bool):
        """
        Mute or unmute audio feedback.

        Args:
            muted (bool): True to mute, False to unmute
        """
        self.muted = muted
        logging.info(f"Audio feedback {'muted' if muted else 'unmuted'}")

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.speaking

    def is_running(self) -> bool:
        """Check if audio feedback is running."""
        return self.running

    def get_available_voices(self) -> List[Tuple[int, str, str]]:
        """
        Get list of available TTS voices.

        Returns:
            List[Tuple[int, str, str]]: List of (index, name, id) tuples
        """
        # Not applicable with macOS 'say' command, but keeping for interface consistency
        return [(0, "Default", "")]

    def set_voice_properties(self, rate: Optional[int] = None, volume: Optional[float] = None):
        """
        Update voice properties.

        Args:
            rate (int, optional): New speech rate
            volume (float, optional): New volume level
        """
        # Not applicable with macOS 'say' command, but keeping for interface consistency
        if rate is not None:
            self.voice_rate = rate
        if volume is not None:
            self.voice_volume = max(0.0, min(1.0, volume))

        logging.info(f"Voice properties updated: rate={self.voice_rate}, volume={self.voice_volume}")


    def add_custom_response(self, key: str, text: str):
        """
        Add a custom response.

        Args:
            key (str): Response key
            text (str): Response text
        """
        self.responses[key] = text
        logging.info(f"Added custom response: {key}")

    def get_queue_size(self) -> int:
        """Get current TTS queue size."""
        return self.tts_queue.qsize()

    def clear_queue(self):
        """Clear the TTS queue."""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
        logging.info("TTS queue cleared")
