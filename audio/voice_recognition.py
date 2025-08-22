import threading
import logging
import speech_recognition as sr
import numpy as np
import time
import queue
from typing import Optional, Callable, Dict, List
from audio.audio_manager import AudioManager

class VoiceRecognitionThread(threading.Thread):
    """
    Advanced voice recognition system with improved audio processing and command handling.
    Integrates with AudioManager for better audio input management.
    """

    def __init__(self,
                 audio_manager: AudioManager,
                 command_callback: Optional[Callable] = None,
                 recognition_language: str = 'en-US',
                 confidence_threshold: float = 0.7):
        """
        Initialize voice recognition thread.

        Args:
            audio_manager (AudioManager): Audio input manager
            command_callback (callable, optional): Callback for processed commands
            recognition_language (str): Language for speech recognition
            confidence_threshold (float): Minimum confidence for recognition
        """
        super().__init__()
        self.audio_manager = audio_manager
        self.command_callback = command_callback
        self.recognition_language = recognition_language
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.daemon = True

        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3

        # Voice activity detection parameters
        self.silence_threshold = 0.01
        self.min_audio_length = 1.0  # seconds
        self.max_audio_length = 5.0  # seconds
        self.speech_timeout = 10.0   # seconds

        # Audio processing
        self.audio_buffer = []
        self.buffer_duration = 0.0
        self.last_speech_time = time.time()

        # Recognition statistics
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0

        # Enhanced voice commands for facial detection and hand tracking
        self.voice_commands = {
            # System control
            'start detection': self._handle_start_detection,
            'stop detection': self._handle_stop_detection,
            'enable audio': self._handle_enable_audio,
            'disable audio': self._handle_disable_audio,
            'mute': self._handle_mute,
            'unmute': self._handle_unmute,

            # Hand detection control
            'show hands': self._handle_show_hands,
            'hide hands': self._handle_hide_hands,
            'enable hand tracking': self._handle_enable_hands,
            'disable hand tracking': self._handle_disable_hands,

            # Facial feature pointing commands
            'point at nose': self._handle_point_nose,
            'point at eyes': self._handle_point_eyes,
            'point at mouth': self._handle_point_mouth,
            'point at ears': self._handle_point_ears,
            'point nose': self._handle_point_nose,
            'point eyes': self._handle_point_eyes,
            'point mouth': self._handle_point_mouth,
            'point ears': self._handle_point_ears,

            # Information commands
            'hello': self._handle_hello,
            'help': self._handle_help,
            'status': self._handle_status,
            'statistics': self._handle_statistics,

            # Calibration commands
            'calibrate audio': self._handle_calibrate_audio,
            'adjust sensitivity': self._handle_adjust_sensitivity,
        }

        # Command aliases for better recognition
        self.command_aliases = {
            'hi': 'hello',
            'hey': 'hello',
            'start': 'start detection',
            'stop': 'stop detection',
            'begin': 'start detection',
            'end': 'stop detection',
            'hands on': 'show hands',
            'hands off': 'hide hands',
            'touch nose': 'point at nose',
            'touch eyes': 'point at eyes',
            'touch mouth': 'point at mouth',
            'touch ears': 'point at ears',
        }

        logging.info("VoiceRecognitionThread initialized")

    def run(self):
        """Main voice recognition loop with improved error handling."""
        logging.info("Voice recognition thread started")
        self.running = True

        try:
            while self.running:
                try:
                    # Get audio data from audio manager
                    audio_data = self.audio_manager.get_audio_data()

                    if audio_data is not None:
                        self._process_audio_chunk(audio_data)
                    else:
                        time.sleep(0.01)

                    # Check for speech timeout
                    if time.time() - self.last_speech_time > self.speech_timeout:
                        self._handle_speech_timeout()

                except Exception as e:
                    logging.error(f"Voice recognition loop error: {e}")
                    time.sleep(0.1)

        except KeyboardInterrupt:
            logging.info("Voice recognition interrupted by user")
        except Exception as e:
            logging.error(f"Voice recognition fatal error: {e}")
        finally:
            self.running = False
            logging.info("Voice recognition thread stopped")

    def _process_audio_chunk(self, audio_data: np.ndarray):
        """
        Process incoming audio chunk for voice activity detection.

        Args:
            audio_data (np.ndarray): Audio data chunk
        """
        # Calculate audio level
        audio_level = np.abs(audio_data).mean()

        # Voice activity detection
        if audio_level > self.silence_threshold:
            # Add to buffer
            self.audio_buffer.append(audio_data)
            self.buffer_duration += len(audio_data) / self.audio_manager.sample_rate
            self.last_speech_time = time.time()

            # Check if we have enough audio or max length reached
            if self.buffer_duration >= self.max_audio_length:
                self._process_speech_buffer()
        else:
            # Silence detected, process buffer if we have enough audio
            if self.buffer_duration >= self.min_audio_length:
                self._process_speech_buffer()
            else:
                # Clear short buffer
                self._clear_buffer()

    def _process_speech_buffer(self):
        """Process accumulated audio buffer for speech recognition."""
        if not self.audio_buffer:
            return

        try:
            # Combine audio chunks
            combined_audio = np.concatenate(self.audio_buffer)

            # Convert to speech_recognition format
            audio_data = sr.AudioData(
                frame_data=(combined_audio * 32767).astype(np.int16).tobytes(),
                sample_rate=self.audio_manager.sample_rate,
                sample_width=2
            )

            # Perform speech recognition
            self._recognize_speech(audio_data)

        except Exception as e:
            logging.error(f"Speech processing error: {e}")
            self.failed_recognitions += 1
        finally:
            self._clear_buffer()

    def _recognize_speech(self, audio_data: sr.AudioData):
        """
        Recognize speech from audio data with confidence checking.

        Args:
            audio_data (sr.AudioData): Audio data for recognition
        """
        try:
            self.total_recognitions += 1

            # Use Google Speech Recognition with confidence
            try:
                # Try Google recognition first
                text = self.recognizer.recognize_google(audio_data, language=self.recognition_language)
                text = text.lower().strip()

                logging.info(f"Recognized speech: '{text}'")
                self.successful_recognitions += 1

                # Process the recognized text
                self._process_voice_command(text)

            except sr.UnknownValueError:
                logging.debug("Could not understand audio")
                self.failed_recognitions += 1
            except sr.RequestError as e:
                logging.error(f"Speech recognition service error: {e}")
                self.failed_recognitions += 1
                # Could add fallback to offline recognition here

        except Exception as e:
            logging.error(f"Speech recognition error: {e}")
            self.failed_recognitions += 1

    def _process_voice_command(self, text: str):
        """
        Process recognized text for voice commands with improved matching.

        Args:
            text (str): Recognized text to process
        """
        # Normalize text
        text = text.lower().strip()

        # Check for exact matches first
        if text in self.voice_commands:
            self.voice_commands[text]()
            return

        # Check command aliases
        if text in self.command_aliases:
            alias_command = self.command_aliases[text]
            if alias_command in self.voice_commands:
                self.voice_commands[alias_command]()
                return

        # Check partial matches with improved scoring
        best_match = None
        best_score = 0

        for command in self.voice_commands.keys():
            if command in text:
                # Calculate match score based on command length and position
                score = len(command) / len(text)
                if text.startswith(command):
                    score *= 1.5  # Boost for commands at start

                if score > best_score and score >= 0.5:  # Minimum 50% match
                    best_score = score
                    best_match = command

        if best_match:
            logging.info(f"Matched command '{best_match}' with score {best_score:.2f}")
            self.voice_commands[best_match]()
            return

        # No command found, call generic callback
        if self.command_callback:
            self.command_callback(text)
        else:
            logging.info(f"Unhandled voice input: '{text}'")

    def _handle_speech_timeout(self):
        """Handle speech timeout (no voice activity for extended period)."""
        if self.audio_buffer:
            logging.debug("Speech timeout - clearing buffer")
            self._clear_buffer()
        self.last_speech_time = time.time()

    def _clear_buffer(self):
        """Clear audio buffer and reset duration."""
        self.audio_buffer = []
        self.buffer_duration = 0.0

    # Enhanced Voice command handlers
    def _handle_start_detection(self):
        """Handle start detection command."""
        logging.info("Voice command: Start detection")
        if self.command_callback:
            self.command_callback("start_detection")

    def _handle_stop_detection(self):
        """Handle stop detection command."""
        logging.info("Voice command: Stop detection")
        if self.command_callback:
            self.command_callback("stop_detection")

    def _handle_enable_audio(self):
        """Handle enable audio command."""
        logging.info("Voice command: Enable audio")
        if self.command_callback:
            self.command_callback("enable_audio")

    def _handle_disable_audio(self):
        """Handle disable audio command."""
        logging.info("Voice command: Disable audio")
        if self.command_callback:
            self.command_callback("disable_audio")

    def _handle_mute(self):
        """Handle mute command."""
        logging.info("Voice command: Mute")
        if self.command_callback:
            self.command_callback("mute")

    def _handle_unmute(self):
        """Handle unmute command."""
        logging.info("Voice command: Unmute")
        if self.command_callback:
            self.command_callback("unmute")

    def _handle_show_hands(self):
        """Handle show hands command."""
        logging.info("Voice command: Show hands")
        if self.command_callback:
            self.command_callback("show_hands")

    def _handle_hide_hands(self):
        """Handle hide hands command."""
        logging.info("Voice command: Hide hands")
        if self.command_callback:
            self.command_callback("hide_hands")

    def _handle_enable_hands(self):
        """Handle enable hand tracking command."""
        logging.info("Voice command: Enable hand tracking")
        if self.command_callback:
            self.command_callback("enable_hand_tracking")

    def _handle_disable_hands(self):
        """Handle disable hand tracking command."""
        logging.info("Voice command: Disable hand tracking")
        if self.command_callback:
            self.command_callback("disable_hand_tracking")

    def _handle_point_nose(self):
        """Handle point at nose command."""
        logging.info("Voice command: Point at nose")
        if self.command_callback:
            self.command_callback("point_nose")

    def _handle_point_eyes(self):
        """Handle point at eyes command."""
        logging.info("Voice command: Point at eyes")
        if self.command_callback:
            self.command_callback("point_eyes")

    def _handle_point_mouth(self):
        """Handle point at mouth command."""
        logging.info("Voice command: Point at mouth")
        if self.command_callback:
            self.command_callback("point_mouth")

    def _handle_point_ears(self):
        """Handle point at ears command."""
        logging.info("Voice command: Point at ears")
        if self.command_callback:
            self.command_callback("point_ears")

    def _handle_hello(self):
        """Handle hello command."""
        logging.info("Voice command: Hello")
        if self.command_callback:
            self.command_callback("hello")

    def _handle_help(self):
        """Handle help command."""
        logging.info("Voice command: Help")
        if self.command_callback:
            self.command_callback("help")

    def _handle_status(self):
        """Handle status command."""
        logging.info("Voice command: Status")
        if self.command_callback:
            self.command_callback("status")

    def _handle_statistics(self):
        """Handle statistics command."""
        logging.info("Voice command: Statistics")
        success_rate = (self.successful_recognitions / max(1, self.total_recognitions)) * 100
        stats_text = f"Recognition statistics: {self.successful_recognitions} successful out of {self.total_recognitions} attempts. Success rate: {success_rate:.1f}%"

        if self.command_callback:
            self.command_callback(f"stats:{stats_text}")

    def _handle_calibrate_audio(self):
        """Handle audio calibration command."""
        logging.info("Voice command: Calibrate audio")
        if self.command_callback:
            self.command_callback("calibrate_audio")

    def _handle_adjust_sensitivity(self):
        """Handle sensitivity adjustment command."""
        logging.info("Voice command: Adjust sensitivity")
        if self.command_callback:
            self.command_callback("adjust_sensitivity")

    # Public methods
    def stop(self):
        """Stop voice recognition thread."""
        self.running = False
        logging.info("Voice recognition stop requested")

    def get_statistics(self) -> Dict[str, int]:
        """
        Get recognition statistics.

        Returns:
            Dict[str, int]: Statistics dictionary
        """
        return {
            'total_recognitions': self.total_recognitions,
            'successful_recognitions': self.successful_recognitions,
            'failed_recognitions': self.failed_recognitions,
            'success_rate': (self.successful_recognitions / max(1, self.total_recognitions)) * 100
        }

    def update_sensitivity(self, energy_threshold: int):
        """
        Update recognition sensitivity.

        Args:
            energy_threshold (int): New energy threshold
        """
        self.recognizer.energy_threshold = energy_threshold
        logging.info(f"Updated recognition energy threshold to {energy_threshold}")

    def add_custom_command(self, command: str, handler: Callable):
        """
        Add a custom voice command.

        Args:
            command (str): Voice command text
            handler (callable): Command handler function
        """
        self.voice_commands[command.lower()] = handler
        logging.info(f"Added custom voice command: '{command}'")

    def remove_command(self, command: str):
        """
        Remove a voice command.

        Args:
            command (str): Voice command to remove
        """
        if command.lower() in self.voice_commands:
            del self.voice_commands[command.lower()]
            logging.info(f"Removed voice command: '{command}'")

    def is_running(self) -> bool:
        """Check if voice recognition is running."""
        return self.running
