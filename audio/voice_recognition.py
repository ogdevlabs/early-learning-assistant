import threading
import logging
import speech_recognition as sr
import numpy as np
import time
import queue

class VoiceRecognitionThread(threading.Thread):
    def __init__(self, audio_manager, command_callback=None, recognition_language='en-US'):
        super().__init__()
        self.audio_manager = audio_manager
        self.command_callback = command_callback
        self.recognition_language = recognition_language
        self.running = False
        self.daemon = True

        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3

        # Voice activity detection
        self.silence_threshold = 0.01
        self.min_audio_length = 0.5  # seconds
        self.max_audio_length = 5.0  # seconds

        # Audio buffer for recognition
        self.audio_buffer = []
        self.buffer_duration = 0.0

        # Predefined voice commands
        self.voice_commands = {
            'start detection': self._handle_start_detection,
            'stop detection': self._handle_stop_detection,
            'show hands': self._handle_show_hands,
            'hide hands': self._handle_hide_hands,
            'point at nose': self._handle_point_nose,
            'point at eyes': self._handle_point_eyes,
            'point at mouth': self._handle_point_mouth,
            'point at ears': self._handle_point_ears,
            'hello': self._handle_hello,
            'help': self._handle_help,
            'status': self._handle_status
        }

        logging.info("VoiceRecognitionThread initialized")

    def run(self):
        """Main voice recognition loop"""
        logging.info("Voice recognition thread started")
        self.running = True

        while self.running:
            try:
                # Get audio data from audio manager
                audio_data = self.audio_manager.get_audio_data()

                if audio_data is not None:
                    self._process_audio_chunk(audio_data)
                else:
                    time.sleep(0.01)

            except Exception as e:
                logging.error(f"Voice recognition error: {e}")
                time.sleep(0.1)

        logging.info("Voice recognition thread stopped")

    def _process_audio_chunk(self, audio_data):
        """Process incoming audio chunk for voice activity detection"""
        # Calculate audio level
        audio_level = np.abs(audio_data).mean()

        # Voice activity detection
        if audio_level > self.silence_threshold:
            # Add to buffer
            self.audio_buffer.append(audio_data)
            self.buffer_duration += len(audio_data) / self.audio_manager.sample_rate

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
        """Process accumulated audio buffer for speech recognition"""
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
        finally:
            self._clear_buffer()

    def _recognize_speech(self, audio_data):
        """Recognize speech from audio data"""
        try:
            # Use Google Speech Recognition (free tier)
            text = self.recognizer.recognize_google(audio_data, language=self.recognition_language)
            text = text.lower().strip()

            logging.info(f"Recognized speech: '{text}'")

            # Check for voice commands
            self._process_voice_command(text)

        except sr.UnknownValueError:
            logging.debug("Could not understand audio")
        except sr.RequestError as e:
            logging.error(f"Speech recognition request error: {e}")
        except Exception as e:
            logging.error(f"Speech recognition error: {e}")

    def _process_voice_command(self, text):
        """Process recognized text for voice commands"""
        # Check exact matches first
        if text in self.voice_commands:
            self.voice_commands[text]()
            return

        # Check partial matches
        for command, handler in self.voice_commands.items():
            if command in text:
                handler()
                return

        # No command found, call generic callback
        if self.command_callback:
            self.command_callback(text)

    def _clear_buffer(self):
        """Clear audio buffer"""
        self.audio_buffer = []
        self.buffer_duration = 0.0

    # Voice command handlers
    def _handle_start_detection(self):
        logging.info("Voice command: Start detection")
        if self.command_callback:
            self.command_callback("start_detection")

    def _handle_stop_detection(self):
        logging.info("Voice command: Stop detection")
        if self.command_callback:
            self.command_callback("stop_detection")

    def _handle_show_hands(self):
        logging.info("Voice command: Show hands")
        if self.command_callback:
            self.command_callback("show_hands")

    def _handle_hide_hands(self):
        logging.info("Voice command: Hide hands")
        if self.command_callback:
            self.command_callback("hide_hands")

    def _handle_point_nose(self):
        logging.info("Voice command: Point at nose")
        if self.command_callback:
            self.command_callback("point_nose")

    def _handle_point_eyes(self):
        logging.info("Voice command: Point at eyes")
        if self.command_callback:
            self.command_callback("point_eyes")

    def _handle_point_mouth(self):
        logging.info("Voice command: Point at mouth")
        if self.command_callback:
            self.command_callback("point_mouth")

    def _handle_point_ears(self):
        logging.info("Voice command: Point at ears")
        if self.command_callback:
            self.command_callback("point_ears")

    def _handle_hello(self):
        logging.info("Voice command: Hello")
        if self.command_callback:
            self.command_callback("hello")

    def _handle_help(self):
        logging.info("Voice command: Help")
        if self.command_callback:
            self.command_callback("help")

    def _handle_status(self):
        logging.info("Voice command: Status")
        if self.command_callback:
            self.command_callback("status")

    def stop(self):
        """Stop voice recognition thread"""
        self.running = False
