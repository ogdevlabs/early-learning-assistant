"""
VoiceRecognitionThread diagnostic runner.
Recommended usage:
    python -m audio.voice_recognition
Or (for direct execution):
    python audio/voice_recognition.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import threading
import logging
import time
from typing import Optional, Callable, Dict, List

import speech_recognition as sr
import numpy as np

from audio.audio_manager import AudioManager

logger = logging.getLogger(__name__)

class VoiceRecognitionThread(threading.Thread):
    """Minimal, test-friendly voice recognition thread.

    Responsibilities:
      - Collect audio chunks from AudioManager (non-blocking queue polling)
      - Basic voice activity detection (mean absolute amplitude threshold)
      - Buffer speech, then send to Google SR when silence encountered
      - Map recognized phrases to command handlers or echo via callback
    """

    def __init__(
        self,
        audio_manager: AudioManager,
        command_callback: Optional[Callable[[str], None]] = None,
        recognition_language: str = 'en-US',
        confidence_threshold: float = 0.7,
        silence_threshold: float = 0.01,
        min_audio_length: float = 1.0,
        max_audio_length: float = 5.0,
        speech_timeout: float = 10.0,
    ):
        super().__init__(daemon=True)
        self.audio_manager = audio_manager
        self.command_callback = command_callback
        self.recognition_language = recognition_language
        self.confidence_threshold = confidence_threshold

        # VAD settings
        self.silence_threshold = silence_threshold
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.speech_timeout = speech_timeout

        # Runtime state
        self.running = False
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

        self.audio_buffer: List[np.ndarray] = []
        self.buffer_duration = 0.0
        self.last_speech_time = time.time()

        # Statistics
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0

        # Command registry
        self.voice_commands: Dict[str, Callable[[], None]] = {
            'hello': self._handle_hello,
            'help': self._handle_help,
            'status': self._handle_status,
            'statistics': self._handle_statistics,
            'show hands': self._handle_show_hands,
            'hide hands': self._handle_hide_hands,
            'enable hand tracking': self._handle_enable_hands,
            'disable hand tracking': self._handle_disable_hands,
            'mute': self._handle_mute,
            'unmute': self._handle_unmute,
        }

        self.command_aliases: Dict[str, str] = {
            'hi': 'hello',
            'hey': 'hello',
            'stats': 'statistics',
            'hands on': 'show hands',
            'hands off': 'hide hands',
        }

    # ---------------- Thread Loop ----------------
    def run(self):
        self.running = True
        logger.info("Voice recognition thread started")
        try:
            while self.running:
                audio_chunk = self.audio_manager.get_audio_data() if self.audio_manager else None
                if audio_chunk is not None:
                    self._process_audio_chunk(audio_chunk)
                else:
                    time.sleep(0.01)
                # Timeout flush/clear
                if time.time() - self.last_speech_time > self.speech_timeout:
                    self._handle_speech_timeout()
        except Exception as e:
            logger.error(f"Voice recognition loop error: {e}")
        finally:
            self.running = False
            logger.info("Voice recognition thread stopped")

    # ---------------- Processing ----------------
    def _process_audio_chunk(self, audio_data: np.ndarray):
        if audio_data.size == 0:
            return
        if audio_data.ndim > 1:
            audio_data = audio_data.reshape(-1)
        level = float(np.abs(audio_data).mean())
        # Voice active
        if level > self.silence_threshold:
            self.audio_buffer.append(audio_data)
            sr_val = getattr(self.audio_manager, 'sample_rate', 16000)
            self.buffer_duration += len(audio_data) / sr_val
            self.last_speech_time = time.time()
            if self.buffer_duration >= self.max_audio_length:
                self._process_speech_buffer()
        else:
            # Silence: finalize if enough speech
            if self.buffer_duration >= self.min_audio_length:
                self._process_speech_buffer()
            else:
                self._clear_buffer()

    def _process_speech_buffer(self):
        if not self.audio_buffer:
            return
        try:
            audio = np.concatenate(self.audio_buffer)
            sr_val = getattr(self.audio_manager, 'sample_rate', 16000)
            pcm16 = self._float_to_int16_bytes(audio)
            audio_data = sr.AudioData(pcm16, sr_val, 2)
            self._clear_buffer()
            try:
                text = self.recognizer.recognize_google(audio_data, language=self.recognition_language)
                if text:
                    self.total_recognitions += 1
                    self.successful_recognitions += 1
                    self._handle_recognized_text(text.strip().lower())
            except sr.UnknownValueError:
                self.total_recognitions += 1
                self.failed_recognitions += 1
            except sr.RequestError as e:
                logger.error(f"Speech service error: {e}")
                self.total_recognitions += 1
                self.failed_recognitions += 1
        except Exception as e:
            logger.error(f"Buffer processing error: {e}")
            self._clear_buffer()

    def _handle_recognized_text(self, text: str):
        # Alias mapping
        mapped = self.command_aliases.get(text, text)
        if mapped in self.voice_commands:
            try:
                self.voice_commands[mapped]()
            except Exception as e:
                logger.error(f"Command handler error for '{mapped}': {e}")
            return
        # Fallback to callback
        if self.command_callback:
            try:
                self.command_callback(text)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _float_to_int16_bytes(self, audio: np.ndarray) -> bytes:
        if audio.dtype not in (np.float32, np.float64):
            audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16).tobytes()

    def _handle_speech_timeout(self):
        self._clear_buffer()
        self.last_speech_time = time.time()

    def _clear_buffer(self):
        self.audio_buffer = []
        self.buffer_duration = 0.0

    # ---------------- Command Handlers ----------------
    def _handle_hello(self):
        self._callback_literal("hello")

    def _handle_help(self):
        self._callback_literal("help")

    def _handle_status(self):
        self._callback_literal("status")

    def _handle_statistics(self):
        success_rate = (self.successful_recognitions / max(1, self.total_recognitions)) * 100
        self._callback_literal(f"stats:success={self.successful_recognitions} total={self.total_recognitions} rate={success_rate:.1f}%")

    def _handle_show_hands(self):
        self._callback_literal("show_hands")

    def _handle_hide_hands(self):
        self._callback_literal("hide_hands")

    def _handle_enable_hands(self):
        self._callback_literal("enable_hand_tracking")

    def _handle_disable_hands(self):
        self._callback_literal("disable_hand_tracking")

    def _handle_mute(self):
        self._callback_literal("mute")

    def _handle_unmute(self):
        self._callback_literal("unmute")

    def _callback_literal(self, text: str):
        if self.command_callback:
            try:
                self.command_callback(text)
            except Exception as e:
                logger.error(f"Callback literal error: {e}")

    # ---------------- Public API ----------------
    def stop(self):
        self.running = False

    def get_statistics(self) -> Dict[str, int]:
        return {
            'total_recognitions': self.total_recognitions,
            'successful_recognitions': self.successful_recognitions,
            'failed_recognitions': self.failed_recognitions,
            'success_rate': (self.successful_recognitions / max(1, self.total_recognitions)) * 100
        }

    def update_sensitivity(self, energy_threshold: int):
        self.recognizer.energy_threshold = energy_threshold

    def add_custom_command(self, command: str, handler: Callable):
        self.voice_commands[command.lower()] = handler

    def remove_command(self, command: str):
        self.voice_commands.pop(command.lower(), None)

    def is_running(self) -> bool:
        return self.running

# Diagnostic usage when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    am = AudioManager()
    am.enable_microphone()
    am.start_recording()
    def printer(cmd):
        print(f"[DIAG] Command: {cmd}")
    vr = VoiceRecognitionThread(am, command_callback=printer)
    vr.start()
    print("VoiceRecognitionThread running. Ctrl+C to exit.")
    try:
        while vr.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        vr.stop()
        am.shutdown()
        print("Stopped.")
