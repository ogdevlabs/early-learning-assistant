import threading
import logging
import pyttsx3
import queue
import time

class AudioFeedback:
    def __init__(self, voice_rate=150, voice_volume=0.8, voice_id=None):
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume
        self.voice_id = voice_id

        # Text-to-speech engine
        self.tts_engine = None
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.running = False

        # Initialize TTS engine
        self._initialize_tts()

        # Predefined responses
        self.responses = {
            'hello': "Hello! I'm ready to help with face and hand detection.",
            'help': "Available commands: start detection, stop detection, show hands, hide hands, point at nose, eyes, mouth, or ears.",
            'status': "Face detection is active. Hand tracking is enabled.",
            'start_detection': "Starting detection systems.",
            'stop_detection': "Stopping detection systems.",
            'show_hands': "Hand markers are now visible.",
            'hide_hands': "Hand markers are now hidden.",
            'point_nose': "Point your index finger at your nose.",
            'point_eyes': "Point your index finger at your eyes.",
            'point_mouth': "Point your index finger at your mouth.",
            'point_ears': "Point your index finger at your ears.",
            'finger_detected': "Index finger detected at facial feature.",
            'finger_held': "Good! Finger held for required duration.",
            'face_detected': "Face detected successfully.",
            'no_face': "No face detected. Please position yourself in front of the camera.",
            'hands_detected': "Hands detected successfully.",
            'no_hands': "No hands detected. Please show your hands to the camera.",
            'error': "An error occurred. Please try again."
        }

        logging.info("AudioFeedback initialized")

    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()

            # Set voice properties
            self.tts_engine.setProperty('rate', self.voice_rate)
            self.tts_engine.setProperty('volume', self.voice_volume)

            # Set voice if specified
            if self.voice_id:
                voices = self.tts_engine.getProperty('voices')
                if self.voice_id < len(voices):
                    self.tts_engine.setProperty('voice', voices[self.voice_id].id)

            logging.info("TTS engine initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize TTS engine: {e}")
            return False

    def start(self):
        """Start audio feedback thread"""
        if self.running:
            logging.warning("Audio feedback already running")
            return

        self.running = True
        self.tts_thread = threading.Thread(target=self._tts_worker)
        self.tts_thread.daemon = True
        self.tts_thread.start()
        logging.info("Audio feedback thread started")

    def stop(self):
        """Stop audio feedback thread"""
        self.running = False
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)
        logging.info("Audio feedback thread stopped")

    def _tts_worker(self):
        """TTS worker thread"""
        while self.running:
            try:
                # Get text from queue with timeout
                text = self.tts_queue.get(timeout=0.1)

                if text and self.tts_engine:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()

                self.tts_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS worker error: {e}")
                time.sleep(0.1)

    def speak(self, text):
        """Add text to speech queue"""
        if not self.running:
            logging.warning("Audio feedback not started")
            return

        try:
            self.tts_queue.put_nowait(text)
            logging.info(f"Queued for speech: '{text}'")
        except queue.Full:
            logging.warning("TTS queue full, dropping message")

    def speak_response(self, response_key):
        """Speak a predefined response"""
        if response_key in self.responses:
            self.speak(self.responses[response_key])
        else:
            logging.warning(f"Unknown response key: {response_key}")

    def announce_detection(self, detection_type, status):
        """Announce detection status"""
        if detection_type == "face" and status:
            self.speak_response("face_detected")
        elif detection_type == "face" and not status:
            self.speak_response("no_face")
        elif detection_type == "hands" and status:
            self.speak_response("hands_detected")
        elif detection_type == "hands" and not status:
            self.speak_response("no_hands")

    def announce_pointing(self, facial_feature):
        """Announce finger pointing detection"""
        self.speak(f"Index finger pointing at {facial_feature}")

    def announce_pointing_held(self, facial_feature, duration):
        """Announce finger held at feature"""
        self.speak(f"Good! Finger held at {facial_feature} for {duration} seconds")

    def get_available_voices(self):
        """Get list of available TTS voices"""
        if not self.tts_engine:
            return []

        try:
            voices = self.tts_engine.getProperty('voices')
            return [(i, voice.name, voice.id) for i, voice in enumerate(voices)]
        except Exception as e:
            logging.error(f"Error getting voices: {e}")
            return []
