import cv2
import mediapipe as mp
import logging
import threading
import time
from face_detection.detector import FaceDetector
from audio.audio_manager import AudioManager
from audio.audio_feedback import AudioFeedback
from audio.voice_recognition import VoiceRecognitionThread

class FaceDetectionApp:
    def __init__(self, enable_audio=True):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
        logging.info("Initializing FaceDetectionApp core (video + detectors)...")

        self.cap = cv2.VideoCapture(0)
        self.face_detector = FaceDetector()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.latest_frame = None
        self.latest_facial_points = {}

        # Audio related
        self.enable_audio = enable_audio
        self.audio_manager = None
        self.audio_feedback = None
        self.voice_recognition = None
        self.audio_initialized = False

        # State flags
        self.monitoring_hands = False
        self.capture_repeat_active = False

        # Hand detection state
        self.frames_processed = 0
        self.current_hands_present = False
        self._hands_consecutive_present = 0
        self._hands_consecutive_absent = 0
        # Tuning parameters
        self.HAND_WARMUP_FRAMES = 15        # frames before starting hand evaluation
        self.HAND_PRESENT_THRESHOLD = 10    # consecutive frames required to confirm presence
        self.HAND_ABSENT_THRESHOLD = 30     # consecutive frames to confirm absence after presence
        self.HAND_PROMPT_INTERVAL = 25      # seconds between repeated prompts

        logging.info("FaceDetectionApp core initialized (audio deferred)")

    # --------------------------- Audio Initialization ---------------------------
    def initialize_audio(self, input_device_id=None, output_device_id=None, test_interaction=True):
        """Blocking audio & microphone (and speakers) setup.
        Args:
            input_device_id (int|None): Specific microphone device index.
            output_device_id (int|None): Specific speaker device index.
            test_interaction (bool): If True run a brief IO self-test.
        """
        if not self.enable_audio:
            logging.info("Audio disabled; skipping initialization")
            return False
        if self.audio_initialized:
            logging.info("Audio already initialized")
            return True
        logging.info("Starting blocking audio & microphone setup...")
        try:
            self._initialize_audio(input_device_id, output_device_id, test_interaction=test_interaction)
            self.audio_initialized = True
            logging.info("Audio & microphone setup complete")
            return True
        except Exception as e:
            logging.error(f"Audio initialization failed: {e}")
            self.enable_audio = False
            self.audio_initialized = False
            raise

    def _initialize_audio(self, input_device_id=None, output_device_id=None, test_interaction=True):
        logging.info("Initializing audio system...")
        self.audio_manager = AudioManager()
        if not self.audio_manager.enable_microphone(device_id=input_device_id):
            raise Exception("Failed to enable microphone")
        logging.info("Microphone enabled")

        # Speakers (optional path, mainly for config/logging symmetry)
        try:
            self.audio_manager.enable_speakers(device_id=output_device_id)
        except Exception as e:
            logging.warning(f"Speakers enable attempt failed (continuing - pyttsx3 handles output): {e}")

        self.audio_feedback = AudioFeedback()
        if not self.audio_feedback.start():
            raise Exception("Failed to start audio feedback")
        logging.info("Audio feedback started")

        self.voice_recognition = VoiceRecognitionThread(
            self.audio_manager,
            command_callback=self._handle_voice_command
        )
        logging.info("Voice recognition thread created")

        if not self.audio_manager.start_recording():
            raise Exception("Failed to start recording")
        logging.info("Audio recording started")

        self.voice_recognition.start()
        logging.info("Voice recognition thread started")

        # Welcome
        try:
            self.audio_feedback.speak("Audio system ready")
        except Exception as e:
            logging.warning(f"Welcome TTS failed: {e}")

        if test_interaction:
            self._basic_audio_self_test()

        threading.Timer(2.0, self._audio_initialized_announcement).start()

    def _audio_initialized_announcement(self):
        try:
            if self.audio_feedback:
                self.audio_feedback.speak("System initialized")
        except Exception as e:
            logging.error(f"Audio test announcement failed: {e}")

    def _basic_audio_self_test(self, sample_time=1.0, min_level=0.0005):
        """Collect short audio sample and log level; warn if very low."""
        start = time.time()
        levels = []
        while time.time() - start < sample_time:
            data = self.audio_manager.get_audio_data()
            if data is not None:
                try:
                    import numpy as np
                    levels.append(float(abs(data).mean()))
                except Exception:
                    pass
            time.sleep(0.05)
        if levels:
            avg = sum(levels)/len(levels)
            logging.info(f"Audio input level average: {avg:.6f}")
            if avg < min_level:
                logging.warning("Detected very low microphone input level; check mic or choose another device.")
        else:
            logging.warning("No audio samples captured during self-test.")

    # --------------------------- Monitoring Threads ---------------------------
    def _start_hand_detection_monitoring(self):
        if self.monitoring_hands:
            return
        try:
            logging.info("Starting hand detection monitoring thread (debounced)...")
            self.monitoring_hands = True
            self.hands_detected = False
            self.hands_missing = True
            self.last_hands_seen_time = None
            self.last_request_time = None  # set after first prompt actually spoken
            self.hand_request_reminder_count = 0
            threading.Thread(target=self._monitor_detection_status, daemon=True).start()
        except Exception as e:
            logging.error(f"Failed to start monitoring thread: {e}")

    def _monitor_detection_status(self):
        while self.monitoring_hands:
            try:
                # Wait for warmup frames so we don't speak too early
                if self.frames_processed < self.HAND_WARMUP_FRAMES:
                    time.sleep(0.2)
                    continue

                # Decide on prompt if still missing and enough time passed
                now = time.time()
                if self.hands_missing and self.last_request_time is None:
                    # First prompt after warmup
                    self._speak_safe("Please put your hands in the camera view")
                    self.last_request_time = now
                elif self.hands_missing and self.last_request_time and (now - self.last_request_time) > self.HAND_PROMPT_INTERVAL:
                    self.hand_request_reminder_count += 1
                    self._speak_safe("Please put your hands in the camera view")
                    self.last_request_time = now

                # Debounced detection logic based on consecutive frame counters (updated in main loop)
                if not self.hands_detected and self._hands_consecutive_present >= self.HAND_PRESENT_THRESHOLD:
                    self.hands_detected = True
                    self.hands_missing = False
                    self.last_hands_seen_time = now
                    self._speak_safe("Excellent! Hands detected successfully.")
                    # Enter capture & repeat mode soon after confirmation
                    threading.Timer(2.0, self._start_capture_repeat_mode).start()

                if self.hands_detected:
                    if self._hands_consecutive_absent >= self.HAND_ABSENT_THRESHOLD:
                        # Hands disappeared
                        self.hands_detected = False
                        self.hands_missing = True
                        self.last_hands_seen_time = None
                        self.last_request_time = None  # reset so prompt happens again
                        logging.info("Hands lost (debounced) – returning to prompt cycle")

                time.sleep(0.2)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(0.5)

    def _speak_safe(self, text):
        if self.audio_feedback:
            try:
                self.audio_feedback.speak(text)
            except Exception as e:
                logging.error(f"Audio feedback failed: {e}")

    def _check_current_hands(self):
        """Legacy method retained (now just returns debounced state)."""
        if self.frames_processed < self.HAND_WARMUP_FRAMES:
            return False
        return self.current_hands_present

    # --------------------------- Modes & Commands ---------------------------
    def _start_capture_repeat_mode(self):
        if self.capture_repeat_active:
            return
        try:
            logging.info("Entering capture & repeat mode")
            self.capture_repeat_active = True
            if self.audio_feedback:
                self.audio_feedback.speak("Capture and repeat mode activated. Say something and I will repeat it. Say stop or quit to exit.")
        except Exception as e:
            logging.error(f"Failed to start capture repeat mode: {e}")

    def _handle_voice_command(self, command):
        logging.info(f"Voice command: {command}")
        if self.capture_repeat_active:
            if any(x in command.lower() for x in ["stop", "quit"]):
                self.capture_repeat_active = False
                if self.audio_feedback:
                    self.audio_feedback.speak("Capture and repeat mode deactivated. Returning to normal mode.")
                return
            if self.audio_feedback:
                self.audio_feedback.speak(f"You said: {command}")
            return
        if not self.audio_feedback:
            return
        if command == "hello":
            self.audio_feedback.speak_response("hello")
        elif command == "help":
            self.audio_feedback.speak_response("help")
        elif command == "status":
            self.audio_feedback.speak_response("status")
        elif command == "show_hands":
            self.audio_feedback.speak("Hand markers are visible")
        elif command == "hide_hands":
            self.audio_feedback.speak("Hand markers are hidden")
        elif "point" in command and "nose" in command:
            self.audio_feedback.speak("Point your index finger at your nose")
        elif "point" in command and "eyes" in command:
            self.audio_feedback.speak("Point your index finger at your eyes")
        elif "point" in command and "mouth" in command:
            self.audio_feedback.speak("Point your index finger at your mouth")
        elif "point" in command and "ears" in command:
            self.audio_feedback.speak("Point your index finger at your ears")
        else:
            logging.info(f"Unhandled command: {command}")

    # --------------------------- Frame Accessors ---------------------------
    def get_latest_frame(self):
        return self.latest_frame

    def get_latest_facial_points(self):
        return self.latest_facial_points

    # --------------------------- Main Loop ---------------------------
    def run(self):
        if not self.cap.isOpened():
            logging.error("Could not open webcam.")
            return
        self._start_hand_detection_monitoring()
        try:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh, self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        logging.warning("Failed to read frame from webcam.")
                        break
                    self.latest_frame = frame
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    face_results = face_mesh.process(image_rgb)
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            features = self.face_detector.detect(
                                face_landmarks,
                                image_width=frame.shape[1],
                                image_height=frame.shape[0]
                            )
                            self.latest_facial_points = self._extract_facial_points(features)
                            for part, pts in features.items():
                                if pts is None:
                                    continue
                                if part == 'nose' and isinstance(pts, tuple) and len(pts) == 2:
                                    cv2.circle(frame, (int(pts[0]), int(pts[1])), 4, (0,0,255), -1)
                                    continue
                                if isinstance(pts, dict):
                                    for v in pts.values():
                                        if isinstance(v, (list, tuple)) and len(v) == 2:
                                            cv2.circle(frame, (int(v[0]), int(v[1])), 2, (255,255,255), -1)
                                        elif isinstance(v, list):
                                            for pt in v:
                                                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                                                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255,255,255), -1)
                                elif isinstance(pts, list):
                                    for v in pts:
                                        if isinstance(v, (list, tuple)) and len(v) == 2:
                                            cv2.circle(frame, (int(v[0]), int(v[1])), 2, (255,255,255), -1)

                    # Single hand processing per frame (no duplicate processing in monitoring thread)
                    hand_results = hands.process(image_rgb)
                    present_now = bool(hand_results.multi_hand_landmarks)
                    # Update consecutive counters
                    if present_now:
                        self._hands_consecutive_present += 1
                        self._hands_consecutive_absent = 0
                    else:
                        self._hands_consecutive_absent += 1
                        self._hands_consecutive_present = 0
                    self.current_hands_present = present_now

                    # Draw hand markers
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            h, w, _ = frame.shape
                            for i, landmark in enumerate(hand_landmarks.landmark):
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                if i == 8:
                                    cv2.circle(frame, (x, y), 6, (255,0,0), -1)
                                else:
                                    cv2.circle(frame, (x, y), 2, (255,255,255), -1)

                    self.frames_processed += 1

                    cv2.imshow('Face Detection', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def _extract_facial_points(self, features):
        points = {}
        for part, value in features.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        points[f"{part}_{k}"] = v
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                points[part] = value
        return points

    def _draw_hand_markers(self, frame, hands, image_rgb):
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    if i == 8:
                        cv2.circle(frame, (x, y), 6, (255,0,0), -1)
                    else:
                        cv2.circle(frame, (x, y), 2, (255,255,255), -1)

    # --------------------------- Shutdown ---------------------------
    def shutdown_audio(self):
        if not self.enable_audio:
            return
        try:
            if self.voice_recognition:
                self.voice_recognition.stop()
                self.voice_recognition.join(timeout=2.0)
            if self.audio_manager:
                self.audio_manager.stop_recording()
                self.audio_manager.shutdown()
            if self.audio_feedback:
                self.audio_feedback.stop()
            logging.info("Audio system shutdown complete")
        except Exception as e:
            logging.error(f"Audio shutdown error: {e}")

    def __del__(self):
        try:
            self.shutdown_audio()
        except:
            pass
