import os
import cv2
import mediapipe as mp
import logging
import threading
import time
from face_detection.detector import FaceDetector
from audio.audio_manager import AudioManager
from audio.audio_feedback import AudioFeedback
from audio.voice_recognition import VoiceRecognitionThread

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("ELA_LOG_LEVEL", "ERROR").upper()
if not hasattr(logging, LOG_LEVEL):
    LOG_LEVEL = "ERROR"
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

class FaceDetectionApp:
    """Face + Hand detection with optional audio (TTS + voice recognition).

    Public methods used externally:
      - initialize_audio(...)
      - run()
      - get_latest_frame()
      - get_latest_facial_points()
      - shutdown_audio()
    """

    def __init__(self, enable_audio: bool = True):
        self.enable_audio = enable_audio

        # Video / detection components
        self.cap = cv2.VideoCapture(0)
        self.face_detector = FaceDetector()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands

        # Frame state
        self.latest_frame = None
        self.latest_facial_points = {}

        # Audio components
        self.audio_manager = None
        self.audio_feedback = None
        self.voice_recognition = None
        self.audio_initialized = False

        # Hand detection / debouncing
        self.monitoring_hands = False
        self.capture_repeat_active = False
        self.frames_processed = 0
        self.current_hands_present = False
        self._hands_consecutive_present = 0
        self._hands_consecutive_absent = 0
        self.hands_detected = False
        self.hands_missing = True
        self.last_hands_seen_time = None
        self.last_request_time = None
        self.hand_request_reminder_count = 0

        # Tunables
        self.HAND_WARMUP_FRAMES = 10
        self.HAND_PRESENT_THRESHOLD = 8
        self.HAND_ABSENT_THRESHOLD = 25
        self.HAND_PROMPT_INTERVAL = 25  # seconds

    # ---------------- Audio Initialization ----------------
    def initialize_audio(self, input_device_id=None, output_device_id=None, test_interaction=True) -> bool:
        """Initialize audio stack (blocking). Returns True if audio active."""
        if not self.enable_audio:
            return False
        if self.audio_initialized:
            return True
        try:
            self._initialize_audio(input_device_id, output_device_id, test_interaction)
            self.audio_initialized = True
            return True
        except Exception as e:
            logger.error(f"Audio initialization failed: {e}")
            self._cleanup_partial_audio()
            self.enable_audio = False
            return False

    def _initialize_audio(self, input_device_id=None, output_device_id=None, test_interaction=True):
        self.audio_manager = AudioManager()
        if not self.audio_manager.enable_microphone(device_id=input_device_id):
            raise RuntimeError("Microphone enable failed")
        try:
            self.audio_manager.enable_speakers(device_id=output_device_id)
        except Exception:
            pass
        self.audio_feedback = AudioFeedback()
        if not self.audio_feedback.start():
            raise RuntimeError("TTS start failed")
        if not self.audio_manager.start_recording():
            raise RuntimeError("Recording start failed")
        self.voice_recognition = VoiceRecognitionThread(
            self.audio_manager,
            command_callback=self._handle_voice_command
        )
        self.voice_recognition.start()
        try:
            self.audio_feedback.speak("Audio system ready")
        except Exception:
            pass
        if test_interaction:
            self._basic_audio_self_test()
        threading.Timer(2.0, self._audio_initialized_announcement).start()

    def _audio_initialized_announcement(self):
        if self.audio_feedback:
            try:
                self.audio_feedback.speak("System initialized")
            except Exception:
                pass

    def _basic_audio_self_test(self, sample_time=0.5):
        start = time.time()
        while time.time() - start < sample_time:
            _ = self.audio_manager.get_audio_data() if self.audio_manager else None
            time.sleep(0.05)

    def _cleanup_partial_audio(self):
        try:
            if self.audio_feedback:
                self.audio_feedback.stop()
        except Exception:
            pass
        try:
            if self.audio_manager:
                self.audio_manager.stop_recording()
                self.audio_manager.shutdown()
        except Exception:
            pass
        self.audio_feedback = None
        self.audio_manager = None
        self.voice_recognition = None

    # ---------------- Hand Monitoring ----------------
    def _start_hand_detection_monitoring(self):
        if self.monitoring_hands:
            return
        self.monitoring_hands = True
        threading.Thread(target=self._monitor_detection_status, daemon=True).start()

    def _monitor_detection_status(self):
        while self.monitoring_hands:
            try:
                if self.frames_processed < self.HAND_WARMUP_FRAMES:
                    time.sleep(0.2)
                    continue
                now = time.time()
                if self.hands_missing and self.last_request_time is None:
                    self._speak_safe("Please put your hands in the camera view")
                    self.last_request_time = now
                elif self.hands_missing and self.last_request_time and (now - self.last_request_time) > self.HAND_PROMPT_INTERVAL:
                    self._speak_safe("Please put your hands in the camera view")
                    self.last_request_time = now
                if not self.hands_detected and self._hands_consecutive_present >= self.HAND_PRESENT_THRESHOLD:
                    self.hands_detected = True
                    self.hands_missing = False
                    self.last_hands_seen_time = now
                    self._speak_safe("Excellent! Hands detected successfully.")
                    threading.Timer(2.0, self._start_capture_repeat_mode).start()
                if self.hands_detected and self._hands_consecutive_absent >= self.HAND_ABSENT_THRESHOLD:
                    self.hands_detected = False
                    self.hands_missing = True
                    self.last_hands_seen_time = None
                    self.last_request_time = None
                time.sleep(0.2)
            except Exception:
                time.sleep(0.5)

    def _speak_safe(self, text: str):
        if self.audio_feedback:
            try:
                self.audio_feedback.speak(text)
            except Exception:
                pass

    # ---------------- Voice Commands ----------------
    def _start_capture_repeat_mode(self):
        if self.capture_repeat_active:
            return
        self.capture_repeat_active = True
        if self.audio_feedback:
            self.audio_feedback.speak(
                "Capture and repeat mode activated. Say something and I will repeat it. Say stop or quit to exit."
            )

    def _handle_voice_command(self, command: str):
        if self.capture_repeat_active:
            if any(x in command.lower() for x in ["stop", "quit"]):
                self.capture_repeat_active = False
                if self.audio_feedback:
                    self.audio_feedback.speak("Capture and repeat mode deactivated.")
                return
            if self.audio_feedback:
                self.audio_feedback.speak(f"You said: {command}")
            return
        if not self.audio_feedback:
            return
        mapping = {
            "hello": "hello",
            "help": "help",
            "status": "status",
            "show_hands": "Hand markers are visible",
            "hide_hands": "Hand markers are hidden"
        }
        if command in mapping:
            if command in ("hello", "help", "status"):
                self.audio_feedback.speak_response(command)
            else:
                self.audio_feedback.speak(mapping[command])
            return
        if "point" in command and "nose" in command:
            self.audio_feedback.speak("Point your index finger at your nose")
        elif "point" in command and "eyes" in command:
            self.audio_feedback.speak("Point your index finger at your eyes")
        elif "point" in command and "mouth" in command:
            self.audio_feedback.speak("Point your index finger at your mouth")
        elif "point" in command and "ears" in command:
            self.audio_feedback.speak("Point your index finger at your ears")
        else:
            self.audio_feedback.speak(f"You said: {command}")

    # ---------------- Accessors ----------------
    def get_latest_frame(self):
        return self.latest_frame

    def get_latest_facial_points(self):
        return self.latest_facial_points

    # ---------------- Main Loop ----------------
    def run(self):
        if not self.cap.isOpened():
            logger.error("Could not open webcam.")
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
                        break
                    self.latest_frame = frame
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_results = face_mesh.process(rgb)
                    if face_results.multi_face_landmarks:
                        for fl in face_results.multi_face_landmarks:
                            features = self.face_detector.detect(
                                fl, frame.shape[1], frame.shape[0]
                            )
                            self.latest_facial_points = self._extract_facial_points(features)
                            self._draw_facial_markers(frame, features)
                    hand_results = hands.process(rgb)
                    present = bool(hand_results.multi_hand_landmarks)
                    if present:
                        self._hands_consecutive_present += 1
                        self._hands_consecutive_absent = 0
                    else:
                        self._hands_consecutive_absent += 1
                        self._hands_consecutive_present = 0
                    self.current_hands_present = present
                    if hand_results.multi_hand_landmarks:
                        self._draw_hand_markers(frame, hand_results)
                    self.frames_processed += 1
                    cv2.imshow('Face Detection', frame)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord('q'), ord('Q')):
                        break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    # ---------------- Drawing Helpers ----------------
    def _draw_facial_markers(self, frame, features):
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
                        for sub in v:
                            if isinstance(sub, (list, tuple)) and len(sub) == 2:
                                cv2.circle(frame, (int(sub[0]), int(sub[1])), 2, (255,255,255), -1)
            elif isinstance(pts, list):
                for v in pts:
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        cv2.circle(frame, (int(v[0]), int(v[1])), 2, (255,255,255), -1)

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

    def _draw_hand_markers(self, frame, hand_results):
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for i, lm in enumerate(hand_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                if i == 8:
                    cv2.circle(frame, (x, y), 6, (255,0,0), -1)
                else:
                    cv2.circle(frame, (x, y), 2, (255,255,255), -1)

    # ---------------- Shutdown ----------------
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
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown_audio()
        except Exception:
            pass
