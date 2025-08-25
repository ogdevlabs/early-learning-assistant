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
        # Configure logging first
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
        logging.info("Initializing FaceDetectionApp...")

        self.cap = cv2.VideoCapture(0)
        self.face_detector = FaceDetector()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.latest_frame = None
        self.latest_facial_points = {}

        # Audio components (secondary, independent)
        self.enable_audio = enable_audio
        self.audio_manager = None
        self.audio_feedback = None
        self.voice_recognition = None

        # Initialize audio in background thread so it doesn't block video
        if self.enable_audio:
            audio_thread = threading.Thread(target=self._initialize_audio_async, daemon=True)
            audio_thread.start()

        logging.info("FaceDetectionApp initialization complete")

    def _initialize_audio_async(self):
        """Initialize audio system asynchronously in background thread"""
        try:
            logging.info("Starting audio initialization in background...")
            self._initialize_audio()
        except Exception as e:
            logging.error(f"Background audio initialization failed: {e}")
            self.enable_audio = False

    def _initialize_audio(self):
        """Initialize audio system in independent thread"""
        try:
            logging.info("Initializing audio system...")

            # Initialize audio manager
            self.audio_manager = AudioManager()
            if not self.audio_manager.enable_microphone():
                raise Exception("Failed to enable microphone")
            logging.info("Audio manager initialized")

            # Initialize audio feedback (TTS) with timeout
            self.audio_feedback = AudioFeedback()
            if not self.audio_feedback.start():
                raise Exception("Failed to start audio feedback")
            logging.info("Audio feedback started")

            # Initialize voice recognition
            self.voice_recognition = VoiceRecognitionThread(
                self.audio_manager,
                command_callback=self._handle_voice_command
            )
            logging.info("Voice recognition initialized")

            # Start audio recording with timeout
            if not self.audio_manager.start_recording():
                raise Exception("Failed to start recording")
            logging.info("Audio recording started")

            # Start voice recognition
            self.voice_recognition.start()
            logging.info("Voice recognition thread started")

            # Welcome message (non-blocking)
            try:
                self.audio_feedback.speak("Face and hand detection system ready")
            except:
                pass  # Don't fail if TTS has issues

            logging.info("Audio system fully initialized")

            # Start audio testing sequence after 3 seconds
            threading.Timer(3.0, self._start_audio_test).start()

        except Exception as e:
            logging.error(f"Failed to initialize audio: {e}")
            self.enable_audio = False
            # Cleanup partial initialization
            try:
                if self.audio_feedback:
                    self.audio_feedback.stop()
                if self.audio_manager:
                    self.audio_manager.shutdown()
            except:
                pass

    def _start_audio_test(self):
        """Start the audio testing sequence"""
        try:
            logging.info("Starting audio test sequence...")

            # Announce system initialization
            self.audio_feedback.speak("System initialized")

            # Start hand detection monitoring instead of capture and repeat mode
            threading.Timer(2.0, self._start_hand_detection_monitoring).start()

        except Exception as e:
            logging.error(f"Error starting audio test: {e}")

    def _start_hand_detection_monitoring(self):
        """Start monitoring for hand detection after system initialization"""
        try:
            logging.info("Starting hand detection monitoring...")
            self.monitoring_hands = True
            self.hands_detected = False
            self.face_detected = False
            self.hand_request_sent = False
            self.hand_request_reminder_count = 0

            # Initial prompt to put hands in frame
            self.audio_feedback.speak("Please put your hands in the camera view")

            # Start monitoring thread
            threading.Thread(target=self._monitor_detection_status, daemon=True).start()

        except Exception as e:
            logging.error(f"Error starting hand detection monitoring: {e}")

    def _monitor_detection_status(self):
        """Monitor face and hand detection status"""
        last_request_time = time.time()
        reminder_interval = 5  # Seconds between reminders

        while self.monitoring_hands:
            try:
                current_time = time.time()

                # Check for hand landmarks in the latest frame
                current_hands = self._check_current_hands()

                if current_hands and not self.hands_detected:
                    # Hands are detected for the first time
                    self.hands_detected = True
                    logging.info("Hand landmarks identified!")
                    self.audio_feedback.speak("Excellent! Hands detected successfully.")

                    # If we want to continue to another mode after hands are detected
                    threading.Timer(2.0, self._start_capture_repeat_mode).start()
                    self.monitoring_hands = False  # Stop monitoring

                elif not current_hands and not self.hands_detected and (current_time - last_request_time) > reminder_interval:
                    # No hands detected yet, provide reminder
                    self.hand_request_reminder_count += 1
                    last_request_time = current_time

                    # Vary the message based on how many reminders we've given
                    if self.hand_request_reminder_count == 1:
                        self.audio_feedback.speak("I don't see your hands yet. Please place both hands in the camera view.")
                    elif self.hand_request_reminder_count == 2:
                        self.audio_feedback.speak("Please position your hands so they are clearly visible to the camera.")
                    elif self.hand_request_reminder_count == 3:
                        self.audio_feedback.speak("Make sure your hands are in the frame and well lit so I can detect them.")
                    else:
                        self.audio_feedback.speak("Still waiting for hands to be detected. Please put your hands in the frame.")

                    logging.info(f"Requesting user to show hands (reminder {self.hand_request_reminder_count})")

                time.sleep(0.5)  # Check every 500ms

            except Exception as e:
                logging.error(f"Error in detection monitoring: {e}")
                time.sleep(1)

    def _check_current_hands(self):
        """Check if hands are currently detected in the latest frame"""
        try:
            if self.latest_frame is None:
                return False

            # Use MediaPipe to check for hands in current frame
            import mediapipe as mp
            mp_hands = mp.solutions.hands

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:
                image_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    logging.info(f"Found {len(results.multi_hand_landmarks)} hand(s) with landmarks")
                    return True

            return False

        except Exception as e:
            logging.error(f"Error checking hands: {e}")
            return False

    def _start_capture_repeat_mode(self):
        """Start capture and repeat mode"""
        try:
            logging.info("Starting capture and repeat mode...")
            self.capture_repeat_active = True

            # Announce mode
            self.audio_feedback.speak("Capture and repeat mode activated. Say something and I will repeat it. Say stop or quit to exit.")

        except Exception as e:
            logging.error(f"Error starting capture repeat mode: {e}")

    def _handle_voice_command(self, command):
        """Handle voice commands for face and hand detection"""
        logging.info(f"Voice command received: {command}")

        # Check if we're in capture and repeat mode
        if hasattr(self, 'capture_repeat_active') and self.capture_repeat_active:
            # Check for exit commands
            if "stop" in command.lower() or "quit" in command.lower():
                self.capture_repeat_active = False
                self.audio_feedback.speak("Capture and repeat mode deactivated. Returning to normal mode.")
                logging.info("Exiting capture and repeat mode")
                return

            # Repeat what was said
            logging.info(f"Repeating captured audio: {command}")
            self.audio_feedback.speak(f"You said: {command}")
            return

        # Normal command handling
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
            logging.info(f"Unhandled voice command: {command}")

    def get_latest_frame(self):
        return self.latest_frame

    def get_latest_facial_points(self):
        return self.latest_facial_points

    def run(self):
        if not self.cap.isOpened():
            logging.error('Could not open webcam.')
            return

        # Start the hand detection monitoring immediately when the application runs
        threading.Thread(target=self._start_hand_detection_monitoring, daemon=True).start()

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
                        logging.warning('Failed to read frame from webcam.')
                        break
                    self.latest_frame = frame
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            features = self.face_detector.detect(
                                face_landmarks,
                                image_width=frame.shape[1],
                                image_height=frame.shape[0]
                            )
                            self.latest_facial_points = self._extract_facial_points(features)

                            # Draw facial feature markers
                            for part, points in features.items():
                                if points is None:
                                    continue

                                # Handle nose as a tuple (x, y)
                                if part == 'nose' and isinstance(points, tuple) and len(points) == 2:
                                    cv2.circle(frame, (int(points[0]), int(points[1])), 4, (0, 0, 255), -1)  # Red dot for nose
                                    continue

                                if isinstance(points, dict):
                                    for k, v in points.items():
                                        if v is None:
                                            continue
                                        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                                            color = (0, 0, 255) if part == 'nose' else (255, 255, 255)  # Red for nose, white for others
                                            cv2.circle(frame, (int(v[0]), int(v[1])), 2, color, -1)
                                        elif isinstance(v, list):
                                            for pt in v:
                                                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                                                    color = (0, 0, 255) if part == 'nose' else (255, 255, 255)
                                                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, color, -1)
                                elif isinstance(points, list):
                                    for v in points:
                                        if isinstance(v, (list, tuple)) and len(v) == 2:
                                            color = (0, 0, 255) if part == 'nose' else (255, 255, 255)
                                            cv2.circle(frame, (int(v[0]), int(v[1])), 2, color, -1)

                    # Process hand detection and draw markers on the same frame
                    self._draw_hand_markers(frame, hands, image_rgb)

                    cv2.imshow('Face Detection', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q') or key == ord('Q'):  # ESC, q, or Q key
                        break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def _extract_facial_points(self, features):
        # Extracts key facial points from features dict for hand proximity logic
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
        """Draw hand landmarks with white dots and blue index finger markers"""
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                # Draw all hand landmarks
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    if i == 8:  # Index finger tip
                        cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)  # Blue for index finger
                    else:
                        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)  # White for other landmarks

    def shutdown_audio(self):
        """Cleanup audio resources"""
        if self.enable_audio:
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
                logging.error(f"Error during audio shutdown: {e}")

    def __del__(self):
        """Destructor to ensure audio cleanup"""
        self.shutdown_audio()
