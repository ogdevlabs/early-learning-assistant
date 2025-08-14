import cv2
import mediapipe as mp
import logging
from face_detection.detector import FaceDetector
from audio.audio_manager import AudioManager
from audio.voice_recognition import VoiceRecognitionThread
from audio.audio_feedback import AudioFeedback

class FaceDetectionApp:
    def __init__(self, enable_audio=True):
        # Configure logging first
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
        logging.info("Initializing FaceDetectionApp...")

        self.cap = cv2.VideoCapture(0)
        self.face_detector = FaceDetector()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.latest_frame = None
        self.latest_facial_points = {}

        # Audio management
        self.enable_audio = enable_audio
        self.audio_manager = None
        self.voice_recognition = None
        self.audio_feedback = None

        if self.enable_audio:
            self._initialize_audio()

        logging.info("FaceDetectionApp initialization complete")

    def _initialize_audio(self):
        """Initialize audio management system"""
        try:
            logging.info("Initializing audio management system...")

            # Initialize audio manager
            self.audio_manager = AudioManager()
            logging.info("Audio manager created")

            # Initialize audio feedback
            self.audio_feedback = AudioFeedback()
            self.audio_feedback.start()
            logging.info("Audio feedback started")

            # Initialize voice recognition with command callback
            self.voice_recognition = VoiceRecognitionThread(
                self.audio_manager,
                command_callback=self._handle_voice_command
            )
            logging.info("Voice recognition initialized")

            logging.info("Audio management system initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize audio system: {e}")
            logging.info("Continuing without audio features")
            self.enable_audio = False
            self.audio_manager = None
            self.audio_feedback = None
            self.voice_recognition = None
            return False

    def _handle_voice_command(self, command):
        """Handle voice commands from recognition system"""
        logging.info(f"Processing voice command: {command}")

        if command == "hello":
            self.audio_feedback.speak_response("hello")
        elif command == "help":
            self.audio_feedback.speak_response("help")
        elif command == "status":
            self.audio_feedback.speak_response("status")
        elif command == "show_hands":
            self.audio_feedback.speak_response("show_hands")
            # This could toggle hand marker visibility
        elif command == "hide_hands":
            self.audio_feedback.speak_response("hide_hands")
            # This could toggle hand marker visibility
        elif command in ["point_nose", "point_eyes", "point_mouth", "point_ears"]:
            feature = command.replace("point_", "")
            self.audio_feedback.speak(f"Point your index finger at your {feature}")
        else:
            # Generic command - could be used for future expansion
            logging.info(f"Unhandled voice command: {command}")

    def get_latest_frame(self):
        return self.latest_frame

    def get_latest_facial_points(self):
        return self.latest_facial_points

    def run(self):
        if not self.cap.isOpened():
            logging.error('Could not open webcam.')
            return

        # Start audio systems if enabled
        if self.enable_audio:
            self._start_audio_systems()

        try:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        logging.warning('Failed to read frame from webcam.')
                        break
                    self.latest_frame = frame
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)
                    if results.multi_face_landmarks:
                        # Announce face detection (only occasionally to avoid spam)
                        if hasattr(self, '_face_detected_announced'):
                            if not self._face_detected_announced and self.enable_audio:
                                self.audio_feedback.announce_detection("face", True)
                                self._face_detected_announced = True
                        else:
                            self._face_detected_announced = True

                        for face_landmarks in results.multi_face_landmarks:
                            features = self.face_detector.detect(
                                face_landmarks,
                                image_width=frame.shape[1],
                                image_height=frame.shape[0]
                            )
                            self.latest_facial_points = self._extract_facial_points(features)
                            for part, points in features.items():
                                if points is None:
                                    continue
                                if isinstance(points, dict):
                                    for k, v in points.items():
                                        if v is None:
                                            continue
                                        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                                            cv2.circle(frame, (int(v[0]), int(v[1])), 2, (255,255,255), -1)
                                        elif isinstance(v, list):
                                            for pt in v:
                                                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                                                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255,255,255), -1)
                                elif isinstance(points, list):
                                    for v in points:
                                        if isinstance(v, (list, tuple)) and len(v) == 2:
                                            cv2.circle(frame, (int(v[0]), int(v[1])), 2, (255,255,255), -1)
                                elif isinstance(points, (list, tuple)) and len(points) == 2 and all(isinstance(x, (int, float)) for x in points):
                                    cv2.circle(frame, (int(points[0]), int(points[1])), 2, (255,255,255), -1)
                            self.draw_nose(frame, face_landmarks, frame.shape[1], frame.shape[0])
                    else:
                        if hasattr(self, '_face_detected_announced'):
                            if self._face_detected_announced and self.enable_audio:
                                self.audio_feedback.announce_detection("face", False)
                                self._face_detected_announced = False
                        else:
                            self._face_detected_announced = False
                        logging.info('No face detected in frame.')

                    import time
                    time.sleep(0.05)
                    cv2.imshow('Face Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self._stop_audio_systems()
            self.cap.release()
            cv2.destroyAllWindows()

    def _start_audio_systems(self):
        """Start audio capture and voice recognition"""
        if not self.enable_audio:
            return

        try:
            # Enable microphone
            if self.audio_manager.enable_microphone():
                # Start audio recording
                if self.audio_manager.start_recording():
                    # Start voice recognition
                    self.voice_recognition.start()
                    logging.info("Audio systems started successfully")

                    # Welcome message
                    if self.audio_feedback:
                        self.audio_feedback.speak_response("hello")
                else:
                    logging.error("Failed to start audio recording")
            else:
                logging.error("Failed to enable microphone")

        except Exception as e:
            logging.error(f"Failed to start audio systems: {e}")

    def _stop_audio_systems(self):
        """Stop all audio systems"""
        if not self.enable_audio:
            return

        try:
            # Stop voice recognition
            if self.voice_recognition:
                self.voice_recognition.stop()

            # Stop audio recording
            if self.audio_manager:
                self.audio_manager.stop_recording()

            # Stop audio feedback
            if self.audio_feedback:
                self.audio_feedback.stop()

            logging.info("Audio systems stopped")

        except Exception as e:
            logging.error(f"Error stopping audio systems: {e}")

    def get_latest_frame(self):
        return self.latest_frame

    def get_latest_facial_points(self):
        return self.latest_facial_points

    def draw_nose(self, frame, face_landmarks, image_width, image_height):
        """
        Draws a red dot on the nose tip if detected.
        """
        from face_detection.nose import NoseDetector
        nose_detector = NoseDetector()
        nose_point = nose_detector.detect(face_landmarks, image_width, image_height)
        if nose_point is not None:
            # Ensure the point is within frame bounds
            x, y = nose_point
            if 0 <= x < image_width and 0 <= y < image_height:
                import cv2
                cv2.circle(frame, (int(x), int(y)), 7, (0, 0, 255), -1)  # Larger red dot (BGR)
            else:
                import logging
                logging.warning(f'Nose point out of bounds: {nose_point}')
        else:
            import logging
            logging.info('Nose not detected in this frame.')

    def _extract_facial_points(self, features):
        # Flatten features dict to {name: (x, y)} for hand detection
        points = {}
        for part, value in features.items():
            if value is None:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    if v is not None and isinstance(v, (list, tuple)) and len(v) == 2:
                        points[f"{part}_{k}"] = (int(v[0]), int(v[1]))
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                points[part] = (int(value[0]), int(value[1]))
        return points
