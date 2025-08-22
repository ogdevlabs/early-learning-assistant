import cv2
import mediapipe as mp
import logging
from face_detection.detector import FaceDetector

class FaceDetectionApp:
    def __init__(self):
        # Configure logging first
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
        logging.info("Initializing FaceDetectionApp...")

        self.cap = cv2.VideoCapture(0)
        self.face_detector = FaceDetector()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.latest_frame = None
        self.latest_facial_points = {}

        logging.info("FaceDetectionApp initialization complete")

    def get_latest_frame(self):
        return self.latest_frame

    def get_latest_facial_points(self):
        return self.latest_facial_points

    def run(self):
        if not self.cap.isOpened():
            logging.error('Could not open webcam.')
            return

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
