import threading
import time
import logging
import mediapipe as mp
import cv2

class HandDetectorThread(threading.Thread):
    def __init__(self, frame_provider, facial_points_provider, log_pointing_callback=None, show_hand_markers=True, hand_marker_color=(255,255,255), index_marker_color=(255,0,0), audio_feedback=None):
        super().__init__()
        self.frame_provider = frame_provider  # Callable that returns the latest frame
        self.facial_points_provider = facial_points_provider  # Callable that returns facial points dict
        self.log_pointing_callback = log_pointing_callback  # Optional callback for logging
        self.running = True
        self.mp_hands = mp.solutions.hands
        self.index_pointing_start = {}  # {facial_point_name: start_time}
        self.pointing_threshold = 3  # seconds
        self.near_threshold = 80  # pixels, adjust as needed
        self.show_hand_markers = show_hand_markers  # Enable/disable hand markers
        self.hand_marker_color = hand_marker_color
        self.index_marker_color = index_marker_color
        self.audio_feedback = audio_feedback  # Add audio feedback support
        self._hands_detected_announced = False  # Track announcement state

    def run(self):
        logging.info("Hand detection thread started")
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Lower threshold
            min_tracking_confidence=0.3   # Lower threshold
        ) as hands:
            while self.running:
                frame = self.frame_provider()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # logging.info(f"Processing frame: {frame.shape}")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    logging.info(f"Hand detected! Number of hands: {len(results.multi_hand_landmarks)}")
                    if not self._hands_detected_announced and self.audio_feedback:
                        self.audio_feedback("Hand detected")
                        self._hands_detected_announced = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        self._process_hand(hand_landmarks, frame)
                else:
                    self._hands_detected_announced = False  # Reset announcement state when no hands are detected
                time.sleep(0.01)
        logging.info("Hand detection thread stopped")

    def _process_hand(self, hand_landmarks, frame):
        h, w = frame.shape[:2]
        logging.info(f"Processing hand on frame {w}x{h}, show_markers={self.show_hand_markers}")

        # Always draw hand landmarks when show_hand_markers is True
        if self.show_hand_markers:
            logging.info(f"Drawing {len(hand_landmarks.landmark)} hand landmarks")
            for i, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                if i == 8:  # Index finger tip
                    cv2.circle(frame, (x, y), 6, self.index_marker_color, -1)
                    logging.info(f"Drew index finger at ({x}, {y})")
                else:
                    cv2.circle(frame, (x, y), 2, self.hand_marker_color, -1)

        # Check proximity for index finger tip
        facial_points = self.facial_points_provider()
        if facial_points:
            index_tip = hand_landmarks.landmark[8]
            index_tip_xy = (int(index_tip.x * w), int(index_tip.y * h))

            for name, pt in facial_points.items():
                if pt is None:
                    continue
                if self._is_near(index_tip_xy, pt, self.near_threshold):
                    now = time.time()
                    if name not in self.index_pointing_start:
                        self.index_pointing_start[name] = now
                    elif now - self.index_pointing_start[name] >= self.pointing_threshold:
                        msg = f"Index finger pointed at {name} for {self.pointing_threshold} seconds."
                        logging.info(msg)
                        if self.log_pointing_callback:
                            self.log_pointing_callback(name, pt)
                        if self.audio_feedback:
                            self.audio_feedback(f"Pointing at {name}")
                        self.index_pointing_start[name] = now
                else:
                    self.index_pointing_start.pop(name, None)

    def _is_near(self, pt1, pt2, threshold):
        dx = pt1[0] - pt2[0]
        dy = pt1[1] - pt2[1]
        return (dx*dx + dy*dy) ** 0.5 < threshold

    def stop(self):
        self.running = False
