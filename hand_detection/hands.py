import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

class HandDetector:
    INDEX_FINGER_TIP = 8  # Landmark for index finger tip

    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else []

    def get_index_finger_tip(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        index_finger_tip = hand_landmarks.landmark[self.INDEX_FINGER_TIP]
        return int(index_finger_tip.x * w), int(index_finger_tip.y * h)

    def draw_hands(self, frame, hand_landmarks):
        height, width, _ = frame.shape
        for landmarks in hand_landmarks:
            for i, lm in enumerate(landmarks.landmark):
                x, y = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
