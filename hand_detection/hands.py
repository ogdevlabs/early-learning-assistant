import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

class HandDetector:
    INDEX_FINGER_TIP = 8  # Landmark for index finger tip

    def __init__(self, detection_confidence=0.7, tracking_confidence=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing_utils = mp.solutions.drawing_utils
        self.previous_hand_positions = None

    def detect_hands(self, frame, roi=None):
        small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if roi is not None:
            rgb_frame = rgb_frame[roi[1]:roi[3], roi[0]:roi[2]]

        results = self.hands.process(rgb_frame)
        hand_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                hand_landmarks.append(hand_landmark)

        if roi is not None:
            for hand in hand_landmarks:
                for landmark in hand.landmark:
                    landmark.x = roi[0] + (landmark.x * (roi[2] - roi[0]))
                    landmark.y = roi[1] + (landmark.y * (roi[3] - roi[1]))

        return hand_landmarks

    def get_index_finger_tip(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        index_finger_tip = hand_landmarks.landmark[self.INDEX_FINGER_TIP]
        return int(index_finger_tip.x * w), int(index_finger_tip.y * h)

    def draw_hands(self, frame, hand_landmarks):
        for hand_landmark in hand_landmarks:
            for landmark in hand_landmark.landmark:
                self.drawing_utils.draw_landmarks(frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS)
