import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


class EarsDetector:
    LEFT_EAR = [234, 93, 132, 58, 172, 136, 361]  # Left ear approximation
    RIGHT_EAR = [454, 323, 361, 288, 397, 365, 132]  # Right ear approximation

    def detect(self, frame, face_landmarks):
        h, w, _ = frame.shape

        # Draw left ear
        left_ear_points = [(int(face_landmarks.landmark[l].x * w), int(face_landmarks.landmark[l].y * h)) for l in self.LEFT_EAR]
        cv2.polylines(frame, [np.array(left_ear_points)], isClosed=True, color=(255, 0, 0), thickness=1, lineType=0)

        # Draw right ear
        right_ear_points = [(int(face_landmarks.landmark[r].x * w), int(face_landmarks.landmark[r].y * h)) for r in self.RIGHT_EAR]
        cv2.polylines(frame, [np.array(right_ear_points)], isClosed=True, color=(255, 0, 0), thickness=1, lineType=0)