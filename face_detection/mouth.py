import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

class MouthDetector:
    def detect(self, frame, face_landmarks):
        None
        # mp_drawing.draw_landmarks(
        #     image=frame,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_LIPS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)  # Yellow lines/dots
        # )