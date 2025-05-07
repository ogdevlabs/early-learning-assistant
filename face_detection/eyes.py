import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

class EyesDetector:
    def detect(self, frame, face_landmarks):
        for eye in [mp_face_mesh.FACEMESH_LEFT_EYE, mp_face_mesh.FACEMESH_RIGHT_EYE]:
            None
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=face_landmarks,
            #     connections=eye,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)  # Light gray, bolder lines
            # )