import mediapipe as mp


class EarsDetector:
    def __init__(self):
        # Mediapipe Face Mesh provides 468 landmarks; ears are at specific indices
        self.left_ear_idx = 234  # Approximate left ear landmark
        self.right_ear_idx = 454  # Approximate right ear landmark

    def detect(self, face_landmarks, image_width=None, image_height=None):
        """
        Detects both ears from the given face landmarks.
        Args:
            face_landmarks: The facial landmarks data (from mediapipe FaceMesh).
            image_width: Width of the image (for pixel coordinates).
            image_height: Height of the image (for pixel coordinates).
        Returns:
            dict: {'left_ear': (x, y), 'right_ear': (x, y)} or None if not found.
        """
        if not face_landmarks:
            return None
        try:
            left_ear_landmark = face_landmarks.landmark[self.left_ear_idx]
            right_ear_landmark = face_landmarks.landmark[self.right_ear_idx]
            if image_width and image_height:
                left_ear = (int(left_ear_landmark.x * image_width), int(left_ear_landmark.y * image_height))
                right_ear = (int(right_ear_landmark.x * image_width), int(right_ear_landmark.y * image_height))
            else:
                left_ear = (left_ear_landmark.x, left_ear_landmark.y)
                right_ear = (right_ear_landmark.x, right_ear_landmark.y)
            return {'left_ear': left_ear, 'right_ear': right_ear}
        except Exception:
            return None
