import mediapipe as mp

class EyesDetector:
    def __init__(self):
        # Mediapipe Face Mesh indices for left and right eyes
        self.left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 246]
        self.right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 466]

    def detect(self, face_landmarks, image_width=None, image_height=None):
        """
        Detects both eyes from the given face landmarks.
        Args:
            face_landmarks: The facial landmarks data (from mediapipe FaceMesh).
            image_width: Width of the image (for pixel coordinates).
            image_height: Height of the image (for pixel coordinates).
        Returns:
            dict: {'left_eye': [(x, y), ...], 'right_eye': [(x, y), ...]} or None if not found.
        """
        if not face_landmarks:
            return None
        try:
            left_eye = []
            right_eye = []
            for idx in self.left_eye_indices:
                lm = face_landmarks.landmark[idx]
                if image_width and image_height:
                    left_eye.append((int(lm.x * image_width), int(lm.y * image_height)))
                else:
                    left_eye.append((lm.x, lm.y))
            for idx in self.right_eye_indices:
                lm = face_landmarks.landmark[idx]
                if image_width and image_height:
                    right_eye.append((int(lm.x * image_width), int(lm.y * image_height)))
                else:
                    right_eye.append((lm.x, lm.y))
            return {'left_eye': left_eye, 'right_eye': right_eye}
        except Exception:
            return None
