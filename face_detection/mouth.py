import mediapipe as mp

class MouthDetector:
    def __init__(self):
        # Mediapipe Face Mesh indices for mouth corners and center
        self.mouth_left_idx = 61
        self.mouth_right_idx = 291
        self.mouth_center_idx = 13  # Lower lip center
        self.mouth_upper_lip_idx = 0  # Upper lip center
        self.mouth_lower_lip_idx = 17  # Lower lip bottom

    def detect(self, face_landmarks, image_width=None, image_height=None):
        """
        Detects the mouth from the given face landmarks.
        Args:
            face_landmarks: The facial landmarks data (from mediapipe FaceMesh).
            image_width: Width of the image (for pixel coordinates).
            image_height: Height of the image (for pixel coordinates).
        Returns:
            dict: {'mouth_left': (x, y), 'mouth_right': (x, y), 'mouth_center': (x, y),
                   'upper_lip': (x, y), 'lower_lip': (x, y)} or None if not found.
        """
        if not face_landmarks:
            return None
        try:
            def get_point(idx):
                lm = face_landmarks.landmark[idx]
                if image_width and image_height:
                    return (int(lm.x * image_width), int(lm.y * image_height))
                else:
                    return (lm.x, lm.y)
            return {
                'mouth_left': get_point(self.mouth_left_idx),
                'mouth_right': get_point(self.mouth_right_idx),
                'mouth_center': get_point(self.mouth_center_idx),
                'upper_lip': get_point(self.mouth_upper_lip_idx),
                'lower_lip': get_point(self.mouth_lower_lip_idx)
            }
        except Exception:
            return None
