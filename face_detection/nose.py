class NoseDetector:
    def detect(self, face_landmarks, image_width=640, image_height=480):
        """
        Detects the nose from the given face landmarks.
        Args:
            face_landmarks: The facial landmarks data (e.g., from mediapipe or dlib).
            image_width: Width of the image/frame.
            image_height: Height of the image/frame.
        Returns:
            tuple: (x, y) coordinates of the nose or None if not found.
        """
        import logging
        if face_landmarks is None:
            logging.info('No face landmarks provided to NoseDetector.')
            return None
        # Try both index 1 and 4 for nose tip
        for idx in [1, 4]:
            try:
                nose_landmark = face_landmarks.landmark[idx] if hasattr(face_landmarks, 'landmark') else face_landmarks[idx]
                x = int(getattr(nose_landmark, 'x', 0) * image_width)
                y = int(getattr(nose_landmark, 'y', 0) * image_height) - 20  # Move dot 10px up
                # logging.info(f'Nose detected at index {idx}: ({x}, {y})')
                return x, y
            except (IndexError, AttributeError, TypeError) as e:
                logging.warning(f'Failed to get nose at index {idx}: {e}')
        logging.info('Nose not detected in landmarks.')
        return None
