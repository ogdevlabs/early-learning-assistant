import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Tuple, Dict
import pyttsx3
import time


class SpeechManager:
    def __init__(self, cooldown_seconds=3):
        self.engine = pyttsx3.init()
        self.spoken_recently = {}
        self.cooldown = cooldown_seconds

    def speak(self, label: str):
        current_time = time.time()
        last_spoken = self.spoken_recently.get(label, 0)

        if current_time - last_spoken >= self.cooldown:
            try:
                self.engine.say(label)
                self.engine.runAndWait()
                self.spoken_recently[label] = current_time
            except Exception as e:
                logging.error(f"Speech error: {e}")


class FaceHandInteractionSystem:
    def __init__(self, max_workers=4):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.speech_manager = SpeechManager()

        # MediaPipe initializations
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        self.detectors = self._initialize_detectors()

    def _initialize_detectors(self):
        from face_detection import (EyesDetector, NoseDetector, MouthDetector, EarsDetector)
        from hand_detection import HandDetector

        return {
            'eyes': EyesDetector(),
            'nose': NoseDetector(),
            'mouth': MouthDetector(),
            'ears': EarsDetector(),
            'hands': HandDetector()
        }

    def calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def get_facial_points(self, face_landmarks, frame) -> Dict[str, Tuple[int, int]]:
        def cords(idx):
            return (
                int(face_landmarks.landmark[idx].x * frame.shape[1]),
                int(face_landmarks.landmark[idx].y * frame.shape[0])
            )

        return {
            "Pointing Nose": cords(4),
            "Pointing Mouth": cords(13),
            "Pointing Eyes": cords(159),
            "Pointing Ears": cords(234)
        }

    def process_frame(self, frame):
        try:
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face and hand landmarks
            face_results = self.face_mesh.process(rgb_frame)
            hand_landmarks = self.detectors['hands'].detect_hands(frame)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Detect and draw facial features
                    for detector_name, detector in self.detectors.items():
                        if detector_name != 'hands':
                            detector.detect(frame, face_landmarks)
                    # Draw hands
                    self.detectors['hands'].draw_hands(frame, hand_landmarks)

                    # Get a facial points array [eyes, ears, mouth, nose]
                    facial_points = self.get_facial_points(face_landmarks, frame)

                    # Check for hand index finger
                    for hand in hand_landmarks:
                        index_finger_tip = self.detectors['hands'].get_index_finger_tip(frame, hand)

                        for label, target in facial_points.items():
                            if self.calculate_distance(index_finger_tip, target) < 30:
                                # Speak and annotate
                                cv2.putText(frame, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                self.speech_manager.speak(label)
            return frame
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow('Early Learning Assistant', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self.logger.critical(f"System error: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown(wait=True)


def main():
    logging.basicConfig(level=logging.INFO)
    system = FaceHandInteractionSystem()
    system.run()


if __name__ == "__main__":
    main()
