import logging
import random
from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import mediapipe as mp

from app_flow.speech import SpeechManager
from app_flow.utils import calculate_distance, get_facial_points


class FaceHandInteractionSystem:
    def __init__(self, max_workers=4):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.speech_manager = SpeechManager()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        self.detectors = self._initialize_detectors()
        self.facial_labels = ["Pointing Nose", "Pointing Mouth", "Pointing Eyes", "Pointing Ears"]
        self.current_target = random.choice(self.facial_labels)
        self.previous_target = self.current_target
        self.last_incorrect_time = 0
        self.incorrect_cooldown = 2
        self.announced_target = False

        self.correct_time = 0
        self.waiting_after_success = False

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

    def _select_new_target(self):
        new_target = self.current_target
        while new_target == self.previous_target:
            new_target = random.choice(self.facial_labels)
        self.previous_target = new_target
        self.current_target = new_target

    def process_frame(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb_frame)
            hand_landmarks = self.detectors['hands'].detect_hands(frame)

            # Top-right label
            cv2.putText(
                frame,
                f"Target: {self.current_target.split()[1]}",
                (frame.shape[1] - 220, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            current_time = time.time()
            if self.waiting_after_success:
                if current_time - self.correct_time > 1.5:
                    self.waiting_after_success = False
                    self._select_new_target()
                    self.announced_target = False
                return frame

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    for detector_name, detector in self.detectors.items():
                        if detector_name != 'hands':
                            detector.detect(frame, face_landmarks)

                    self.detectors['hands'].draw_hands(frame, hand_landmarks)

                    facial_points = get_facial_points(face_landmarks, frame)

                    if not self.announced_target:
                        self.speech_manager.speak(f"Touch your {self.current_target.split()[1]}")
                        self.announced_target = True

                    for hand in hand_landmarks:
                        index_finger_tip = self.detectors['hands'].get_index_finger_tip(frame, hand)

                        target_coords = facial_points.get(self.current_target)
                        if target_coords and calculate_distance(index_finger_tip, target_coords) < 30:
                            cv2.putText(frame, f"Correct: {self.current_target}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 2)
                            self.speech_manager.speak("Good job!")
                            self.correct_time = current_time
                            self.waiting_after_success = True
                            break

                        else:
                            if time.time() - self.last_incorrect_time > self.incorrect_cooldown:
                                part = self.current_target.split()[1]
                                self.speech_manager.speak(f"Try again, touch your {part}")
                                self.last_incorrect_time = time.time()

            return frame
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        try:
            self.speech_manager.speak("Welcome to Early Learning Game. Let's start!")
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