import logging
import random
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
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
        self.incorrect_cooldown = 15
        self.announced_target = False

        self.correct_time = 0
        self.waiting_after_success = False

        # Add a proximity threshold as configurable parameter
        self.face_proximity_threshold = 110
        self.threshold_adjustment_step = 10

        # Initialize the selfie segmentation model to blur the background
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

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


    # Adding background blur functionality, near to Zoom or Teams by 99 ksize
    def apply_background_blur(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmenter.process(rgb_frame)

        if results.segmentation_mask is None:
            return frame

        condition = results.segmentation_mask > 0.1
        blurred = cv2.GaussianBlur(frame, (99, 99), 0)
        output = np.where(condition[..., None], frame, blurred)
        return output

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

                    # Array of facial points
                    facial_points = get_facial_points(face_landmarks, frame)

                    # Calculate the center of the face
                    face_center = None
                    if 'Pointing Nose' in facial_points:
                        face_center = facial_points['Pointing Nose']

                    # Display face center legend
                    if face_center:
                        cv2.circle(frame, face_center, 5, (255, 0, 0), -1)
                        cv2.putText(frame, "Face Center",
                                    (face_center[0] + 10, face_center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 0, 0), 2)

                    # Check if the target is already announced
                    if not self.announced_target:
                        self.speech_manager.speak(f"Touch your {self.current_target.split()[1]}")
                        self.announced_target = True

                    for hand in hand_landmarks:
                        index_finger_tip = self.detectors['hands'].get_index_finger_tip(frame, hand)

                        # Display hand coordinates
                        cv2.putText(frame, f"Hand: {index_finger_tip[0]}, {index_finger_tip[1]})",
                                    (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 1)

                        # Add proximity check to calculate distance to face from index finger tip
                        distance_to_face = None
                        is_near_face = False

                        if face_center:
                            distance_to_face = calculate_distance(index_finger_tip, face_center)
                            is_near_face = distance_to_face < self.face_proximity_threshold

                            # draw visual line between index finger tip and face center
                            cv2.line(frame, index_finger_tip, face_center,
                                     (0, 255, 0) if is_near_face else (0, 0, 255), 2)

                        # Display distance to face
                        legend_y_start = frame.shape[0] - 140
                        cv2.rectangle(frame, (10, legend_y_start), (350, frame.shape[0]-10), (0, 0, 0), -1)

                        # Hand coordinates
                        cv2.putText(frame, f"Hand: {index_finger_tip[0]}, {index_finger_tip[1]})",
                                    (20, legend_y_start + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 1)

                        if facial_points:
                            face_center = facial_points.get('face_center', None)
                            if face_center:
                                distance_to_face = calculate_distance(index_finger_tip, face_center)
                                is_near_face = distance_to_face < 110

                        # Display on the screen as helpful information
                        hand_status_text = "Hand near face" if is_near_face else "Hand not near face"
                        hand_status_color = (0, 255, 0) if is_near_face else (0, 0, 255)

                        # Add legend for hand status
                        cv2.rectangle(frame, (10, frame.shape[0] - 40), (250, frame.shape[0] - 10), (0, 0, 0), -1)
                        cv2.putText(frame, hand_status_text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    hand_status_color, 2)

                        if is_near_face:
                            target_coords = facial_points.get(self.current_target)
                            if target_coords and calculate_distance(index_finger_tip, target_coords) < 30:
                                cv2.putText(frame, f"Correct: {self.current_target}", (50, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX,
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

            frame = self.apply_background_blur(frame)
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
