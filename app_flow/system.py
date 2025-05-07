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
        self.incorrect_cooldown = 3
        self.announced_target = False

        self.correct_time = 0
        self.waiting_after_success = False

        # Add a proximity threshold as configurable parameter
        self.face_proximity_threshold = 110
        self.threshold_adjustment_step = 10

        # Initialize the selfie segmentation model to blur the background
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        # Add hand stability tracking variables
        self.hand_position_history = []
        self.hand_stability_threshold = 15  # Number of frames to consider hand stable
        self.stability_required_time = 0.5  # Time in seconds to consider hand stable
        self.last_position_time = 0
        self.hand_is_stable = False

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

    def get_revised_face_boundary(self, face_landmarks, frame):
        h, w, _ = frame.shape
        face_boundary = []

        # Top of head
        for idx in [103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]:
            pt = face_landmarks.landmark[idx]
            face_boundary.append((int(pt.x * w), int(pt.y * h)))

        # Mouth region
        for idx in [0, 13, 14, 17, 84, 91, 181, 61, 39, 37, 267, 269, 270, 409, 291, 306]:
            pt = face_landmarks.landmark[idx]
            face_boundary.append((int(pt.x * w), int(pt.y * h)))

        # Side face
        for idx in [127, 234, 93, 35, 132, 108, 287, 336, 451, 365, 205, 424, 215, 54]:
            pt = face_landmarks.landmark[idx]
            face_boundary.append((int(pt.x * w), int(pt.y * h)))

        # Cheek and jaw line
        for idx in [127, 162, 21, 54, 93, 172, 136, 150, 149, 176, 148, 152]:
            pt = face_landmarks.landmark[idx]
            face_boundary.append((int(pt.x * w), int(pt.y * h)))

        return np.array(face_boundary, dtype=np.int32)

    def is_hand_near_face(self, hand_point, face_landmarks, frame):
        # Calculate distance from hand point to face landmarks
        h, w, _ = frame.shape
        # Create a polygon from face boundary
        face_boundary = self.get_revised_face_boundary(face_landmarks, frame)

        # calculate the distance from hand point to face boundary
        if cv2.pointPolygonTest(face_boundary, hand_point, False) >= 0:
            return True, 0

        # if outside the polygon, calculate the distance
        min_dist = cv2.pointPolygonTest(face_boundary, hand_point, True)

        # calculate approximate face size for adaptive threshold
        face_rect = cv2.boundingRect(face_boundary)
        face_size = max(face_rect[2], face_rect[3])

        # Adaptive threshold based on face size (15% of face size)
        threshold = face_size * 0.15

        # return whether the hand is near the face and the distance
        return abs(min_dist) < threshold, abs(min_dist)

    def get_face_boundary(self, face_landmarks, frame):
        h, w, _ = frame.shape
        face_boundary = []

        for idx in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 127, 162, 21, 54, 103, 67, 109, 10]:
            pt = face_landmarks.landmark[idx]
            face_boundary.append((int(pt.x * w), int(pt.y * h)))

        return np.array(face_boundary, dtype=np.int32)

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
                face_landmarks = face_results.multi_face_landmarks[0]

                # for face_landmarks in face_results.multi_face_landmarks:

                for detector_name, detector in self.detectors.items():
                    if detector_name != 'hands':
                        detector.detect(frame, face_landmarks)

                self.detectors['hands'].draw_hands(frame, hand_landmarks)

                # Array of facial points
                facial_points = get_facial_points(face_landmarks, frame)

                # Blinking logic
                target_coords = facial_points.get(self.current_target)
                if target_coords:
                    for coord in target_coords:
                        if coord is not None and 0 <= coord[0] < frame.shape[1] and 0 <= coord[1] < frame.shape[0]:
                            if time.time() % 1 > 0.5:
                                indicator_color = (255, 255, 255)
                            else:
                                indicator_color = None

                            if indicator_color:
                                cv2.circle(frame, coord, 10, indicator_color, -1)

                # Check if the target is already announced
                if not self.announced_target:
                    self.speech_manager.speak(f"Can you point your {self.current_target.split()[1]}")
                    self.announced_target = True

                # Check if the hand is near the face

                for hand in hand_landmarks:
                    index_finger_tip = self.detectors['hands'].get_index_finger_tip(frame, hand)

                    if isinstance(index_finger_tip, (tuple, list)) and len(index_finger_tip) == 2:
                        pass
                    else:
                        self.logger.error("Invalid index finger tip coordinates")
                        continue

                    is_near = False
                    distance = float('inf')

                    is_near, distance = self.is_hand_near_face(index_finger_tip, face_landmarks, frame)

                    # Display hand coordinates
                    cv2.putText(frame, f"Hand: {index_finger_tip[0]}, {index_finger_tip[1]})",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 1)

                    # Display distance to face
                    legend_y_start = frame.shape[0] - 140
                    cv2.rectangle(frame, (10, legend_y_start), (350, frame.shape[0] - 10), (0, 0, 0), -1)

                    # Ensure distance is valid before displaying
                    if distance != float('inf'):
                        hand_status_text = f"Hand near face ({int(distance)}px)" if is_near else f"Hand not near face ({int(distance)}px)"
                    else:
                        hand_status_text = "Hand not near face (N/A)"
                    hand_status_color = (0, 255, 0) if is_near else (0, 0, 255)

                    # Add legend for hand status
                    cv2.rectangle(frame, (10, frame.shape[0] - 40), (250, frame.shape[0] - 10), (0, 0, 0), -1)
                    cv2.putText(frame, hand_status_text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                hand_status_color, 2)

                    if is_near and not self.announced_target:
                        self.speech_manager.speak(f"Can you point your {self.current_target.split()[1]}?")
                        self.announced_target = True

                    if is_near:
                        current_time = time.time()

                        if current_time - self.last_position_time > 0.1:
                            self.hand_position_history.append((index_finger_tip, current_time))
                            self.last_position_time = current_time

                            # keep only positions within the last second
                            self.hand_position_history = [p for p in self.hand_position_history if current_time - p[1] < 1.0]

                            # Check if the hand is stable
                            if len(self.hand_position_history) >=3:
                                max_movement = 0
                                for i in range(1, len(self.hand_position_history)):
                                    prev_pos = self.hand_position_history[i-1][0]
                                    curr_pos = self.hand_position_history[i][0]
                                    movement = calculate_distance(prev_pos, curr_pos)
                                    max_movement = max(max_movement, movement)

                                # hand is stable if the max movement is less than the threshold
                                self.hand_is_stable = max_movement < self.hand_stability_threshold
                                stable_duration = current_time - self.hand_position_history[0][1] if self.hand_is_stable else 0
                            else:
                                self.hand_is_stable = False
                                stable_duration = 0


                        target_coords = facial_points.get(self.current_target)
                        if target_coords and self.hand_is_stable:
                            success = False
                            # if self.current_target in ["Pointing Nose", "Pointing Mouth"]:
                            for coord in target_coords:
                                if coord is not None and calculate_distance(index_finger_tip, coord) < 30:
                                    cv2.putText(frame, f"Correct: {self.current_target}", (50, 100),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (0, 255, 0), 2)
                                    self.speech_manager.speak("Good job!")
                                    self.correct_time = current_time
                                    self.waiting_after_success = True
                                    self.last_incorrect_time = current_time
                                    success = True
                                    break

                            if not success and not self.waiting_after_success:
                                # Provide continuous feedback for incorrect placement
                                cv2.putText(frame,
                                            f"Not quite, let's try again. Can you point your {self.current_target.split()[1]}",
                                            (50, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 0, 255), 2)


                                if (time.time() - self.last_incorrect_time > self.incorrect_cooldown and
                                        time.time() - self.correct_time > 3):
                                    part = self.current_target.split()[1]
                                    self.speech_manager.speak(f"Let's try again, try to point your {part}")
                                    self.last_incorrect_time = time.time()

            #frame = self.apply_background_blur(frame)
            return frame
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        try:
            self.speech_manager.speak("Welcome to Early Learning Game. Let's have fun!")
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
