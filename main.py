import cv2
import mediapipe as mp
from face_detection import EyesDetector, NoseDetector, MouthDetector, EarsDetector
from hand_detection import HandDetector
from speech.speech_flow import SpeechFlow


def calculate_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Initialize Detectors
eyes_detector = EyesDetector()
nose_detector = NoseDetector()
mouth_detector = MouthDetector()
ears_detector = EarsDetector()
hand_detector = HandDetector()
speech = SpeechFlow()

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame color to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    hand_landmarks = hand_detector.detect_hands(frame)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            nose_detector.detect(frame, face_landmarks)
            mouth_detector.detect(frame, face_landmarks)
            eyes_detector.detect(frame, face_landmarks)
            ears_detector.detect(frame, face_landmarks)
            hand_detector.draw_hands(frame, hand_landmarks)

            # Define facil areas
            coords = lambda idx:(
                int(face_landmarks.landmark[idx].x * frame.shape[1]),
                int(face_landmarks.landmark[idx].y * frame.shape[0])
            )

            POINTS = {
                "Pointing Nose": coords(4),
                "Pointing Mouth": coords(13),
                "Pointing Eyes": coords(159),
                "Pointing Ears": coords(234),
                "Pointing Forehead": coords(10),
                "Tapping Head": coords(1),
            }

            for hand in hand_landmarks:
                index_finger_tip = hand_detector.get_index_finger_tip(frame,hand)

                for label, target in POINTS.items():
                    if calculate_distance(index_finger_tip, target) < 30:
                        offset_y = {
                            "Pointing Nose":50,
                            "Pointing Mouth": 100,
                            "Pointing Eyes": 150,
                            "Pointing Ears": 200,
                            "Pointing Forehead": 250,
                            "Tapping Head": 300,
                        }.get(label)

                        cv2.putText(frame, label, (50, offset_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

                        speech.speak(label)
                        break

            # # Define ear centers
            # left_ear_center = (int(face_landmarks.landmark[234].x * frame.shape[1]),
            #                    int(face_landmarks.landmark[234].y * frame.shape[0]))
            # right_ear_center = (int(face_landmarks.landmark[454].x * frame.shape[1]),
            #                     int(face_landmarks.landmark[454].y * frame.shape[0]))
            #
            # # Define eye centers
            # left_eye_center = (int(face_landmarks.landmark[159].x * frame.shape[1]),
            #                    int(face_landmarks.landmark[159].y * frame.shape[0]))
            # right_eye_center = (int(face_landmarks.landmark[386].x * frame.shape[1]),
            #                     int(face_landmarks.landmark[386].y * frame.shape[0]))
            #
            # # Define nose tip
            # nose_tip = (int(face_landmarks.landmark[4].x * frame.shape[1]),
            #             int(face_landmarks.landmark[4].y * frame.shape[0]))
            #
            # # Define a mouth center
            # mouth_center = (int(face_landmarks.landmark[13].x * frame.shape[1]),
            #                 int(face_landmarks.landmark[13].y * frame.shape[0]))
            # # Forehead center
            # forehead_center = (int(face_landmarks.landmark[10].x * frame.shape[1]),
            #                    int(face_landmarks.landmark[10].y * frame.shape[0]))
            # # Top head
            # hair_center = (int(face_landmarks.landmark[1].x * frame.shape[1]),
            #                    int(face_landmarks.landmark[1].y * frame.shape[0]))
            #
            # # Check hand proximity to nose and mouth
            # for hand in hand_landmarks:
            #     index_finger_tip = hand_detector.get_index_finger_tip(frame, hand)
            #
            #     if calculate_distance(index_finger_tip, nose_tip) < 30:
            #         cv2.putText(frame, "Pointing Nose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     elif calculate_distance(index_finger_tip, mouth_center) < 30:
            #         cv2.putText(frame, "Pointing Mouth", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     elif (calculate_distance(index_finger_tip, left_eye_center) < 30
            #           or calculate_distance(index_finger_tip,right_eye_center) < 30):
            #         cv2.putText(frame, "Pointing Eyes", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     elif (calculate_distance(index_finger_tip, left_ear_center) < 30
            #           or calculate_distance(index_finger_tip,right_ear_center) < 30):
            #         cv2.putText(frame, "Pointing Ears", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     elif calculate_distance(index_finger_tip, forehead_center) < 30:
            #         cv2.putText(frame, "Pointing Forehead", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     elif calculate_distance(index_finger_tip, hair_center) < 30:
            #         cv2.putText(frame, "Tapping head", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face and Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
speech.shutdown()
