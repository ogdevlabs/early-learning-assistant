import threading
import time

import cv2
import mediapipe as mp
from face_detection import EyesDetector, NoseDetector, MouthDetector, EarsDetector
from hand_detection import HandDetector
from speech.speech_flow import SpeechFlow

running =  True
speech = SpeechFlow()

def calculate_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def speech_thread():
    global running
    global speech

def video_thread():
    global running
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

                # Define facial areas
                cords = lambda idx: (
                    int(face_landmarks.landmark[idx].x * frame.shape[1]),
                    int(face_landmarks.landmark[idx].y * frame.shape[0])
                )

                POINTS = {
                    "Pointing Nose": cords(4),
                    "Pointing Mouth": cords(13),
                    "Pointing Eyes": cords(159),
                    "Pointing Ears": cords(234),
                    "Pointing Forehead": cords(10),
                }

                for hand in hand_landmarks:
                    index_finger_tip = hand_detector.get_index_finger_tip(frame, hand)

                    for label, target in POINTS.items():
                        if calculate_distance(index_finger_tip, target) < 30:
                            offset_y = {
                                "Pointing Nose": 50,
                                "Pointing Mouth": 100,
                                "Pointing Eyes": 150,
                                "Pointing Ears": 200,
                                "Pointing Forehead": 250,
                                "Tapping Head": 300,
                            }.get(label)

                            cv2.putText(frame, label, (50, offset_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            speech.speak(label)
        cv2.imshow('Face and Hand Detection', frame)

    # Exit with key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    global running
    try:
        # Start both threads
        vt = threading.Thread(target=video_thread, daemon=True)
        st = threading.Thread(target=speech_thread, daemon=True)

        vt.start()
        st.start()

        while running:
            time.sleep(0.5)  # Keep main thread alive

    except KeyboardInterrupt:
        print("Exiting...")
        running = False

    # Wait for threads to close gracefully
    vt.join()
    st.join()

if __name__ == "__main__":
    main()