import cv2
import dlib

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def detect_faces(self, frame):
        frame_to_gray_color_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.detector(frame_to_gray_color_scale)
        return detected_faces

    def get_landmarks(self, frame, detected_face):
        frame_to_gray_color_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(frame_to_gray_color_scale, detected_face)
        return landmarks

class EyesDetector(FaceDetector):
    def detect_eyes(self, landmarks):
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]
        return left_eye, right_eye

class NoseDetector(FaceDetector):
    def detect_nose(self, landmarks):
        nose = landmarks.parts()[27:36]
        return nose

class MouthDetector(FaceDetector):
    def detect_mouth(self, landmarks):
        mouth = landmarks.parts()[48:68]
        return mouth

class EarsDetector(FaceDetector):
    def detect_ears(self, landmarks):
        left_ear = landmarks.parts()[0:2]
        right_ear = landmarks.parts()[15:17]
        return left_ear, right_ear

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    eyes_detector = EyesDetector()
    nose_detector = NoseDetector()
    mouth_detector = MouthDetector()
    ears_detector = EarsDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect_faces(frame)
        for face in faces:
            landmarks = face_detector.get_landmarks(frame, face)

            # Draw landmarks
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Eyes
            left_eye, right_eye = eyes_detector.detect_eyes(landmarks)
            for eye in [left_eye, right_eye]:
                for i in range(len(eye)):
                    point1 = (eye[i].x, eye[i].y)
                    point2 = (eye[(i + 1) % len(eye)].x, eye[(i + 1) % len(eye)].y)
                    cv2.line(frame, point1, point2, (211, 211, 211), 2)

            # Nose
            nose = nose_detector.detect_nose(landmarks)
            for point in nose:
                cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

            # Mouth
            mouth = mouth_detector.detect_mouth(landmarks)
            for i in range(len(mouth)):
                point1 = (mouth[i].x, mouth[i].y)
                point2 = (mouth[(i + 1) % len(mouth)].x, mouth[(i + 1) % len(mouth)].y)
                cv2.line(frame, point1, point2, (0, 255, 255), 1)

            # Ears
            left_ear, right_ear = ears_detector.detect_ears(landmarks)
            for ear in [left_ear, right_ear]:
                for point in ear:
                    cv2.circle(frame, (point.x, point.y), 2, (255, 0, 0), -1)

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
