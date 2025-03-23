import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh


class NoseDetector:
    # Nose landmarks for bridge and side contours
    NOSE_BRIDGE = [6, 197, 195, 5, 4]  # Nose bridge
    NOSE_SIDES = [49, 279, 458, 278, 48]  # Left and right sides

    def detect(self, frame, face_landmarks):
        h, w, _ = frame.shape

        # Function to draw slim dotted lines
        def draw_dotted_line(points, color):
            for i in range(1, len(points)):
                p1 = (int(face_landmarks.landmark[points[i - 1]].x * w), int(face_landmarks.landmark[points[i - 1]].y * h))
                p2 = (int(face_landmarks.landmark[points[i]].x * w), int(face_landmarks.landmark[points[i]].y * h))
                # Dotted effect by skipping some pixels along the line
                num_dots = max(1, int(((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5) // 3)
                for j in range(num_dots):
                    alpha = j / num_dots
                    dot_x = int((1 - alpha) * p1[0] + alpha * p2[0])
                    dot_y = int((1 - alpha) * p1[1] + alpha * p2[1])
                    cv2.circle(frame, (dot_x, dot_y), 5, color=color, lineType=0, thickness=-1)

        # Draw the nose bridge with dotted lines
        draw_dotted_line(self.NOSE_BRIDGE, (255, 1, 255))

        # Draw the sides of the nose with dotted lines
        draw_dotted_line(self.NOSE_SIDES, (255, 1, 255))