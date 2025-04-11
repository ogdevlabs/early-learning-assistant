from typing import Tuple, Dict

def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_facial_points(face_landmarks, frame) -> Dict[str, Tuple[int, int]]:
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
