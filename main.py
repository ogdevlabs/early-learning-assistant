import logging
from app_flow.system import FaceHandInteractionSystem
from scoring.evaluator import ScoringEvaluator
from video.capture import VideoCapture


def main():
    logging.basicConfig(level=logging.INFO)
    scorer = ScoringEvaluator()
    video = VideoCapture()

    video.run()

    # system = FaceHandInteractionSystem(scorer)
    #
    # system.run()

if __name__ == "__main__":
    main()
