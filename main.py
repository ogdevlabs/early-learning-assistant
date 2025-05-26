import logging
from app_flow.system import FaceHandInteractionSystem
from scoring.evaluator import ScoringEvaluator

def main():
    logging.basicConfig(level=logging.INFO)
    scorer = ScoringEvaluator()
    system = FaceHandInteractionSystem(scorer)

    system.run()

if __name__ == "__main__":
    main()
