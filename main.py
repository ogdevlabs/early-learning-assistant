import logging
from app_flow.system import FaceHandInteractionSystem

def main():
    logging.basicConfig(level=logging.INFO)
    system = FaceHandInteractionSystem()
    system.run()

if __name__ == "__main__":
    main()
