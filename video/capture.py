import cv2
import logging


class VideoCapture:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    def run(self):
        cap = cv2.VideoCapture(0)
        try:
            while True:
                # Read a frame from the video source
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow('Early Learning Assistant', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        except Exception as e:
            self.logger.critical(f"System error: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

