from application.face_detection_app import FaceDetectionApp
from hand_detection.detector import HandDetectorThread
import threading
import time

if __name__ == '__main__':
    print("Starting Face Detection Application...")

    try:
        app = FaceDetectionApp(enable_audio=False)  # Disable audio temporarily to test
        print("App created successfully")

        hand_thread = HandDetectorThread(
            app.get_latest_frame,
            app.get_latest_facial_points,
            audio_feedback=None  # No audio for now
        )
        hand_thread.start()
        print("Hand detection thread started")

        print("Starting main application loop...")
        app.run()  # Keep GUI in main thread

    except KeyboardInterrupt:
        print("Stopping application...")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if 'hand_thread' in locals():
            hand_thread.stop()
            hand_thread.join()
        print("Application stopped")
