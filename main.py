from application.face_detection_app import FaceDetectionApp
from hand_detection.detector import HandDetectorThread

if __name__ == '__main__':
    print("Starting Face Detection Application...")
    hand_thread = None
    app = None

    try:
        # Try with audio first, fallback to no audio if it fails
        try:
            app = FaceDetectionApp(enable_audio=True)
            print("App created with audio successfully")
        except Exception as audio_error:
            print(f"Audio initialization failed: {audio_error}")
            print("Continuing without audio...")
            app = FaceDetectionApp(enable_audio=False)
            print("App created without audio successfully")

        hand_thread = HandDetectorThread(
            app.get_latest_frame,
            app.get_latest_facial_points
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
        if hand_thread and hand_thread.is_alive():
            hand_thread.stop()
            hand_thread.join()
        if app:
            app.shutdown_audio()
        print("Application stopped")
