from application.face_detection_app import FaceDetectionApp
from hand_detection.detector import HandDetectorThread

def _prompt_audio_devices():
    """Interactively list and select input/output audio devices.
    Returns (input_id, output_id) or (None, None) if default/skip."""
    try:
        import sounddevice as sd
    except Exception:
        print("sounddevice not available; using default audio devices.")
        return None, None
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"Unable to query audio devices: {e}. Using defaults.")
        return None, None
    print("\nAvailable audio devices:")
    for idx, dev in enumerate(devices):
        caps = []
        if dev.get('max_input_channels', 0) > 0:
            caps.append('In')
        if dev.get('max_output_channels', 0) > 0:
            caps.append('Out')
        print(f"  [{idx:02d}] {dev.get('name','?')} ({'/'.join(caps) or 'None'})")
    def _ask(kind):
        while True:
            raw = input(f"Select {kind} device index (Enter=default): ").strip()
            if raw == '':
                return None
            if not raw.isdigit():
                print("Enter a numeric index or press Enter for default.")
                continue
            idx = int(raw)
            if 0 <= idx < len(devices):
                if kind == 'input' and devices[idx]['max_input_channels'] == 0:
                    print("Chosen device has no input channels.")
                    continue
                if kind == 'output' and devices[idx]['max_output_channels'] == 0:
                    print("Chosen device has no output channels.")
                    continue
                return idx
            print("Index out of range.")
    in_id = _ask('input')
    out_id = _ask('output')
    return in_id, out_id

if __name__ == '__main__':
    print("Starting Face Detection Application...")
    hand_thread = None
    app = None

    try:
        # Interactive audio device selection BEFORE constructing audio
        input_id, output_id = _prompt_audio_devices()

        # Create application core (video + detectors)
        app = FaceDetectionApp(enable_audio=True)
        print("Core application created (audio deferred)")

        # Explicit blocking audio & microphone setup before launching streams
        try:
            print("Initializing audio & microphone (blocking)...")
            if app.initialize_audio(input_device_id=input_id, output_device_id=output_id):
                print("Audio & microphone setup complete")
            else:
                print("Audio initialization skipped or not enabled")
        except Exception as audio_error:
            print(f"Audio setup failed: {audio_error}")
            print("Falling back to no-audio mode...")
            # Release camera and any audio resources from the failed instance
            try:
                if app.cap and app.cap.isOpened():
                    app.cap.release()
                app.shutdown_audio()
            except Exception:
                pass
            # Re-create clean instance with audio disabled
            app = FaceDetectionApp(enable_audio=False)
            print("Application running without audio")

        # Start hand detection thread (independent of audio)
        hand_thread = HandDetectorThread(
            app.get_latest_frame,
            app.get_latest_facial_points
        )
        hand_thread.start()
        print("Hand detection thread started")

        print("Starting main application loop (video/face/hands)...")
        app.run()  # Keep GUI in main thread

    except KeyboardInterrupt:
        print("Stopping application (KeyboardInterrupt)...")
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
