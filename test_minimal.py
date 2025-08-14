#!/usr/bin/env python3
print("=== MINIMAL TEST VERSION ===")

try:
    print("1. Testing basic imports...")
    import cv2
    print("   - OpenCV imported successfully")

    import mediapipe as mp
    print("   - MediaPipe imported successfully")

    from face_detection.detector import FaceDetector
    print("   - FaceDetector imported successfully")

    print("2. Testing camera initialization...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("   - Camera opened successfully")
        cap.release()
    else:
        print("   - ERROR: Could not open camera")
        exit(1)

    print("3. Testing basic application creation...")
    from application.face_detection_app import FaceDetectionApp
    app = FaceDetectionApp(enable_audio=False)
    print("   - FaceDetectionApp created successfully")

    print("4. Testing basic frame capture...")
    frame = app.get_latest_frame()
    print(f"   - Initial frame: {type(frame)}")

    print("5. All basic tests passed!")
    print("6. Starting minimal video loop (press 'q' to quit)...")

    # Minimal video loop
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} captured successfully")

        cv2.imshow('Minimal Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Minimal test completed successfully!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
