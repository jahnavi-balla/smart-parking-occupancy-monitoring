"""
main.py
-------
Entry point for the real-time object detection app.

Flow:
    1. Load the YOLOv8 model (detector.py)
    2. Open the camera / video source (camera.py)
    3. Loop: read a frame -> detect objects -> draw boxes -> show FPS -> display
    4. Press 'q' to quit cleanly

Run it with:
    python src/main.py                  # uses default webcam (index 0)
    python src/main.py --source 1       # uses webcam index 1
    python src/main.py --source video.mp4   # runs on a video file instead
"""

import argparse
import sys

import cv2

from camera import Camera, CameraError
from detector import ObjectDetector, ModelLoadError
from utils import FPSCounter, draw_fps


def parse_args():
    """Reads command-line options so the user can customize the run
    without editing the source code."""
    parser = argparse.ArgumentParser(description="Real-time object detection with YOLOv8 + OpenCV")
    parser.add_argument(
        "--source",
        default=0,
        help="Camera index (e.g. 0) or path to a video file. Default: 0 (default webcam).",
    )
    parser.add_argument(
        "--model",
        default="models/yolov8n.pt",
        help="Path or name of the YOLOv8 model weights. Default: yolov8n.pt (nano - fastest).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence score (0-1) required to show a detection. Default: 0.5.",
    )
    args = parser.parse_args()

    # --source is read as a string from the command line. If it's a plain
    # number (like "0" or "1"), it means "webcam index" and cv2 expects an
    # int, not a string, so we convert it here.
    if isinstance(args.source, str) and args.source.isdigit():
        args.source = int(args.source)

    return args


def main():
    args = parse_args()

    # --- Step 1: Load the model ---
    # If this fails (bad weights path, missing dependency, etc.), we want
    # to fail fast with a clear message rather than opening a camera window
    # for no reason.
    try:
        print(f"Loading YOLOv8 model '{args.model}'...")
        detector = ObjectDetector(model_path=args.model, confidence_threshold=args.confidence)
        print("Model loaded successfully.")
    except ModelLoadError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # --- Step 2: Open the camera / video source ---
    try:
        camera = Camera(source=args.source).open()
        print(f"Camera/video source '{args.source}' opened successfully.")
    except CameraError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    fps_counter = FPSCounter()

    print("Press 'q' in the video window to quit.")

    # --- Step 3: Main detection loop ---
    try:
        for frame in camera.frames():
            # Run detection on this frame.
            results = detector.detect(frame)

            # Draw bounding boxes + labels directly onto the frame.
            frame = detector.draw_detections(frame, results)

            # Compute and overlay the current FPS.
            fps = fps_counter.update()
            frame = draw_fps(frame, fps)

            # Show the frame in a window.
            cv2.imshow("YOLOv8 Real-Time Object Detection", frame)

            # Wait 1ms for a key press; quit cleanly if 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit key pressed. Shutting down...")
                break

    except KeyboardInterrupt:
        # Lets the user press Ctrl+C in the terminal to stop gracefully too.
        print("\nInterrupted by user. Shutting down...")

    finally:
        # --- Step 4: Always clean up, even if something went wrong above ---
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed. Goodbye!")


if __name__ == "__main__":
    main()