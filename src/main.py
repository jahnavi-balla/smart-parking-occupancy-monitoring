"""main.py
---------
Smart Parking Occupancy Monitoring System
Entry point — orchestrates all modules into a running application.

Responsibilities of this file only:
    1. Configure application logging.
    2. Parse CLI arguments.
    3. Initialise camera, detector, parking lot, FPS counter, CSV logger.
    4. Run the main frame loop.
    5. Shut down cleanly on quit or error.

No parking logic, drawing, or detection logic lives here.
Those concerns belong to parking.py, utils.py, and detector.py respectively.

Usage::

    # Run on the parking video with the VisDrone model (default):
    python src/main.py --source assets/parking_video.mp4

    # Override the model or slot file:
    python src/main.py --source assets/parking_video.mp4 \\
                       --model models/visdrone_yol11s.pt \\
                       --slots assets/slots.json

    # Live webcam (index 0):
    python src/main.py --source 0

Calibration workflow (first-time setup for a new camera)::

    # 1. Define parking zones interactively:
    python tools/define_slots.py --source assets/parking_video.mp4

    # 2. Run the system — slots load automatically:
    python src/main.py --source assets/parking_video.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from camera import Camera, CameraError
from detector import ObjectDetector, ModelLoadError, DEFAULT_MODEL_PATH
from logger import OccupancyLogger, setup_logging
from parking import ParkingLot, ParkingSpot
from utils import FPSCounter, draw_status_panel

log = logging.getLogger(__name__)

# Default path to the slot definition file produced by tools/define_slots.py.
DEFAULT_SLOTS_PATH = "assets/slots.json"


# ---------------------------------------------------------------------------
# Parking layout — JSON loader
# ---------------------------------------------------------------------------

def load_parking_lot_from_json(path: str = DEFAULT_SLOTS_PATH) -> ParkingLot:
    """Load parking spot definitions from a JSON file and return a ParkingLot.

    Expected JSON format (produced by tools/define_slots.py)::

        [
            {"id": "A1", "bbox": [x1, y1, x2, y2]},
            {"id": "A2", "bbox": [x1, y1, x2, y2]},
            ...
        ]

    Args:
        path: Path to the JSON slot definition file.

    Returns:
        A ``ParkingLot`` ready for the main loop.

    Raises:
        SystemExit: On missing file, invalid JSON, or malformed entries.
    """
    slots_path = Path(path)

    if not slots_path.exists():
        log.error(
            "Slot definition file not found: '%s'\n"
            "  Run the calibration tool first:\n"
            "  python tools/define_slots.py --source <your_video_or_image>",
            slots_path.resolve(),
        )
        sys.exit(1)

    try:
        entries: list[dict] = json.loads(slots_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("Failed to parse '%s' as JSON: %s", path, exc)
        sys.exit(1)
    except OSError as exc:
        log.error("Could not read slot file '%s': %s", path, exc)
        sys.exit(1)

    if not entries:
        log.error(
            "Slot file '%s' contains no entries. "
            "Draw at least one parking slot and press S to save.",
            path,
        )
        sys.exit(1)

    spots: list[ParkingSpot] = []
    for i, entry in enumerate(entries):
        if "id" not in entry or "bbox" not in entry:
            log.error("Entry %d in '%s' is missing 'id' or 'bbox' key: %s", i, path, entry)
            sys.exit(1)

        raw_bbox = entry["bbox"]
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            log.error(
                "Entry '%s' has an invalid bbox (expected [x1, y1, x2, y2]): %s",
                entry.get("id", i), raw_bbox,
            )
            sys.exit(1)

        bbox: tuple[int, int, int, int] = (
            int(raw_bbox[0]), int(raw_bbox[1]),
            int(raw_bbox[2]), int(raw_bbox[3]),
        )
        spots.append(ParkingSpot(spot_id=str(entry["id"]), bbox=bbox))

    log.info("Loaded %d parking spot(s) from '%s'", len(spots), slots_path.resolve())
    return ParkingLot(spots=spots)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with: source, model, confidence, log_dir,
        log_interval, width, height, slots.
    """
    parser = argparse.ArgumentParser(
        description="Smart Parking Occupancy Monitor — YOLO11s + VisDrone + OpenCV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=0,
        help="Camera index (0, 1, …) or path to a video file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO weights file. Defaults to the VisDrone YOLO11s checkpoint.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help=(
            "Minimum detection confidence (0.0–1.0). "
            "0.25 is calibrated for the VisDrone YOLO11s model on overhead footage."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for application logs and occupancy CSV files.",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=30.0,
        help="Seconds between occupancy CSV rows.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width (ignored for video files).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height (ignored for video files).",
    )
    parser.add_argument(
        "--slots",
        default=DEFAULT_SLOTS_PATH,
        help="Path to the JSON slot definition file.",
    )

    args = parser.parse_args()

    # VideoCapture expects int for webcam indices, not the string "0".
    if isinstance(args.source, str) and args.source.isdigit():
        args.source = int(args.source)

    return args


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Smart Parking Occupancy Monitoring System.

    Initialises all components, then processes frames in a loop until
    the user presses Q or the video source is exhausted.
    """
    args = parse_args()

    # ── 1. Logging ───────────────────────────────────────────────────────────
    setup_logging(log_dir=args.log_dir)
    log.info("=== Smart Parking Occupancy Monitor starting ===")
    log.info(
        "Source: %s | Model: %s | Confidence: %.2f",
        args.source, args.model, args.confidence,
    )

    # ── 2. Model ─────────────────────────────────────────────────────────────
    log.info("Loading YOLO model '%s' …", args.model)
    try:
        detector = ObjectDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
        )
    except ModelLoadError as exc:
        log.error("Model failed to load: %s", exc)
        sys.exit(1)

    # ── 3. Camera ─────────────────────────────────────────────────────────────
    log.info("Opening video source '%s' …", args.source)
    try:
        camera = Camera(source=args.source, width=args.width, height=args.height).open()
    except CameraError as exc:
        log.error("Camera failed to open: %s", exc)
        sys.exit(1)

    # ── 4. Parking lot, FPS counter, CSV logger ───────────────────────────────
    parking_lot   = load_parking_lot_from_json(args.slots)
    fps_counter   = FPSCounter()
    occupancy_log = OccupancyLogger(log_dir=args.log_dir, interval_seconds=args.log_interval)

    log.info(
        "Parking lot ready: %d spots | CSV → %s",
        parking_lot.total,
        occupancy_log.csv_path,
    )
    log.info("Press Q in the display window to quit.")

    # ── 5. Main frame loop ────────────────────────────────────────────────────
    try:
        for frame in camera.frames():

            vehicles = detector.detect_vehicles(frame)
            parking_lot.update(vehicles)
            parking_lot.draw_spots(frame)
            parking_lot.draw_vehicle_boxes(frame, vehicles)

            fps = fps_counter.update()
            draw_status_panel(
                frame=frame,
                total_spots=parking_lot.total,
                occupied=parking_lot.occupied_count,
                free=parking_lot.free_count,
                fps=fps,
            )

            occupancy_log.log(parking_lot.occupancy_summary())
            cv2.imshow("Smart Parking Occupancy Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Quit key pressed.")
                break

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl-C).")

    # ── 6. Clean shutdown ─────────────────────────────────────────────────────
    finally:
        camera.release()
        cv2.destroyAllWindows()
        log.info(
            "Session ended. Occupancy data saved to: %s",
            occupancy_log.csv_path,
        )
        log.info("=== Smart Parking Occupancy Monitor stopped ===")


if __name__ == "__main__":
    main()