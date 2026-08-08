"""detector.py
-------------
Loads a YOLO model and runs inference on individual frames.

This project uses a YOLO11s checkpoint fine-tuned on the VisDrone dataset
(``models/visdrone_yol11s.pt``).  The VisDrone model was chosen because
COCO-pretrained weights fail on aerial/overhead parking footage — COCO cars
are photographed at street level (side/front views) whereas a parking-lot
camera sees only car rooftops, which the COCO model misclassifies as kitchen
appliances (ovens, bowls, etc.).

VisDrone class IDs used by this model:
    3 = car   4 = van   5 = truck   8 = bus
These are defined in utils.VEHICLE_CLASS_IDS and passed to the model so that
NMS is restricted to vehicle classes only (faster and cleaner than filtering
in Python afterwards).

Single Responsibility: this file owns exactly one thing — loading and
running the YOLO model.
"""

from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from utils import VEHICLE_CLASS_IDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ModelLoadError(Exception):
    """Raised when the YOLO model cannot be loaded."""


# ---------------------------------------------------------------------------
# Detection result type
# ---------------------------------------------------------------------------

# Each detected vehicle is a plain dict so callers never import ultralytics:
#   {
#     "bbox":       (x1, y1, x2, y2),   # ints, pixel coordinates
#     "class_id":   int,                 # VisDrone class index
#     "class_name": str,                 # e.g. "car", "van"
#     "confidence": float,               # 0.0 – 1.0
#   }
DetectionDict = dict[str, object]

# Default model path — relative to the project root (where main.py is run from).
DEFAULT_MODEL_PATH = "models/visdrone_yol11s.pt"


# ---------------------------------------------------------------------------
# ObjectDetector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """Loads a YOLO model once and runs vehicle inference on demand.

    Args:
        model_path: Path to the YOLO weights file.
                    Defaults to the VisDrone YOLO11s checkpoint.
        confidence_threshold: Minimum confidence to accept a detection.
                    0.25 works well for the VisDrone-trained model on
                    overhead parking footage.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model: YOLO = self._load_model()
        logger.info(
            "YOLO model loaded: %s (conf≥%.2f)",
            model_path,
            confidence_threshold,
        )

    def _load_model(self) -> YOLO:
        """Load YOLO weights, raising ModelLoadError on failure.

        Returns:
            Loaded ``YOLO`` model instance.

        Raises:
            ModelLoadError: If the weights file is missing or cannot be read.
        """
        try:
            return YOLO(self.model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load YOLO model '{self.model_path}'.\n"
                "Check that the weights file exists at the specified path.\n"
                f"  → {exc}"
            ) from exc

    def detect(self, frame: np.ndarray):
        """Run inference on one frame and return raw Ultralytics Results.

        ``classes=`` restricts YOLO's NMS step to vehicle classes only,
        avoiding unnecessary scoring of pedestrians, cyclists, etc.
        ``imgsz=640`` is explicit for deterministic, reproducible behaviour.

        Args:
            frame: BGR uint8 image from OpenCV.

        Returns:
            Ultralytics ``Results`` object for this frame.
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=sorted(VEHICLE_CLASS_IDS),  # VisDrone: car=3, van=4, truck=5, bus=8
            imgsz=640,
            verbose=False,
        )
        return results[0]

    def detect_vehicles(self, frame: np.ndarray) -> list[DetectionDict]:
        """Run inference and return vehicle detections as plain dicts.

        Args:
            frame: BGR uint8 image from OpenCV.

        Returns:
            List of detection dicts; empty list if no vehicles found.
            Keys per dict: ``bbox``, ``class_id``, ``class_name``,
            ``confidence``.
        """
        results = self.detect(frame)
        vehicles: list[DetectionDict] = []

        for box in results.boxes:
            class_id = int(box.cls[0])

            # Defensive guard: detect() already filters via classes=,
            # but this makes detect_vehicles() safe if called independently.
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            vehicles.append({
                "bbox":       (x1, y1, x2, y2),
                "class_id":   class_id,
                "class_name": self.model.names[class_id],
                "confidence": float(box.conf[0]),
            })

        logger.debug("Detected %d vehicle(s) in frame", len(vehicles))
        return vehicles