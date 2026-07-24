"""
detector.py
-----------
Wraps the YOLOv8 model: loading it, running inference on a frame, and
drawing the results (boxes + labels + confidence) back onto that frame.

We use "yolov8n.pt" (the "nano" version) because it's the smallest and
fastest YOLOv8 model - a good fit for real-time webcam inference on a
CPU, while still being reasonably accurate. Ultralytics downloads this
weights file automatically the first time it's used.
"""

import cv2
from ultralytics import YOLO


class ModelLoadError(Exception):
    """Raised when the YOLO model fails to load."""
    pass


class ObjectDetector:
    """
    Loads a YOLOv8 model once, then repeatedly detects objects in frames.

    Keeping the model loaded as an instance attribute (instead of reloading
    it every frame) is what makes real-time performance possible - model
    loading is slow, inference on an already-loaded model is fast.
    """

    def __init__(self, model_path: str = "models/yolov8n.pt", confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Loads the YOLO model, wrapping any failure in a clear error."""
        try:
            model = YOLO(self.model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load YOLO model '{self.model_path}'. "
                f"Make sure you have an internet connection (to download "
                f"the weights the first time) and that 'ultralytics' is "
                f"installed correctly.\nOriginal error: {exc}"
            ) from exc
        return model

    def detect(self, frame):
        """
        Runs object detection on a single frame.

        Returns the raw Ultralytics "Results" object, which contains all
        detected boxes, their classes, and confidence scores. We keep this
        separate from drawing so the caller (main.py) could, in theory,
        use the raw results for something other than drawing (e.g. logging,
        counting objects, triggering alerts, etc.).
        """
        # verbose=False stops Ultralytics from printing a log line per frame.
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        return results[0]  # a single frame in -> a single Results object out

    def draw_detections(self, frame, results):
        """
        Draws bounding boxes, class names, and confidence scores onto the
        frame based on the detection results.

        Returns the same frame, modified in place, for convenience.
        """
        for box in results.boxes:
            # --- Extract box coordinates ---
            # xyxy = (x1, y1, x2, y2) = top-left and bottom-right corners.
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # --- Extract class name and confidence ---
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])

            # --- Draw the bounding box ---
            color = (0, 255, 0)  # green, in BGR (OpenCV's color order)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

            # --- Draw the label (class name + confidence) above the box ---
            label = f"{class_name} {confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Background rectangle behind the text so it's readable on
            # any background color.
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 4, y1),
                color,
                thickness=-1,  # -1 fills the rectangle
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # black text for contrast against the green box
                2,
                cv2.LINE_AA,
            )

        return frame