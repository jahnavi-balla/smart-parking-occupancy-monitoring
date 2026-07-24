# Real-Time Object Detection with YOLOv8 and OpenCV

A beginner-friendly, interview-ready computer vision project that detects objects
in a live webcam feed (or a video file) in real time, drawing bounding boxes,
class labels, confidence scores, and an FPS counter on screen.

```
[Webcam / Video] --> [OpenCV frame capture] --> [YOLOv8 inference] --> [Draw boxes + labels] --> [Display]
```

---

## 1. Project Structure

```
yolo-object-detection/
├── src/
│   ├── main.py        # Entry point - ties everything together, runs the main loop
│   ├── detector.py     # Loads YOLOv8 model, runs inference, draws detections
│   ├── camera.py       # Wraps OpenCV VideoCapture (webcam or video file)
│   └── utils.py        # FPS counter + small drawing helper
├── requirements.txt
└── README.md
```

Only 4 Python files, each with a single clear responsibility:

| File | Responsibility |
|---|---|
| `camera.py` | Own the video source (open, read frames, release, handle errors) |
| `detector.py` | Own the model (load, run inference, draw results) |
| `utils.py` | Small stateless/self-contained helpers (FPS math, text overlay) |
| `main.py` | Wire the above together into a runnable program |

This is intentionally **not over-engineered** — no config files, no plugin
systems, no factories. Just three focused classes/modules and a script that
uses them, which is exactly what a beginner (or an interviewer) can read
top-to-bottom in a few minutes.

---

## 2. Installation

### Prerequisites
- Python 3.9+
- A working webcam (optional — you can run on a video file instead)

### Steps

```bash
# 1. Clone / download the project, then move into it
cd yolo-object-detection

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`ultralytics` will automatically download the YOLOv8 nano weights
(`yolov8n.pt`, ~6 MB) the first time you run the app — no manual download
needed, as long as you have an internet connection for that first run.

---

## 3. Usage

Run with the default webcam:

```bash
cd src
python main.py
```

Other options:

```bash
# Use a different webcam (e.g. an external USB camera)
python main.py --source 1

# Run on a video file instead of a live camera
python main.py --source ../videos/sample.mp4

# Use a bigger/more accurate YOLOv8 model (slower)
python main.py --model yolov8s.pt

# Only show detections the model is at least 70% confident about
python main.py --confidence 0.7
```

Press **`q`** in the video window (or **Ctrl+C** in the terminal) to quit.
The app always releases the camera and closes windows cleanly, even if it
was interrupted.

---

## 4. How It Works (Project Explanation)

1. **Model loading (`detector.py`)** — `ObjectDetector` loads a pretrained
   YOLOv8 nano model (`yolov8n.pt`) once at startup using the `ultralytics`
   library. YOLOv8n is the smallest variant of the YOLOv8 family — it
   trades a little accuracy for speed, which is exactly what's needed for
   real-time inference on a normal CPU.

2. **Video capture (`camera.py`)** — `Camera` wraps `cv2.VideoCapture` and
   exposes a simple `frames()` generator, so `main.py` can just do
   `for frame in camera.frames():` without dealing with `read()`/`ret`
   boilerplate. It also implements the context-manager protocol
   (`with Camera(...) as cam:`) so the camera is always released properly.

3. **Detection loop (`main.py`)** — For every frame captured:
   - `detector.detect(frame)` runs a forward pass through the YOLOv8 model
     and returns all detected objects (class, confidence, box coordinates).
   - `detector.draw_detections(frame, results)` draws a bounding box and a
     `class_name confidence` label for each detection.
   - `FPSCounter.update()` (in `utils.py`) computes a smoothed FPS value
     (a rolling average over the last 10 frames, so the number on screen
     doesn't jitter wildly) which is overlaid in the corner.
   - The frame is shown with `cv2.imshow`; the loop exits when the user
     presses `q`.

4. **Error handling** — Two custom exceptions keep failures readable:
   - `ModelLoadError` — raised if the YOLO weights can't be loaded
     (e.g. no internet on first run, corrupted file, missing dependency).
   - `CameraError` — raised if the webcam/video file can't be opened
     (e.g. no camera connected, wrong index, camera in use by another app).

   Both are caught in `main.py` and printed as a clear, human-readable
   message instead of a raw stack trace, and the program exits with a
   non-zero status code so it plays nicely in scripts/CI.

---

## 5. Key Concepts to Know for an Interview

- **Why YOLOv8 nano?** It's a single-stage detector (predicts boxes and
  classes in one forward pass, unlike two-stage detectors like Faster
  R-CNN), which makes it fast enough for real-time video. "Nano" is the
  smallest of the YOLOv8 size variants (n/s/m/l/x) — fewer parameters,
  faster inference, slightly lower accuracy than larger variants.
- **Why a generator for frames?** It keeps memory flat — one frame in
  memory at a time — instead of loading an entire video into a list first.
- **Why separate `detect()` and `draw_detections()`?** Single
  Responsibility Principle — the raw detection results could be reused for
  something other than drawing (e.g. counting objects, logging, triggering
  an alert) without duplicating the inference call.
- **Why a rolling-average FPS counter instead of instant FPS?** Instant
  `1 / delta_time` is noisy frame-to-frame; averaging over a short window
  gives a stable, readable number.
- **Confidence threshold:** Detections below the threshold (default 0.5)
  are filtered out by the model itself via the `conf` parameter, reducing
  false positives shown on screen.

---

## 6. Possible Extensions (good talking points)

- Save detections to a CSV/JSON log for analytics.
- Add object tracking (e.g. ByteTrack, already supported by `ultralytics`)
  to assign persistent IDs across frames instead of re-detecting from
  scratch each frame.
- Filter to specific classes only (e.g. only detect `person` and `car`).
- Deploy as a Flask/FastAPI service streaming annotated frames over HTTP.
- Swap `yolov8n.pt` for a custom-trained model on a domain-specific dataset.

---

## 7. Resume Bullet Points

Use one or two of these, adapted to your own results/metrics where possible:

- Built a real-time object detection system in Python using YOLOv8 and
  OpenCV, achieving live webcam inference with on-screen FPS monitoring
  and multi-class bounding box visualization.
- Designed a modular, production-style architecture (camera capture, model
  inference, and utility layers cleanly separated) enabling easy testing,
  extension, and reuse of individual components.
- Implemented robust error handling for hardware (camera) and model
  loading failures, ensuring graceful degradation instead of crashes.
- Applied a pretrained deep learning object detection model (YOLOv8) to a
  live video stream, translating an ML model into a usable real-time
  application.

---

## 8. License / Credits

- Object detection powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
- Video I/O and rendering powered by [OpenCV](https://opencv.org/).
