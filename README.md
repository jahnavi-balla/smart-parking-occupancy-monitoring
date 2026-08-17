# Smart Parking Occupancy Monitoring System

A portfolio-quality computer vision project that monitors parking lot occupancy in real time from a fixed overhead camera, using a YOLO11s model fine-tuned on the VisDrone aerial dataset.

![System output showing 69 parking spots with 53 occupied (77%) at 5.5 FPS](assets/demo_frame.png)

---

## Overview

The system processes video from a fixed elevated camera, detects vehicles using YOLO11s, and compares each vehicle's bounding box against 69 manually defined parking zones using Intersection-over-Union (IoU). Each zone is independently tracked with temporal hysteresis to prevent flickering. Results are displayed in real time and logged to CSV.

---

## Problem Statement

Standard COCO-pretrained YOLO models are trained on street-level photography where vehicles are seen from the side or front. An overhead parking-lot camera sees only car rooftops — a fundamentally different visual perspective. When tested, a COCO-pretrained YOLOv8s model classified parked cars as kitchen appliances (ovens, sinks, refrigerators) at confidence 0.05–0.21, producing zero usable vehicle detections.

---

## Why COCO YOLO Was Not Suitable

COCO's vehicle annotations come from street-level images. The model's classification head learns to associate the "car" class with a distinctive front grille, side profile, and windscreen. Viewed from directly above, a car is just a rectangular roof with no visible grille or windows — the model finds no match to its training data and falls back to the nearest visually similar COCO class (boxy rectangles → kitchen appliances).

---

## Solution

Replace the COCO-pretrained model with a YOLO11s checkpoint fine-tuned on **VisDrone2019-DET**, a large-scale aerial imagery dataset captured from drone-mounted cameras. The VisDrone-trained model already understands the overhead perspective, recognising car rooftops as cars with high confidence.

---

## Model Selection

| Candidate | Result |
|---|---|
| YOLOv8s (COCO pretrained) | Fails — classifies cars as kitchen appliances |
| RT-DETR (COCO) | Worst aerial performance (21.7% mAP on VisDrone); slow on CPU |
| YOLO11n (VisDrone) | Works; ~20 FPS CPU; slightly lower accuracy |
| **YOLO11s (VisDrone)** | **Selected — best accuracy/CPU-speed balance** |
| YOLOv9e (VisDrone) | Best accuracy but >400ms CPU latency; impractical |

### Why YOLO11s

YOLO11s fine-tuned on VisDrone achieves **72.4% mAP@50 for the car class** on aerial imagery. It runs at approximately 5–6 FPS on a CPU-only machine (AMD Ryzen 5 7535HS), which is sufficient for monitoring stationary parked vehicles. Larger models (YOLOv9e) offer higher accuracy but are impractical without a GPU.

### VisDrone Dataset

VisDrone2019-DET is a large-scale aerial detection benchmark created by Tianjin University. It contains 8,629 drone-captured images with 10 annotated classes including car, van, truck, and bus — exactly the overhead perspective needed. The dataset is publicly available for academic use.

### Important: We Did Not Train This Model

The checkpoint (`models/visdrone_yol11s.pt`) is a publicly available pre-trained model fine-tuned on VisDrone. We downloaded it and use it directly. No training was performed in this project.

---

## System Architecture

```
Video / Camera
      │
      ▼
  camera.py          — frame capture and release
      │
      ▼
  detector.py        — YOLO11s inference → vehicle bounding boxes
      │
      ▼
  parking.py         — IoU matching → FREE / OCCUPIED / UNKNOWN per spot
      │
      ├──▶ utils.py  — drawing helpers and HUD overlay
      │
      ▼
  logger.py          — CSV occupancy logging
      │
      ▼
  main.py            — orchestration, CLI, display loop
```

---

## Project Structure

```
smart-parking/
├── src/
│   ├── main.py              # Entry point
│   ├── detector.py          # YOLO inference
│   ├── parking.py           # Occupancy logic
│   ├── camera.py            # Video capture
│   ├── utils.py             # Constants, colours, HUD drawing
│   └── logger.py            # CSV logging
├── tools/
│   └── define_slots.py      # Interactive parking zone calibration tool
├── models/
│   └── visdrone_yol11s.pt   # VisDrone-trained YOLO11s weights (~19 MB)
├── assets/
│   ├── parking_video.mp4    # Test footage
│   └── slots.json           # 69 calibrated parking zone definitions
├── logs/                    # CSV occupancy logs (generated at runtime)
├── requirements.txt
├── .gitignore
├── README.md
├── PROJECT_DOCUMENTATION.md
└── MODEL_SELECTION.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/smart-parking.git
cd smart-parking

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
# Run on the test video (default settings)
python src/main.py --source assets/parking_video.mp4

# Run on a live webcam
python src/main.py --source 0

# Override the model or confidence threshold
python src/main.py \
  --source assets/parking_video.mp4 \
  --model models/visdrone_yol11s.pt \
  --confidence 0.25

# Custom slot file or log directory
python src/main.py \
  --source assets/parking_video.mp4 \
  --slots assets/slots.json \
  --log-dir logs \
  --log-interval 30
```

Press **Q** in the display window (or **Ctrl-C** in the terminal) to quit cleanly.

---

## Parking Slot Configuration

Parking zones are defined once using the interactive calibration tool:

```bash
python tools/define_slots.py --source assets/parking_video.mp4
```

Draw rectangles over each bay. Press **S** to save. The tool writes `assets/slots.json`, which is read automatically at startup.

`slots.json` format:
```json
[
  {"id": "A1", "bbox": [32, 83, 156, 147]},
  {"id": "A2", "bbox": [156, 83, 279, 146]}
]
```

---

## Occupancy Calculation

For each frame:
1. YOLO11s detects vehicle bounding boxes (car, van, truck, bus only).
2. Each detected vehicle is matched to its best-overlapping parking zone using a composite score: vehicle coverage (70%) + IoU (20%) + centre containment bonus (10%).
3. Each vehicle is assigned to at most one zone, preventing a single large detection from occupying multiple adjacent spots.
4. Each zone requires **3 consecutive frames** with a vehicle before becoming OCCUPIED, and **12 consecutive frames** without a vehicle before becoming FREE. This hysteresis prevents flickering from momentary detection failures.

Zone colours:
- **Blue** — OCCUPIED
- **Green** — FREE
- **Grey** — UNKNOWN (not yet confirmed)

---

## Vehicle Classes

The VisDrone model uses these class IDs (different from COCO):

| Class ID | Class | Used? |
|---|---|---|
| 3 | car | Yes |
| 4 | van | Yes |
| 5 | truck | Yes |
| 8 | bus | Yes |
| 0,1,2,6,7,9,10 | pedestrian, people, bicycle, tricycle, awning-tricycle, motor, others | ignored |

---

## Logging

Every 30 seconds (configurable), a row is appended to a timestamped CSV in `logs/`:

```
timestamp,total_spots,occupied,free,occupancy_pct
2026-08-08 18:49:00,69,55,14,79.7
2026-08-08 18:49:30,69,52,17,75.4
2026-08-08 18:50:00,69,54,15,78.3
```

Application events are written to `logs/app.log` (rotating, max 5 MB, 3 backups).

---

## Current Performance

Tested on AMD Ryzen 5 7535HS (CPU only), 28-second parking lot video:

| Metric | Value |
|---|---|
| Parking spots | 69 |
| Typical occupied | 53–55 |
| Typical free | 14–16 |
| Typical occupancy | 77–80% |
| Inference FPS | ~5–6 |

---

## Known Limitations

- Two parking spots near the lot boundaries are occasionally misclassified due to partially visible vehicles. These edge cases were investigated and determined not worth fixing without retraining the model specifically on this camera angle.
- CPU-only inference limits throughput to ~5–6 FPS. ONNX export or GPU deployment would significantly improve this.
- The VisDrone model was not trained on this specific parking lot. Camera-specific fine-tuning would improve accuracy further.

---

## Future Improvements

- Export model to ONNX/OpenVINO for 2–3× CPU speedup
- Add ByteTrack vehicle tracking for temporal consistency
- Web dashboard for live occupancy visualisation
- Historical occupancy analytics and peak-hour reporting
- Camera-specific fine-tuning on this parking lot's footage

---

## Technologies Used

- Python 3.11
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) — object detection
- [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) — aerial training dataset (model source)
- OpenCV — video capture and rendering
- NumPy — array operations
