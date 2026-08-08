"""utils.py
----------
Shared constants, colour palette, and drawing helpers used across the project.

Modules that import from here:
    detector.py  — VEHICLE_CLASS_IDS
    parking.py   — colours, FONT, FONT_SCALE_SM, SPOT_ALPHA
    main.py      — FPSCounter, draw_status_panel
"""

from __future__ import annotations

import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Vehicle class IDs — VisDrone label set
# ---------------------------------------------------------------------------
# The VisDrone-trained YOLO11s checkpoint uses these class indices:
#   0  pedestrian        6  tricycle
#   1  people            7  awning-tricycle
#   2  bicycle           8  bus
#   3  car               9  motor
#   4  van              10  others
#   5  truck
#
# We keep only the four classes that represent parking-relevant vehicles.
# Pedestrians, cyclists, tricycles, and motorcycles are ignored.
VEHICLE_CLASS_IDS: set[int] = {3, 4, 5, 8}  # car, van, truck, bus


# ---------------------------------------------------------------------------
# Colour palette  (BGR — OpenCV convention)
# ---------------------------------------------------------------------------

COLOR_FREE      = ( 34, 197,  94)   # green  — vacant spot
COLOR_OCCUPIED  = ( 68,  68, 239)   # red    — occupied spot
COLOR_UNKNOWN   = (163, 175, 156)   # grey   — not yet confirmed
COLOR_WHITE     = (255, 255, 255)
COLOR_YELLOW    = (  0, 212, 255)   # accent for HUD title
COLOR_PANEL_BG  = ( 15,  15,  15)   # near-black HUD background


# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------

SPOT_ALPHA      = 0.35   # transparency of filled spot rectangles
PANEL_ALPHA     = 0.75   # transparency of the status HUD panel
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SM   = 0.55
FONT_SCALE_MD   = 0.70
FONT_THICKNESS  = 2


# ---------------------------------------------------------------------------
# FPSCounter
# ---------------------------------------------------------------------------

class FPSCounter:
    """Smoothed frames-per-second counter using a rolling average.

    A raw per-frame ``1 / delta_time`` value jumps too wildly to read.
    Averaging over the last N frames produces a stable display number.

    Args:
        avg_over_frames: Number of recent frame timings to average. Default 10.
    """

    def __init__(self, avg_over_frames: int = 10) -> None:
        self.avg_over_frames = avg_over_frames
        self._frame_times: list[float] = []
        self._prev_time: float | None = None

    def update(self) -> float:
        """Record current frame and return smoothed FPS.

        Returns:
            Smoothed FPS as a float. Returns 0.0 on the first call.
        """
        now = time.time()

        if self._prev_time is None:
            self._prev_time = now
            return 0.0

        elapsed = now - self._prev_time
        self._prev_time = now

        self._frame_times.append(elapsed)
        if len(self._frame_times) > self.avg_over_frames:
            self._frame_times.pop(0)

        avg = sum(self._frame_times) / len(self._frame_times)
        return 0.0 if avg <= 0 else 1.0 / avg


# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------

def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Draw FPS counter onto a frame.

    Args:
        frame: OpenCV BGR image (modified in place).
        fps: Smoothed FPS value to display.
        position: Top-left (x, y) pixel position for the text.

    Returns:
        The same frame with FPS text drawn.
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        position,
        FONT,
        FONT_SCALE_MD,
        COLOR_FREE,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return frame


def draw_status_panel(
    frame: np.ndarray,
    total_spots: int,
    occupied: int,
    free: int,
    fps: float,
) -> np.ndarray:
    """Draw the semi-transparent occupancy HUD in the top-left corner.

    Shows total / occupied / free spot counts, a live occupancy bar, and FPS.

    Args:
        frame: OpenCV BGR image (modified in place).
        total_spots: Total number of defined parking spots.
        occupied: Number of currently occupied spots.
        free: Number of currently free spots.
        fps: Current smoothed FPS.

    Returns:
        The same frame with the HUD drawn.
    """
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 240, 145

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        COLOR_PANEL_BG,
        thickness=-1,
    )
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)

    # Text rows
    occupancy_pct = (occupied / total_spots * 100) if total_spots > 0 else 0.0
    rows = [
        ("SMART PARKING",                COLOR_YELLOW,   panel_y + 24),
        (f"Total spots : {total_spots}", COLOR_WHITE,    panel_y + 46),
        (f"Occupied    : {occupied}",    COLOR_OCCUPIED, panel_y + 66),
        (f"Free        : {free}",        COLOR_FREE,     panel_y + 86),
        (f"FPS         : {fps:.1f}",     COLOR_UNKNOWN,  panel_y + 106),
    ]
    for text, colour, y in rows:
        cv2.putText(
            frame, text,
            (panel_x + 10, y),
            FONT, FONT_SCALE_SM,
            colour, 1, cv2.LINE_AA,
        )

    # Occupancy bar
    bar_x  = panel_x + 10
    bar_y  = panel_y + 118
    bar_w  = panel_w - 20
    bar_h  = 10
    filled = int(bar_w * occupancy_pct / 100)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    bar_colour = COLOR_FREE if occupancy_pct < 70 else COLOR_OCCUPIED
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_colour, -1)

    cv2.putText(
        frame,
        f"{occupancy_pct:.0f}%",
        (bar_x + bar_w + 4, bar_y + bar_h),
        FONT, FONT_SCALE_SM,
        COLOR_WHITE, 1, cv2.LINE_AA,
    )

    return frame