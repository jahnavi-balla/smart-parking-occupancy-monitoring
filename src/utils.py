"""
utils.py
--------
Small, reusable helper pieces that don't belong to the camera or the model.

Contains:
    - FPSCounter: tracks how many frames we process per second.
    - draw_fps: draws the FPS value onto a frame.

Keeping these here (instead of copy-pasting into main.py) keeps main.py
focused on the "flow" of the program instead of low-level bookkeeping.
"""

import time
import cv2


class FPSCounter:
    """
    Calculates a smoothed Frames-Per-Second value.

    Why not just do `1 / (time_now - time_last_frame)`?
    Because that raw value jumps around wildly frame to frame (e.g. 28, 34, 19, 31...).
    This class keeps a short rolling average so the number displayed on screen
    is stable and easy to read.
    """

    def __init__(self, avg_over_frames: int = 10):
        # How many recent frame timings to average over.
        self.avg_over_frames = avg_over_frames
        # Stores the last N frame durations (in seconds).
        self._frame_times = []
        # Timestamp of the previous frame (None until the first update).
        self._prev_time = None

    def update(self) -> float:
        """
        Call this once per frame (right after you finish processing the frame).
        Returns the current smoothed FPS value.
        """
        current_time = time.time()

        if self._prev_time is None:
            # First frame ever - we have nothing to compare against yet.
            self._prev_time = current_time
            return 0.0

        # Time taken to process the last frame.
        elapsed = current_time - self._prev_time
        self._prev_time = current_time

        # Keep only the most recent `avg_over_frames` durations.
        self._frame_times.append(elapsed)
        if len(self._frame_times) > self.avg_over_frames:
            self._frame_times.pop(0)

        avg_frame_time = sum(self._frame_times) / len(self._frame_times)

        # Guard against division by zero on the (very unlikely) chance
        # a frame takes 0 seconds to process.
        if avg_frame_time <= 0:
            return 0.0

        return 1.0 / avg_frame_time


def draw_fps(frame, fps: float, position=(10, 30)):
    """
    Draws the FPS counter in the top-left corner of the frame.

    Args:
        frame: the OpenCV/numpy image to draw on (modified in place).
        fps: the FPS value to display.
        position: (x, y) pixel coordinates for the text.
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,          # font scale
        (0, 255, 0),  # green text (BGR format)
        2,            # thickness
        cv2.LINE_AA,  # anti-aliased for cleaner text
    )
    return frame