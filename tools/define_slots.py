"""tools/define_slots.py
-----------------------
Parking Slot Calibration Tool
Smart Parking Occupancy Monitoring System

PURPOSE
-------
Run this tool ONCE on a representative frame from your camera feed.
Draw a rectangle over every parking bay. Press S to save.
The output JSON is consumed directly by main.py (via build_sample_parking_lot()
or a future JSON loader) — so you only calibrate once per camera installation.

USAGE
-----
    # Calibrate from a video file (uses the first frame):
    python tools/define_slots.py --source videos/parking_lot.mp4

    # Calibrate from a still image:
    python tools/define_slots.py --source assets/lot_reference.jpg

    # Override the default output path:
    python tools/define_slots.py --source ... --output assets/custom.json

MOUSE
-----
    Left-drag        Draw a new slot rectangle
    Release          Finalise and label the slot

KEYBOARD
--------
    S                Save all slots to JSON and quit
    R                Undo — remove the last drawn slot
    C                Clear all slots (starts over)
    Q  or  ESC       Quit without saving
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual constants  (kept here so nothing needs to import from src/)
# ---------------------------------------------------------------------------

# Slot rectangle — drawn and filled on the canvas
COLOR_SLOT_BORDER  = (34, 197,  94)   # green border (BGR)
COLOR_SLOT_FILL    = (34, 197,  94)   # green fill (same, blended at alpha)
SLOT_FILL_ALPHA    = 0.25             # transparency of the filled area

# Rectangle the user is currently dragging (not yet finalised)
COLOR_PREVIEW      = (0, 212, 255)    # yellow-amber preview rectangle

# Label text drawn inside each finalised slot
COLOR_LABEL_TEXT   = (255, 255, 255)  # white
COLOR_LABEL_BG     = ( 20,  20,  20)  # near-black backing rectangle

# HUD instructions panel (bottom of screen)
COLOR_HUD_BG       = ( 15,  15,  15)
COLOR_HUD_TEXT     = (200, 200, 200)
COLOR_KEY_HINT     = (0, 212, 255)    # accent colour for key names

FONT               = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL   = 0.52
FONT_SCALE_HUD     = 0.50
FONT_THICKNESS     = 1

# Minimum pixel area for a rectangle to be accepted as a valid slot.
# Prevents accidental single-click "slots" with near-zero area.
MIN_SLOT_AREA      = 400   # pixels²

# Number of slot IDs per row-letter before rolling to the next letter.
# e.g. SLOTS_PER_ROW=10 → A1…A10, B1…B10, …
SLOTS_PER_ROW      = 10

# Default output path (relative to the project root)
DEFAULT_OUTPUT     = "assets/slots.json"


# ---------------------------------------------------------------------------
# ID generator
# ---------------------------------------------------------------------------

def generate_slot_id(index: int) -> str:
    """Convert a zero-based slot index into a row-letter + column-number ID.

    Examples:
        0  → "A1"
        9  → "A10"   (if SLOTS_PER_ROW == 10)
        10 → "B1"
        25 → "C6"

    Args:
        index: Zero-based position of the slot in the list.

    Returns:
        ID string like "A1", "B3", "C12".
    """
    row_letter = chr(ord("A") + index // SLOTS_PER_ROW)
    col_number = (index % SLOTS_PER_ROW) + 1
    return f"{row_letter}{col_number}"


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------

class SlotAnnotationTool:
    """Interactive tool for drawing and saving parking slot bounding boxes.

    Displays a single camera frame and lets the user left-drag to define
    rectangular parking zones. Each zone is labelled automatically (A1, A2 …)
    and the completed list can be saved to a JSON file that the main
    occupancy system reads at startup.

    Args:
        frame:       The BGR image to annotate (grabbed from the source file).
        output_path: File path where the JSON will be written on S keypress.
    """

    # Name of the OpenCV display window
    WINDOW_TITLE = "Parking Slot Calibration  |  S=Save  R=Undo  C=Clear  Q=Quit"

    def __init__(self, frame: np.ndarray, output_path: Path) -> None:
        self._base_frame: np.ndarray = frame.copy()   # never drawn on directly
        self._canvas:     np.ndarray = frame.copy()   # redrawn every event
        self._output_path: Path      = output_path

        # List of finalised slots: each entry is {"id": str, "bbox": [x1,y1,x2,y2]}
        self._slots: list[dict] = []

        # State for the rectangle the user is currently dragging
        self._drag_start: Optional[tuple[int, int]] = None   # mouse-down position
        self._drag_end:   Optional[tuple[int, int]] = None   # current mouse position

        self._is_dragging: bool = False

    # ------------------------------------------------------------------
    # Frame loading (static factory — keeps __init__ simple)
    # ------------------------------------------------------------------

    @staticmethod
    def load_frame(source: str) -> np.ndarray:
        """Load the first (or only) frame from a video file or image.

        Args:
            source: Path to a video file (.mp4, .avi, …) or an image
                    file (.jpg, .png, …).

        Returns:
            BGR numpy array representing the frame.

        Raises:
            SystemExit: If the file cannot be opened or read.
        """
        path = Path(source)
        if not path.exists():
            log.error("Source file not found: %s", source)
            sys.exit(1)

        # Decide whether to treat the source as a video or a still image
        # based on its file extension.
        VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
        is_video = path.suffix.lower() in VIDEO_EXTENSIONS

        if is_video:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                log.error("Could not open video: %s", source)
                sys.exit(1)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                log.error("Could not read first frame from video: %s", source)
                sys.exit(1)
            log.info("Loaded first frame from video: %s", source)
        else:
            frame = cv2.imread(source)
            if frame is None:
                log.error("Could not read image file: %s", source)
                sys.exit(1)
            log.info("Loaded image: %s  (%dx%d)", source, frame.shape[1], frame.shape[0])

        return frame

    # ------------------------------------------------------------------
    # Slot ID generation
    # ------------------------------------------------------------------

    def _next_id(self) -> str:
        """Return the ID that the next drawn slot should receive.

        Returns:
            Auto-generated slot ID string (e.g. "A1", "B3").
        """
        return generate_slot_id(len(self._slots))

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Handle all mouse events for the annotation window.

        OpenCV calls this automatically on every mouse action inside the
        display window. We use three events:
            - LBUTTONDOWN  → record the drag start position
            - MOUSEMOVE    → update the live preview rectangle while dragging
            - LBUTTONUP    → finalise the slot if the rectangle is large enough

        Args:
            event:  OpenCV mouse event constant (e.g. cv2.EVENT_LBUTTONDOWN).
            x:      Current cursor x-coordinate in frame pixels.
            y:      Current cursor y-coordinate in frame pixels.
            flags:  Bitfield of active modifier keys / mouse buttons (unused).
            param:  User data passed via setMouseCallback (unused).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start  = (x, y)
            self._drag_end    = (x, y)
            self._is_dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and self._is_dragging:
            self._drag_end = (x, y)
            # Redraw the canvas so the user sees the rectangle update live
            self._refresh_canvas()

        elif event == cv2.EVENT_LBUTTONUP and self._is_dragging:
            self._is_dragging = False
            self._drag_end = (x, y)
            self._finalise_slot()

    def _finalise_slot(self) -> None:
        """Convert the current drag coordinates into a saved slot.

        Normalises the rectangle (so dragging in any direction works),
        rejects tiny accidental clicks, assigns an ID, and appends the
        slot to the internal list.
        """
        if self._drag_start is None or self._drag_end is None:
            return

        x1 = min(self._drag_start[0], self._drag_end[0])
        y1 = min(self._drag_start[1], self._drag_end[1])
        x2 = max(self._drag_start[0], self._drag_end[0])
        y2 = max(self._drag_start[1], self._drag_end[1])

        area = (x2 - x1) * (y2 - y1)
        if area < MIN_SLOT_AREA:
            log.debug("Ignoring tiny rectangle (area=%d px²) — probably an accidental click.", area)
            self._drag_start = self._drag_end = None
            return

        slot = {
            "id":   self._next_id(),
            "bbox": [x1, y1, x2, y2],
        }
        self._slots.append(slot)
        log.info("Added slot %s  bbox=[%d, %d, %d, %d]", slot["id"], x1, y1, x2, y2)

        # Clear drag state and repaint with the new finalised slot visible
        self._drag_start = self._drag_end = None
        self._refresh_canvas()

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def undo_last_slot(self) -> None:
        """Remove the most recently drawn slot (R key).

        Safe to call when the list is already empty.
        """
        if not self._slots:
            log.info("Nothing to undo.")
            return
        removed = self._slots.pop()
        log.info("Removed slot %s", removed["id"])
        self._refresh_canvas()

    def clear_slots(self) -> None:
        """Remove all drawn slots and reset to a blank canvas (C key)."""
        count = len(self._slots)
        self._slots.clear()
        log.info("Cleared all %d slot(s).", count)
        self._refresh_canvas()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_slots(self) -> bool:
        """Serialise all slots to JSON and write to disk (S key).

        Creates any missing parent directories automatically.

        Returns:
            True if the file was written successfully; False on error.
        """
        if not self._slots:
            log.warning("No slots to save — draw at least one slot first.")
            return False

        try:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._output_path, "w", encoding="utf-8") as fh:
                json.dump(self._slots, fh, indent=4)
            log.info(
                "Saved %d slot(s) → %s",
                len(self._slots),
                self._output_path.resolve(),
            )
            return True
        except OSError as exc:
            log.error("Failed to write JSON: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _refresh_canvas(self) -> None:
        """Repaint the display from scratch and push it to the window.

        Called after every state change (slot added, removed, dragging).
        Painting from the clean base frame each time avoids ghosting from
        previous draws accumulating on the canvas.
        """
        # Start from the original, unmodified frame
        canvas = self._base_frame.copy()

        # Layer 1 — finalised slots (semi-transparent fills + solid borders)
        self._draw_finalised_slots(canvas)

        # Layer 2 — live drag preview (drawn on top of finalised slots)
        if self._is_dragging and self._drag_start and self._drag_end:
            self._draw_preview_rect(canvas)

        # Layer 3 — HUD instructions at the bottom
        self._draw_hud(canvas)

        self._canvas = canvas
        cv2.imshow(self.WINDOW_TITLE, self._canvas)

    def _draw_finalised_slots(self, canvas: np.ndarray) -> None:
        """Draw all saved slots onto the canvas.

        Each slot gets:
            - A semi-transparent filled rectangle (shows the zone coverage).
            - A solid-coloured border (always visible, even over light tarmac).
            - A small label badge with the slot ID centred inside the rect.

        Args:
            canvas: BGR image to draw on (modified in place).
        """
        # We need an overlay copy to blend the filled rectangles at alpha
        overlay = canvas.copy()

        for slot in self._slots:
            x1, y1, x2, y2 = slot["bbox"]

            # Semi-transparent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_SLOT_FILL, thickness=-1)

        # Blend fills onto canvas at SLOT_FILL_ALPHA
        cv2.addWeighted(overlay, SLOT_FILL_ALPHA, canvas, 1.0 - SLOT_FILL_ALPHA, 0, canvas)

        # Now draw solid borders and labels directly (no alpha — always crisp)
        for slot in self._slots:
            x1, y1, x2, y2 = slot["bbox"]
            label = slot["id"]

            # Solid border
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_SLOT_BORDER, thickness=2)

            # Label: measure text, draw a backing rect, then the text
            (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE_LABEL, FONT_THICKNESS)
            pad  = 3
            lx   = x1 + ((x2 - x1) - tw) // 2        # horizontally centred
            ly   = y1 + ((y2 - y1) + th) // 2         # vertically centred

            # Dark backing rectangle so the text is readable on any background
            cv2.rectangle(
                canvas,
                (lx - pad, ly - th - pad),
                (lx + tw + pad, ly + baseline + pad),
                COLOR_LABEL_BG,
                thickness=-1,
            )
            cv2.putText(
                canvas, label, (lx, ly),
                FONT, FONT_SCALE_LABEL, COLOR_LABEL_TEXT,
                FONT_THICKNESS, cv2.LINE_AA,
            )

    def _draw_preview_rect(self, canvas: np.ndarray) -> None:
        """Draw the dashed/coloured preview rectangle while the user drags.

        Uses a dashed-style effect by drawing the rectangle in a
        contrasting colour, making it visually distinct from finalised slots.

        Args:
            canvas: BGR image to draw on (modified in place).
        """
        if self._drag_start is None or self._drag_end is None:
            return

        x1 = min(self._drag_start[0], self._drag_end[0])
        y1 = min(self._drag_start[1], self._drag_end[1])
        x2 = max(self._drag_start[0], self._drag_end[0])
        y2 = max(self._drag_start[1], self._drag_end[1])

        # Solid preview border in accent colour
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_PREVIEW, thickness=2)

        # Corner tick marks to make the boundary corners crisp and obvious
        tick = 10
        for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = 1 if cx == x1 else -1
            dy = 1 if cy == y1 else -1
            cv2.line(canvas, (cx, cy), (cx + dx * tick, cy), COLOR_PREVIEW, 2)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * tick), COLOR_PREVIEW, 2)

        # Upcoming slot ID shown in the top-left corner of the preview rect
        next_id = self._next_id()
        cv2.putText(
            canvas, next_id,
            (x1 + 4, y1 + 16),
            FONT, FONT_SCALE_LABEL, COLOR_PREVIEW,
            FONT_THICKNESS, cv2.LINE_AA,
        )

    def _draw_hud(self, canvas: np.ndarray) -> None:
        """Draw the instruction panel at the bottom of the frame.

        Renders a semi-transparent dark bar listing keyboard shortcuts
        and the current slot count so the user always knows the state.

        Args:
            canvas: BGR image to draw on (modified in place).
        """
        h, w = canvas.shape[:2]
        bar_height = 36
        bar_y      = h - bar_height

        # Semi-transparent HUD background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), COLOR_HUD_BG, thickness=-1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        # Key hint pairs: (key label, action description)
        hints = [
            ("Left-drag", "Draw slot"),
            ("S",         "Save"),
            ("R",         "Undo"),
            ("C",         "Clear"),
            ("Q / ESC",   "Quit"),
        ]

        # Slot counter — right-aligned in the HUD
        count_text = f"Slots: {len(self._slots)}"
        (cw, _), _ = cv2.getTextSize(count_text, FONT, FONT_SCALE_HUD, 1)
        cv2.putText(
            canvas, count_text,
            (w - cw - 12, bar_y + 23),
            FONT, FONT_SCALE_HUD, COLOR_KEY_HINT, 1, cv2.LINE_AA,
        )

        # Keyboard hints — evenly spaced across the bar
        section_w = (w - cw - 24) // len(hints)
        for i, (key, action) in enumerate(hints):
            base_x = 12 + i * section_w
            # Key name in accent colour
            cv2.putText(
                canvas, key,
                (base_x, bar_y + 15),
                FONT, FONT_SCALE_HUD, COLOR_KEY_HINT, 1, cv2.LINE_AA,
            )
            # Action description in muted colour below
            cv2.putText(
                canvas, action,
                (base_x, bar_y + 29),
                FONT, 0.40, COLOR_HUD_TEXT, 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the annotation window and enter the event loop.

        Blocks until the user presses Q, ESC, or S (save + quit).
        All state changes go through the mouse callback and keyboard
        handler below — this method just keeps the window alive and
        dispatches key events.
        """
        cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_TITLE, 1280, 720)
        cv2.setMouseCallback(self.WINDOW_TITLE, self.mouse_callback)

        # Initial paint before any user interaction
        self._refresh_canvas()

        log.info("Annotation window open. Draw slots, then press S to save.")

        while True:
            # waitKey(20) = wait up to 20 ms for a keypress, then loop.
            # This keeps the UI responsive without burning 100% CPU.
            key = cv2.waitKey(20) & 0xFF

            if key == ord("s"):
                if self.save_slots():
                    log.info("Slots saved. Closing tool.")
                    break
                # save_slots() already logged the warning if nothing to save

            elif key == ord("r"):
                self.undo_last_slot()

            elif key == ord("c"):
                self.clear_slots()

            elif key in (ord("q"), 27):   # 27 = ESC
                log.info("Quit without saving.")
                break

            # If the user closes the window via the OS close button,
            # getWindowProperty returns -1.0 — we treat that as a quit.
            try:
                prop = cv2.getWindowProperty(self.WINDOW_TITLE, cv2.WND_PROP_VISIBLE)
                if prop < 1:
                    log.info("Window closed by user.")
                    break
            except cv2.error:
                break

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the calibration tool.

    Returns:
        Namespace with attributes: ``source`` and ``output``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Parking Slot Calibration Tool\n"
            "Draw rectangles over parking bays, press S to save.\n"
            "Output JSON is read directly by the main occupancy system."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a video file or image (e.g. assets/lot.jpg, videos/feed.mp4).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Where to save the JSON file. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — load frame, run the annotation tool, exit cleanly."""
    args   = parse_args()
    frame  = SlotAnnotationTool.load_frame(args.source)
    output = Path(args.output)

    tool = SlotAnnotationTool(frame=frame, output_path=output)
    tool.run()


if __name__ == "__main__":
    main()