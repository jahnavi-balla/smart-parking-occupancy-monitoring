"""parking.py
------------
Core domain logic for parking occupancy monitoring.

Uses vehicle-to-parking-slot matching to determine occupancy.

Each detected vehicle is assigned to at most ONE parking spot.
This prevents a single detection from incorrectly occupying
multiple neighboring spots.

Temporal hysteresis prevents short detection glitches from
immediately changing a parking spot's state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

from utils import (
    COLOR_FREE,
    COLOR_OCCUPIED,
    COLOR_UNKNOWN,
    COLOR_WHITE,
    FONT,
    FONT_SCALE_SM,
    SPOT_ALPHA,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Minimum fraction of the vehicle bounding box that must lie inside
# a parking spot for it to be considered a possible match.
VEHICLE_COVERAGE_THRESHOLD: float = 0.35

# Minimum IoU between vehicle and parking spot for a possible match.
IOU_THRESHOLD: float = 0.20

# Number of consecutive frames required before a spot becomes occupied.
FRAMES_TO_OCCUPY: int = 3

# Number of consecutive frames required before an occupied spot
# becomes free.
FRAMES_TO_FREE: int = 12


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class SpotState(Enum):
    """Possible states for a single parking spot."""

    FREE = auto()
    OCCUPIED = auto()
    UNKNOWN = auto()


@dataclass
class ParkingSpot:
    """Represents one physical parking bay."""

    spot_id: str
    bbox: tuple[int, int, int, int]

    state: SpotState = SpotState.UNKNOWN

    _frames_occupied: int = field(default=0, repr=False)
    _frames_free: int = field(default=0, repr=False)

    @property
    def is_occupied(self) -> bool:
        """True when the confirmed state is OCCUPIED."""
        return self.state == SpotState.OCCUPIED

    @property
    def color(self) -> tuple[int, int, int]:
        """BGR colour corresponding to current state."""
        return {
            SpotState.FREE: COLOR_FREE,
            SpotState.OCCUPIED: COLOR_OCCUPIED,
            SpotState.UNKNOWN: COLOR_UNKNOWN,
        }[self.state]

    def update(self, vehicle_is_present: bool) -> None:
        """Update the spot state using hysteresis."""

        if vehicle_is_present:
            self._frames_occupied += 1
            self._frames_free = 0

        else:
            self._frames_free += 1
            self._frames_occupied = 0

        if self._frames_occupied >= FRAMES_TO_OCCUPY:
            if self.state != SpotState.OCCUPIED:
                logger.debug(
                    "Spot %s -> OCCUPIED",
                    self.spot_id,
                )

            self.state = SpotState.OCCUPIED

        elif self._frames_free >= FRAMES_TO_FREE:
            if self.state != SpotState.FREE:
                logger.debug(
                    "Spot %s -> FREE",
                    self.spot_id,
                )

            self.state = SpotState.FREE


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute Intersection-over-Union between two bounding boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)

    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compute_intersection_area(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> int:
    """Return intersection area between two bounding boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    width = max(0, ix2 - ix1)
    height = max(0, iy2 - iy1)

    return width * height


def compute_vehicle_coverage(
    vehicle_bbox: tuple[int, int, int, int],
    spot_bbox: tuple[int, int, int, int],
) -> float:
    """Calculate the fraction of the vehicle inside the parking spot."""

    intersection = compute_intersection_area(
        vehicle_bbox,
        spot_bbox,
    )

    x1, y1, x2, y2 = vehicle_bbox

    vehicle_area = max(0, x2 - x1) * max(0, y2 - y1)

    if vehicle_area == 0:
        return 0.0

    return intersection / vehicle_area


def is_center_inside(
    spot_bbox: tuple[int, int, int, int],
    vehicle_bbox: tuple[int, int, int, int],
) -> bool:
    """Return True if vehicle center lies inside the parking spot."""

    vx1, vy1, vx2, vy2 = vehicle_bbox

    center_x = (vx1 + vx2) / 2
    center_y = (vy1 + vy2) / 2

    sx1, sy1, sx2, sy2 = spot_bbox

    return (
        sx1 <= center_x <= sx2
        and sy1 <= center_y <= sy2
    )


# ---------------------------------------------------------------------------
# Vehicle → parking spot matching
# ---------------------------------------------------------------------------

def calculate_match_score(
    spot_bbox: tuple[int, int, int, int],
    vehicle_bbox: tuple[int, int, int, int],
) -> float:
    """Calculate how strongly a vehicle belongs to a parking spot.

    The main signal is vehicle coverage by the parking spot.
    IoU is included as a secondary signal.
    """

    coverage = compute_vehicle_coverage(
        vehicle_bbox,
        spot_bbox,
    )

    iou = compute_iou(
        spot_bbox,
        vehicle_bbox,
    )

    center_inside = is_center_inside(
        spot_bbox,
        vehicle_bbox,
    )

    # Strong vehicle coverage is the most useful signal.
    score = coverage * 0.70

    # IoU provides additional geometric evidence.
    score += iou * 0.20

    # Center containment gives a small bonus.
    if center_inside:
        score += 0.10

    return score


def vehicle_matches_spot(
    spot_bbox: tuple[int, int, int, int],
    vehicle_bbox: tuple[int, int, int, int],
) -> bool:
    """Determine whether a vehicle is a valid candidate for a spot."""

    coverage = compute_vehicle_coverage(
        vehicle_bbox,
        spot_bbox,
    )

    iou = compute_iou(
        spot_bbox,
        vehicle_bbox,
    )

    center_inside = is_center_inside(
        spot_bbox,
        vehicle_bbox,
    )

    return (
        coverage >= VEHICLE_COVERAGE_THRESHOLD
        or iou >= IOU_THRESHOLD
        or (
            center_inside
            and coverage >= 0.20
        )
    )


# ---------------------------------------------------------------------------
# ParkingLot
# ---------------------------------------------------------------------------

class ParkingLot:
    """Manages all parking spots and produces annotated video frames."""

    def __init__(self, spots: list[ParkingSpot]) -> None:
        self.spots: list[ParkingSpot] = spots

        logger.info(
            "ParkingLot initialised with %d spots",
            len(spots),
        )

    # ------------------------------------------------------------------
    # Occupancy logic
    # ------------------------------------------------------------------

    def update(self, detections: list[dict]) -> None:
        """Assign vehicles to their best matching parking spots.

        Each vehicle can occupy at most one parking spot.
        """

        vehicle_bboxes = [
            d["bbox"]
            for d in detections
        ]

        # Track which spots have already received a vehicle.
        assigned_spots: set[int] = set()

        # Track which spots have a vehicle in this frame.
        occupied_spots: set[int] = set()

        # Build all valid vehicle → spot candidates.
        candidates: list[tuple[float, int, int]] = []

        for vehicle_index, vehicle_bbox in enumerate(vehicle_bboxes):

            for spot_index, spot in enumerate(self.spots):

                if spot_index in assigned_spots:
                    continue

                if not vehicle_matches_spot(
                    spot.bbox,
                    vehicle_bbox,
                ):
                    continue

                score = calculate_match_score(
                    spot.bbox,
                    vehicle_bbox,
                )

                candidates.append(
                    (
                        score,
                        vehicle_index,
                        spot_index,
                    )
                )

        # Highest-confidence geometric matches first.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        assigned_vehicles: set[int] = set()

        for score, vehicle_index, spot_index in candidates:

            if vehicle_index in assigned_vehicles:
                continue

            if spot_index in assigned_spots:
                continue

            assigned_vehicles.add(vehicle_index)
            assigned_spots.add(spot_index)
            occupied_spots.add(spot_index)

        # Update every parking spot.
        for spot_index, spot in enumerate(self.spots):

            spot.update(
                spot_index in occupied_spots
            )

    # ------------------------------------------------------------------
    # Summary properties
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        """Total number of defined spots."""
        return len(self.spots)

    @property
    def occupied_count(self) -> int:
        """Number of currently occupied spots."""

        return sum(
            1
            for spot in self.spots
            if spot.is_occupied
        )

    @property
    def free_count(self) -> int:
        """Number of currently free spots."""

        return sum(
            1
            for spot in self.spots
            if spot.state == SpotState.FREE
        )

    def occupancy_summary(self) -> dict[str, int | float]:
        """Return current occupancy statistics."""

        pct = (
            self.occupied_count / self.total * 100
            if self.total
            else 0.0
        )

        return {
            "total": self.total,
            "occupied": self.occupied_count,
            "free": self.free_count,
            "occupancy_pct": round(pct, 1),
        }

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_spots(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Draw coloured parking-slot overlays."""

        overlay = frame.copy()

        for spot in self.spots:

            x1, y1, x2, y2 = spot.bbox

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                spot.color,
                thickness=-1,
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                spot.color,
                thickness=2,
            )

            label = spot.spot_id

            (tw, th), _ = cv2.getTextSize(
                label,
                FONT,
                FONT_SCALE_SM,
                1,
            )

            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2

            cv2.putText(
                overlay,
                label,
                (tx, ty),
                FONT,
                FONT_SCALE_SM,
                COLOR_WHITE,
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(
            overlay,
            SPOT_ALPHA,
            frame,
            1 - SPOT_ALPHA,
            0,
            frame,
        )

        return frame

    def draw_vehicle_boxes(
        self,
        frame: np.ndarray,
        detections: list[dict],
    ) -> np.ndarray:
        """Draw detected vehicle bounding boxes."""

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            label = (
                f"{det['class_name']} "
                f"{det['confidence']:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                COLOR_WHITE,
                1,
            )

            (tw, th), _ = cv2.getTextSize(
                label,
                FONT,
                0.45,
                1,
            )

            cv2.rectangle(
                frame,
                (x1, y1 - th - 6),
                (x1 + tw + 4, y1),
                (40, 40, 40),
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                FONT,
                0.45,
                COLOR_WHITE,
                1,
                cv2.LINE_AA,
            )

        return frame