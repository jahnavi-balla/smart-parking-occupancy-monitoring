"""logger.py
-----------
Handles two distinct logging concerns:

1. **Application logs** — human-readable messages (INFO, DEBUG, ERROR)
   sent to the terminal and optionally to a rotating log file.
   Configured once at startup via ``setup_logging()``.

2. **Occupancy CSV logs** — machine-readable, timestamped rows written
   every N seconds to a CSV file so you can plot occupancy over time,
   import into Excel, or feed into a dashboard.
   Managed by ``OccupancyLogger``.

Keeping these two concerns in one file (rather than spreading CSV logic
across main.py or parking.py) follows the Single Responsibility Principle:
if you ever want to swap CSV for a database, you change only this file.
"""

from __future__ import annotations

import csv
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Application logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    to_file: bool = True,
) -> None:
    """Configure root logger for the whole application.

    After calling this, any module that does ``logging.getLogger(__name__)``
    will automatically use these settings.

    Args:
        log_dir: Directory where rotating log files are written.
        level: Minimum log level to capture (e.g. ``logging.DEBUG``).
        to_file: If True, also write logs to a rotating file in log_dir.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Format: timestamp  LEVEL  module_name  message
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),   # always print to terminal
    ]

    if to_file:
        # RotatingFileHandler caps each log file at 5 MB and keeps 3 backups
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=5 * 1024 * 1024,   # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# CSV occupancy logging
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["timestamp", "total_spots", "occupied", "free", "occupancy_pct"]


class OccupancyLogger:
    """Writes parking occupancy snapshots to a CSV file at a fixed interval.

    Each run creates a new timestamped file so historical sessions are
    never overwritten (e.g. ``occupancy_2024-11-15_09-32-11.csv``).

    Args:
        log_dir: Directory where CSV files are written.
        interval_seconds: How many seconds to wait between rows.
                          Default 30 — fine-grained enough for analysis
                          without creating huge files for long sessions.

    Example CSV output::

        timestamp,total_spots,occupied,free,occupancy_pct
        2024-11-15 09:32:11,20,14,6,70.0
        2024-11-15 09:32:41,20,12,8,60.0
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        interval_seconds: float = 1.0,
    ) -> None:
        self._interval = interval_seconds
        self._last_log_time: float = 0.0
        self._logger = logging.getLogger(__name__)

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename per session — never clobber previous data
        session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._csv_path = log_dir / f"occupancy_{session_ts}.csv"

        # Write the CSV header once when the file is created
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

        self._logger.info("Occupancy CSV: %s", self._csv_path)

    def log(self, summary: dict[str, int | float]) -> None:
        """Write one row to the CSV if the interval has elapsed.

        Call this every frame from main.py; it self-throttles so it only
        actually writes every ``interval_seconds`` seconds.

        Args:
            summary: Dict returned by ``ParkingLot.occupancy_summary()``.
                     Must have keys: total, occupied, free, occupancy_pct.
        """
        import time
        now = time.time()
        if now - self._last_log_time < self._interval:
            return   # not time yet — fast exit, no disk I/O

        self._last_log_time = now
        row = {
            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_spots":   summary.get("total", 0),
            "occupied":      summary.get("occupied", 0),
            "free":          summary.get("free", 0),
            "occupancy_pct": summary.get("occupancy_pct", 0.0),
        }

        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
            self._logger.debug(
                "Logged occupancy: %d/%d (%.1f%%)",
                row["occupied"], row["total_spots"], row["occupancy_pct"],
            )
        except OSError as exc:
            # Log the error but don't crash — occupancy display is more
            # important than the CSV write.
            self._logger.error("Failed to write CSV row: %s", exc)

    @property
    def csv_path(self) -> Path:
        """Path to the CSV file for this session."""
        return self._csv_path