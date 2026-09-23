"""Flight logs with a fixed CSV schema shared by the simulator and the firmware.

Columns the writer does not know yet are written as nan. The firmware writes the same
column names for the values it has. Frames and units follow docs/conventions.md.
"""

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

SENSOR_COLUMNS = (
    "accel_bx_mps2", "accel_by_mps2", "accel_bz_mps2",
    "gyro_bx_radps", "gyro_by_radps", "gyro_bz_radps",
    "baro_alt_m",
)
STATE_FIELDS = (
    "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
    "qw", "qx", "qy", "qz", "wx_radps", "wy_radps", "wz_radps",
)
ESTIMATE_COLUMNS = tuple(f"est_{field}" for field in STATE_FIELDS)
TRUE_COLUMNS = tuple(f"true_{field}" for field in STATE_FIELDS) + ("true_mass_kg",)
ACTUATOR_COLUMNS = (
    "gimbal_pitch_cmd_rad", "gimbal_yaw_cmd_rad", "gimbal_pitch_act_rad", "gimbal_yaw_act_rad",
    "servo_pitch_us", "servo_yaw_us",
    "ascent_ignite_cmd", "ascent_thrust_n", "landing_ignite_cmd", "landing_thrust_n",
)
WIND_COLUMNS = ("wind_x_mps", "wind_y_mps")
COLUMNS: tuple[str, ...] = (
    ("time_s", "phase") + SENSOR_COLUMNS + ESTIMATE_COLUMNS + TRUE_COLUMNS + ACTUATOR_COLUMNS + WIND_COLUMNS
)
TEXT_COLUMNS = ("phase",)
NUMBER_FORMAT = "{:.7g}"


class FlightLogWriter:
    """Writes one CSV row per physics step. Use as a context manager."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        self.rows = 0

    def write(self, row: dict[str, Any]) -> None:
        """Write one row. Missing numeric columns become nan, missing text columns empty."""
        unknown = set(row) - set(COLUMNS)
        if unknown:
            raise KeyError(f"unknown flight log columns: {sorted(unknown)}")
        values = []
        for column in COLUMNS:
            if column in TEXT_COLUMNS:
                values.append(str(row.get(column, "")))
            else:
                values.append(NUMBER_FORMAT.format(float(row.get(column, math.nan))))
        self._writer.writerow(values)
        self.rows += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "FlightLogWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_flight_log(path: str | Path) -> dict[str, np.ndarray]:
    """Read a flight log CSV into one numpy array per column."""
    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if reader.fieldnames is None:
        raise ValueError(f"{path}: empty flight log")
    log: dict[str, np.ndarray] = {}
    for column in reader.fieldnames:
        if column in TEXT_COLUMNS:
            log[column] = np.array([row[column] for row in rows], dtype=str)
        else:
            log[column] = np.array([float(row[column]) if row[column] else math.nan for row in rows])
    return log
