"""Plot one flight log CSV: altitude, speeds, distance, tilt, gimbal and thrust against time.

Usage: python scripts/plot_flight.py runs/flight.csv --out flight.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rocketsim.flightlog import read_flight_log  # noqa: E402
from rocketsim.units import rad_to_deg  # noqa: E402

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
GRID_COLOR = "#d9d8d3"
GRID_LINE_WIDTH = 0.6
LINE_WIDTH = 1.5
FIGURE_SIZE = (11, 9)
DPI = 130


def tilt_degrees(log: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Total tilt from vertical, from the logged quaternion (w, x, y, z)."""
    qx, qy = log[f"{prefix}_qx"], log[f"{prefix}_qy"]
    axis_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return rad_to_deg(np.arccos(np.clip(axis_z, -1.0, 1.0)))


def plot_flight_log(log: dict[str, np.ndarray], title: str) -> plt.Figure:
    t = log["time_s"]
    distance = np.hypot(log["true_x_m"], log["true_y_m"])
    fig, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE, sharex=True)
    panels = [
        (axes[0, 0], "Altitude (m)", [("true", log["true_z_m"], SERIES_BLUE)]),
        (axes[0, 1], "Vertical speed (m/s)", [("true", log["true_vz_mps"], SERIES_BLUE)]),
        (axes[1, 0], "Distance from pad (m)", [("true", distance, SERIES_BLUE)]),
        (axes[1, 1], "Tilt (deg)", [("true", tilt_degrees(log, "true"), SERIES_BLUE)]),
        (
            axes[2, 0],
            "Gimbal (deg)",
            [
                ("pitch", rad_to_deg(log["gimbal_pitch_act_rad"]), SERIES_BLUE),
                ("yaw", rad_to_deg(log["gimbal_yaw_act_rad"]), SERIES_ORANGE),
            ],
        ),
        (
            axes[2, 1],
            "Thrust (N)",
            [("ascent", log["ascent_thrust_n"], SERIES_BLUE), ("landing", log["landing_thrust_n"], SERIES_AQUA)],
        ),
    ]
    for ax, label, series in panels:
        for name, values, color in series:
            if np.all(np.isnan(values)):
                continue
            ax.plot(t, values, color=color, linewidth=LINE_WIDTH, label=name)
        ax.set_ylabel(label)
        ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if len(series) > 1 and ax.lines:
            ax.legend(frameon=False)
    for ax in axes[2]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="flight log CSV")
    parser.add_argument("--out", help="PNG to write; defaults to the CSV name with .png")
    args = parser.parse_args()
    log_path = Path(args.log)
    out = Path(args.out) if args.out else log_path.with_suffix(".png")
    try:
        fig = plot_flight_log(read_flight_log(log_path), log_path.stem)
        fig.savefig(out, dpi=DPI)
    except FileNotFoundError as error:
        raise SystemExit(f"{error.filename}: {error.strerror}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
