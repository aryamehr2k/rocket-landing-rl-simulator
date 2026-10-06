"""Figures of electric vehicle flights: one flight against its mission, and many flights overlaid."""

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rocketsim.hop.mission import MissionConfig  # noqa: E402
from rocketsim.units import rad_to_deg  # noqa: E402

BLUE, ORANGE, AQUA, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#9aa1a8", "#c8402a"
GRID = "#d9d8d3"
FIGURE_SIZE = (12, 10)
LINE = 1.4


def _style(ax: Any, label: str) -> None:
    ax.set_ylabel(label)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def mission_figure(log: dict[str, np.ndarray], mission: MissionConfig, title: str) -> plt.Figure:
    """Inputs (throttle, gimbal), outputs (height, speed, position, tilt) and what was expected (reference, criteria)."""
    t = log["time_s"]
    pad_cg = log["true_z_m"][0]
    height = log["true_z_m"] - pad_cg
    fig, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE)
    ax = axes[0, 0]
    target, tol = mission.target_altitude, mission.altitude_tolerance
    ax.axhspan(target - tol, target + tol, color=AQUA, alpha=0.15, label=f"target {target:.0f} +- {tol:.0f} m")
    ax.plot(t, log["ref_height_m"], color=GREY, linewidth=LINE, linestyle="--", label="reference")
    ax.plot(t, height, color=BLUE, linewidth=LINE, label="flown (true)")
    ax.plot(t, log["est_z_m"] - pad_cg, color=ORANGE, linewidth=0.8, alpha=0.8, label="estimated")
    _style(ax, "Height (m)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    ax.plot(t, log["ref_vz_mps"], color=GREY, linewidth=LINE, linestyle="--", label="reference")
    ax.plot(t, log["true_vz_mps"], color=BLUE, linewidth=LINE, label="flown")
    _style(ax, "Vertical speed (m/s)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    circle = np.linspace(0.0, 2.0 * np.pi, 200)
    radius = mission.landing_radius
    ax.plot(radius * np.cos(circle), radius * np.sin(circle), color=AQUA, linewidth=LINE, label=f"landing radius {radius:.0f} m")
    ax.plot(log["true_x_m"], log["true_y_m"], color=BLUE, linewidth=LINE, label="path (true)")
    ax.plot(log["est_x_m"], log["est_y_m"], color=ORANGE, linewidth=0.8, alpha=0.8, label="estimated")
    ax.plot([log["true_x_m"][-1]], [log["true_y_m"][-1]], "o", color=RED, label="touchdown")
    ax.plot([0.0], [0.0], "s", color="#333333", label="pad")
    ax.set_aspect("equal", adjustable="datalim")
    _style(ax, "y (m)")
    ax.set_xlabel("x (m)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    qx, qy = log["true_qx"], log["true_qy"]
    tilt = rad_to_deg(np.arccos(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)))
    ax.plot(t, tilt, color=BLUE, linewidth=LINE)
    _style(ax, "Tilt (deg)")
    ax = axes[2, 0]
    ax.plot(t, log["throttle_cmd"], color=ORANGE, linewidth=LINE, label="commanded")
    ax.plot(t, log["throttle_act"], color=BLUE, linewidth=LINE, label="actual")
    _style(ax, "Throttle (0-1)")
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2, 1]
    ax.plot(t, rad_to_deg(log["gimbal_pitch_act_rad"]), color=BLUE, linewidth=LINE, label="pitch")
    ax.plot(t, rad_to_deg(log["gimbal_yaw_act_rad"]), color=ORANGE, linewidth=LINE, label="yaw")
    wind = np.hypot(log["wind_x_mps"], log["wind_y_mps"])
    twin = ax.twinx()
    twin.plot(t, wind, color=GREY, linewidth=0.8, alpha=0.7)
    twin.set_ylabel("Wind (m/s)", color=GREY)
    _style(ax, "Gimbal (deg)")
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def batch_figure(logs: list[dict[str, np.ndarray]], successes: list[bool], mission: MissionConfig, title: str) -> plt.Figure:
    """Many flights on top of each other: height against time and the path seen from above."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
    target, tol = mission.target_altitude, mission.altitude_tolerance
    left.axhspan(target - tol, target + tol, color=AQUA, alpha=0.15)
    circle = np.linspace(0.0, 2.0 * np.pi, 200)
    right.plot(mission.landing_radius * np.cos(circle), mission.landing_radius * np.sin(circle), color=AQUA, linewidth=LINE)
    for log, ok in zip(logs, successes):
        color = BLUE if ok else RED
        left.plot(log["time_s"], log["true_z_m"] - log["true_z_m"][0], color=color, linewidth=0.8, alpha=0.6)
        right.plot(log["true_x_m"], log["true_y_m"], color=color, linewidth=0.8, alpha=0.6)
        right.plot([log["true_x_m"][-1]], [log["true_y_m"][-1]], "o", color=color, markersize=3)
    right.plot([0.0], [0.0], "s", color="#333333")
    _style(left, "Height (m)")
    left.set_xlabel("Time (s)")
    _style(right, "y (m)")
    right.set_xlabel("x (m)")
    right.set_aspect("equal", adjustable="datalim")
    passed = sum(successes)
    fig.suptitle(f"{title}: {passed} of {len(successes)} met every criterion (blue pass, red fail)")
    fig.tight_layout()
    return fig
