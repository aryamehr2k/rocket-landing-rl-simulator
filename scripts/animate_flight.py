"""Animate one flight log CSV as a 3D view of the rocket, written as a GIF or MP4.

Usage: python scripts/animate_flight.py runs/flight.csv --out flight.gif --speed 2
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from rocketsim.config import RocketConfig, load_rocket_config  # noqa: E402
from rocketsim.hop.vehicle import is_vehicle_file, load_vehicle_config  # noqa: E402
from rocketsim.flightlog import ESTIMATE_COLUMNS, read_flight_log  # noqa: E402
from rocketsim.physics import thrust_direction  # noqa: E402
from rocketsim.quaternion import to_matrix  # noqa: E402
from rocketsim.units import rad_to_deg  # noqa: E402
from rocketsim.yaml_section import ConfigError  # noqa: E402

ROCKET_COLOR = "#2a78d6"
PLUME_COLOR = "#eb6834"
LEG_COLOR = "#1baf7a"
GRID_COLOR = "#d9d8d3"
TEXT_COLOR = "#0b0b0b"
FIGURE_SIZE = (7, 7)
DPI = 90
DEFAULT_FPS = 25
DEFAULT_SPEED = 1.0
DEFAULT_ELEV_DEG = 20.0
DEFAULT_AZIM_DEG = -60.0
DEFAULT_ROCKET = "configs/rockets/example_tvc.yaml"
DEFAULT_LEG_COUNT = 4
FOLLOW_BOX_M = 6.0
PLUME_M_PER_N = 0.02
PLUME_MAX_M = 2.0
VIEW_MARGIN_M = 1.0
TALL_VIEW_RATIO = 8.0  # above this height to width ratio the x and y axes are too short for any text
GRID_LINES = 11
GRID_WIDTH = 0.6
TRAIL_WIDTH = 0.8
BODY_WIDTH = 4.0
LEG_WIDTH = 1.5
PLUME_WIDTH = 3.0
NOSE_MARKER_SIZE = 6.0
PAD_MARKER_SIZE = 7.0
READOUT_POSITION = (0.02, 0.98)
READOUT_FONT_SIZE = 9
FORMATS = ("gif", "mp4")


def rocket_shape(rocket: RocketConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body-frame points relative to the CG: body line (tail, nose), leg lines, gimbal pivot.

    The leg array holds attach point, foot and a nan row per leg so one Line3D draws them all.
    """
    cg, length = rocket.airframe.dry_cg, rocket.airframe.length
    body = np.array([[0.0, 0.0, cg - length], [0.0, 0.0, cg]])
    pivot = np.array([0.0, 0.0, cg - rocket.gimbal.pivot])
    count = getattr(rocket.legs, "count", DEFAULT_LEG_COUNT)
    angles = 2.0 * math.pi * np.arange(count) / count
    attach_radius = rocket.airframe.reference_diameter / 2.0
    foot_radius = rocket.legs.span / 2.0
    legs = np.full((count, 3, 3), np.nan)
    legs[:, 0, 0] = attach_radius * np.cos(angles)
    legs[:, 0, 1] = attach_radius * np.sin(angles)
    legs[:, 0, 2] = cg - (length - rocket.legs.height)
    legs[:, 1, 0] = foot_radius * np.cos(angles)
    legs[:, 1, 1] = foot_radius * np.sin(angles)
    legs[:, 1, 2] = cg - (length + rocket.legs.height)
    return body, legs.reshape(-1, 3), pivot


def gimbal_angles(log: dict[str, np.ndarray], axis: str) -> np.ndarray:
    """Actual gimbal angle, falling back to the command and then to zero where the log has nan."""
    actual = log[f"gimbal_{axis}_act_rad"]
    commanded = log[f"gimbal_{axis}_cmd_rad"]
    return np.nan_to_num(np.where(np.isnan(actual), commanded, actual))


def frame_indices(time: np.ndarray, fps: int, speed: float) -> np.ndarray:
    """Log rows to show, one per output frame, so playback runs at speed times real time."""
    frame_times = np.arange(time[0], time[-1], speed / fps)
    indices = np.searchsorted(time, frame_times)
    return np.append(indices, len(time) - 1)


def state_prefix(log: dict[str, np.ndarray], estimated: bool) -> str:
    """Column prefix to animate. The estimated state must hold at least one value."""
    if not estimated:
        return "true"
    if all(np.all(np.isnan(log[column])) for column in ESTIMATE_COLUMNS):
        raise SystemExit("--estimated: every est_* column in this log is nan, nothing to draw")
    return "est"


class FlightScene:
    """The 3D figure and the artists that move from frame to frame."""

    def __init__(
        self,
        log: dict[str, np.ndarray],
        rocket: RocketConfig,
        prefix: str,
        follow: bool,
        elev: float,
        azim: float,
    ) -> None:
        self.time = log["time_s"]
        self.phase = log["phase"]
        self.follow = follow
        self.length = rocket.airframe.length
        self.body_points, self.leg_points, self.pivot = rocket_shape(rocket)
        self.position = np.column_stack([log[f"{prefix}_{axis}_m"] for axis in "xyz"])
        self.velocity = np.column_stack([log[f"{prefix}_v{axis}_mps"] for axis in "xyz"])
        self.quaternion = np.column_stack([log[f"{prefix}_q{part}"] for part in "wxyz"])
        self.gimbal = np.column_stack([gimbal_angles(log, "pitch"), gimbal_angles(log, "yaw")])
        thrust_columns = [name for name in ("ascent_thrust_n", "landing_thrust_n", "main_thrust_n") if name in log]
        self.thrust = sum(np.nan_to_num(log[name]) for name in thrust_columns)
        self.figure = plt.figure(figsize=FIGURE_SIZE)
        self.axes = self.figure.add_subplot(projection="3d")
        self.axes.view_init(elev=elev, azim=azim)
        low, high = self._view_extent()
        self._draw_ground(low, high)
        if follow:
            self.axes.set_box_aspect((1.0, 1.0, 1.0))
        else:
            self._set_limits(low, high)
            self.axes.set_box_aspect(tuple(high - low))
            self._hide_cramped_axis_text(high - low)
        plot = self.axes.plot
        self.trail, = plot([], [], [], color=ROCKET_COLOR, linewidth=TRAIL_WIDTH)
        self.body, = plot([], [], [], color=ROCKET_COLOR, linewidth=BODY_WIDTH, solid_capstyle="round")
        self.nose, = plot([], [], [], color=ROCKET_COLOR, marker="o", markersize=NOSE_MARKER_SIZE)
        self.legs, = plot([], [], [], color=LEG_COLOR, linewidth=LEG_WIDTH)
        self.plume, = plot([], [], [], color=PLUME_COLOR, linewidth=PLUME_WIDTH)
        self.readout = self.axes.text2D(
            *READOUT_POSITION, "", transform=self.axes.transAxes, va="top",
            family="monospace", size=READOUT_FONT_SIZE, color=TEXT_COLOR,
        )

    def _view_extent(self) -> tuple[np.ndarray, np.ndarray]:
        margin = self.length + VIEW_MARGIN_M
        low = np.minimum(np.nanmin(self.position, axis=0), 0.0) - margin
        high = np.maximum(np.nanmax(self.position, axis=0), 0.0) + margin
        low[2] = min(np.nanmin(self.position[:, 2]) - self.length, 0.0)
        return low, high

    def _draw_ground(self, low: np.ndarray, high: np.ndarray) -> None:
        xs = np.linspace(low[0], high[0], GRID_LINES)
        ys = np.linspace(low[1], high[1], GRID_LINES)
        lines = [((x, low[1]), (x, high[1])) for x in xs] + [((low[0], y), (high[0], y)) for y in ys]
        points = np.array([point for start, end in lines for point in (start, end, (np.nan, np.nan))])
        self.axes.plot(points[:, 0], points[:, 1], np.zeros(len(points)), color=GRID_COLOR, linewidth=GRID_WIDTH)
        self.axes.plot([0.0], [0.0], [0.0], color=TEXT_COLOR, marker="s", markersize=PAD_MARKER_SIZE)
        self.axes.grid(False)
        for axis, label in zip((self.axes.xaxis, self.axes.yaxis, self.axes.zaxis), ("x (m)", "y (m)", "z (m)")):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis.set_label_text(label)

    def _hide_cramped_axis_text(self, extent: np.ndarray) -> None:
        """A tall, thin flight squeezes the x and y axes into a few pixels, where numbers and names only overlap."""
        if extent[2] > TALL_VIEW_RATIO * max(extent[0], extent[1]):
            for axis in (self.axes.xaxis, self.axes.yaxis):
                axis.set_major_formatter(NullFormatter())
                axis.set_label_text("")

    def _set_limits(self, low: np.ndarray, high: np.ndarray) -> None:
        self.axes.set_xlim(low[0], high[0])
        self.axes.set_ylim(low[1], high[1])
        self.axes.set_zlim(low[2], high[2])

    def draw(self, index: int) -> None:
        """Move every changing artist to log row index."""
        rotation = to_matrix(self.quaternion[index])
        cg = self.position[index]
        body = cg + self.body_points @ rotation.T
        legs = cg + self.leg_points @ rotation.T
        pivot = cg + rotation @ self.pivot
        pitch, yaw = self.gimbal[index]
        plume_length = min(PLUME_M_PER_N * self.thrust[index], PLUME_MAX_M)
        plume_end = pivot - plume_length * (rotation @ thrust_direction(pitch, yaw))
        trail = self.position[: index + 1]
        if self.follow:
            # matplotlib does not clip 3D lines to the axes box, so hide the trail outside it
            inside = np.all(np.abs(trail - cg) <= FOLLOW_BOX_M / 2.0, axis=1, keepdims=True)
            trail = np.where(inside, trail, np.nan)
        self.trail.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        self.body.set_data_3d(*body.T)
        self.nose.set_data_3d(*body[1:].T)
        self.legs.set_data_3d(*legs.T)
        self.plume.set_data_3d(*np.array([pivot, plume_end]).T)
        self.readout.set_text(self._readout(index, rotation))
        if self.follow:
            half = FOLLOW_BOX_M / 2.0
            self._set_limits(cg - half, cg + half)

    def _readout(self, index: int, rotation: np.ndarray) -> str:
        speed = np.linalg.norm(np.nan_to_num(self.velocity[index]))
        tilt = math.acos(min(max(rotation[2, 2], -1.0), 1.0))
        pitch, yaw = self.gimbal[index]
        return (
            f"time   {self.time[index]:6.2f} s  {self.phase[index]}\n"
            f"alt    {self.position[index, 2]:6.2f} m\n"
            f"speed  {speed:6.2f} m/s\n"
            f"tilt   {rad_to_deg(tilt):6.1f} deg\n"
            f"gimbal {rad_to_deg(pitch):6.1f} deg pitch, {rad_to_deg(yaw):.1f} deg yaw\n"
            f"thrust {self.thrust[index]:6.1f} N"
        )


def make_writer(fmt: str, fps: int) -> PillowWriter | FFMpegWriter:
    """Movie writer for the output format. MP4 needs ffmpeg on PATH."""
    if fmt == "gif":
        return PillowWriter(fps=fps)
    if not FFMpegWriter.isAvailable():
        raise SystemExit("mp4 output needs ffmpeg on PATH; install ffmpeg or use --format gif")
    return FFMpegWriter(fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="flight log CSV")
    parser.add_argument("--rocket", default=DEFAULT_ROCKET, help="rocket or vehicle YAML that gives the drawn geometry")
    parser.add_argument("--out", help="GIF or MP4 to write; defaults to the CSV name with .gif")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="output frames per second")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="playback speed factor, 2 is twice real time")
    parser.add_argument("--format", choices=FORMATS, help="output format; defaults to the --out suffix")
    parser.add_argument("--elev", type=float, default=DEFAULT_ELEV_DEG, help="camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=DEFAULT_AZIM_DEG, help="camera azimuth in degrees")
    parser.add_argument("--follow", action="store_true", help=f"keep a {FOLLOW_BOX_M:g} m box centred on the rocket")
    parser.add_argument("--estimated", action="store_true", help="draw the est_* state instead of true_*")
    args = parser.parse_args()
    log_path = Path(args.log)
    out = Path(args.out) if args.out else log_path.with_suffix("." + (args.format or FORMATS[0]))
    fmt = args.format or out.suffix.lstrip(".").lower()
    if fmt not in FORMATS:
        parser.error(f"cannot tell the output format from {out.name}; pass --format gif or mp4")
    writer = make_writer(fmt, args.fps)
    try:
        log = read_flight_log(log_path)
        rocket = load_vehicle_config(args.rocket).body if is_vehicle_file(args.rocket) else load_rocket_config(args.rocket)
        scene = FlightScene(log, rocket, state_prefix(log, args.estimated), args.follow, args.elev, args.azim)
        frames = frame_indices(log["time_s"], args.fps, args.speed)
        animation = FuncAnimation(scene.figure, scene.draw, frames=frames, blit=False, repeat=False)
        animation.save(out, writer=writer, dpi=DPI)
    except FileNotFoundError as error:
        raise SystemExit(f"{error.filename}: {error.strerror}")
    except ConfigError as error:
        raise SystemExit(f"{error}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
