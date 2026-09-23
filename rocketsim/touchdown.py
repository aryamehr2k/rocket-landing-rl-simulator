"""Leg geometry, ground contact and touchdown grading. See docs/conventions.md."""

import math
from dataclasses import dataclass

import numpy as np

from rocketsim.config import RocketConfig
from rocketsim.quaternion import to_matrix, total_tilt

HALF = 0.5


@dataclass(frozen=True)
class TouchdownResult:
    time: float
    miss_distance: float
    vertical_speed: float
    lateral_speed: float
    tilt: float
    tip_over_angle: float
    tilt_limit: float
    failures: tuple[str, ...]

    @property
    def success(self) -> bool:
        return not self.failures


def foot_positions_body(rocket: RocketConfig, cg: float) -> np.ndarray:
    """Body positions of the feet relative to the CG, one row per leg."""
    angles = 2.0 * math.pi * np.arange(rocket.legs.count) / rocket.legs.count
    radius = rocket.legs.span * HALF
    feet_z = cg - rocket.feet_station
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, feet_z)])


def lowest_point(rocket: RocketConfig, cg: float, position: np.ndarray, q: np.ndarray) -> float:
    """Altitude of the lowest of the nose tip and the feet."""
    rotation = to_matrix(q)
    nose_body = np.array([0.0, 0.0, cg])
    points = np.vstack([nose_body, foot_positions_body(rocket, cg)])
    heights = position[2] + points @ rotation[2, :]
    return float(heights.min())


def tip_over_angle(rocket: RocketConfig, cg: float) -> float:
    """Tilt at which the CG is above the nearest edge of the foot polygon."""
    inscribed_radius = rocket.legs.span * HALF * math.cos(math.pi / rocket.legs.count)
    cg_height = rocket.feet_station - cg
    return math.atan2(inscribed_radius, cg_height)


def grade_touchdown(
    rocket: RocketConfig, cg: float, time: float, position: np.ndarray, velocity: np.ndarray, q: np.ndarray
) -> TouchdownResult:
    """Compare the touchdown state with the leg limits and the tip-over angle."""
    legs = rocket.legs
    tip_over = tip_over_angle(rocket, cg)
    tilt_limit = min(legs.max_tilt, tip_over)
    vertical_speed = -float(velocity[2])
    lateral_speed = float(math.hypot(velocity[0], velocity[1]))
    tilt = total_tilt(q)
    failures: list[str] = []
    if vertical_speed > legs.max_vertical_speed:
        failures.append("vertical speed")
    if lateral_speed > legs.max_lateral_speed:
        failures.append("lateral speed")
    if tilt > tilt_limit:
        failures.append("tilt")
    return TouchdownResult(
        time=time,
        miss_distance=float(math.hypot(position[0], position[1])),
        vertical_speed=vertical_speed,
        lateral_speed=lateral_speed,
        tilt=tilt,
        tip_over_angle=tip_over,
        tilt_limit=tilt_limit,
        failures=tuple(failures),
    )
