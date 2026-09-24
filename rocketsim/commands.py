"""What the flight computer tells the actuators, and what an outside policy may tell the flight computer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlCommand:
    """One control step's output: servo commands in body gimbal angles plus igniter commands."""

    gimbal_pitch: float = 0.0
    gimbal_yaw: float = 0.0
    ignite_ascent: bool = False
    ignite_landing: bool = False
    brake_fraction: float = 0.0


@dataclass(frozen=True)
class PlaneAction:
    """A policy's action in world planes: gimbal toward +x, toward +y, the landing igniter and the brake.

    `brake` is the drag device opening in [0, 1]; None leaves it to the flight computer's rules.
    """

    delta_x: float
    delta_y: float
    ignite_landing: bool
    brake: float | None = None
