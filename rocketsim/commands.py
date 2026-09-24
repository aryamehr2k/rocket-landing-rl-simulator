"""What the flight computer tells the actuators, and what an outside policy may tell the flight computer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlCommand:
    """One control step's output: servo commands in body gimbal angles plus igniter commands."""

    gimbal_pitch: float = 0.0
    gimbal_yaw: float = 0.0
    ignite_ascent: bool = False
    ignite_landing: bool = False


@dataclass(frozen=True)
class PlaneAction:
    """A policy's action in world planes: gimbal toward +x, toward +y, and the landing igniter."""

    delta_x: float
    delta_y: float
    ignite_landing: bool
