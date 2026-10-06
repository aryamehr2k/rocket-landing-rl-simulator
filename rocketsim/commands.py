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

    Every field may be None, which hands that decision back to the flight computer's own rules
    (PID, trigger table, brake rules). The rocket YAML's `controllers` section says which phases
    a policy may steer at all. `brake` is the drag device opening in [0, 1].
    """

    delta_x: float | None = None
    delta_y: float | None = None
    ignite_landing: bool | None = None
    brake: float | None = None
    throttle: float | None = None  # electric vehicle only, 0 to 1
