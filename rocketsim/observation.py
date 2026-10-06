"""What the policy sees: named fields from the flight computer's estimate, each divided by a scale.

The training YAML lists the fields by name; each means the same for the pitch plane (x) and the yaw plane (y).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rocketsim.flightcomputer import FlightComputer
from rocketsim.phases import Phase

PITCH_PLANE, YAW_PLANE = 0, 1
PLANES = (PITCH_PLANE, YAW_PLANE)


@dataclass(frozen=True)
class ObservationField:
    name: str
    scale: float


@dataclass(frozen=True)
class PlaneState:
    """Everything a field function may look at for one plane at one control step."""

    computer: FlightComputer
    plane: int
    time: float
    gimbal_command: float
    burn_time: float


def _height(s: PlaneState) -> float:
    return s.computer.height


def _vertical_speed(s: PlaneState) -> float:
    return float(s.computer.estimate.velocity[2])


def _lateral_position(s: PlaneState) -> float:
    return float(s.computer.estimate.position[s.plane])


def _lateral_speed(s: PlaneState) -> float:
    return float(s.computer.estimate.velocity[s.plane])


def _tilt(s: PlaneState) -> float:
    return s.computer.estimate.tilt[s.plane]


def _tilt_rate(s: PlaneState) -> float:
    return s.computer.estimate.tilt_rate[s.plane]


def _roll(s: PlaneState) -> float:
    return s.computer.estimate.roll


def _gimbal(s: PlaneState) -> float:
    return s.gimbal_command


def _phase_flag(phase: Phase) -> Callable[[PlaneState], float]:
    return lambda s: 1.0 if s.computer.phase == phase else 0.0


def _burn_progress(s: PlaneState) -> float:
    commanded = s.computer.phases.landing_command_time
    if commanded is None or s.burn_time <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (s.time - commanded) / s.burn_time))


def _trigger_ignite(s: PlaneState) -> float:
    return 1.0 if s.computer.trigger_ignite else 0.0


def _brake(s: PlaneState) -> float:
    return s.computer.command.brake_fraction


def _time_since_liftoff(s: PlaneState) -> float:
    liftoff = s.computer.phases.liftoff_time
    return 0.0 if liftoff is None else s.time - liftoff


def _height_error(s: PlaneState) -> float:
    return s.computer.reference[0] - s.computer.height


def _vertical_speed_error(s: PlaneState) -> float:
    return s.computer.reference[1] - float(s.computer.estimate.velocity[2])


def _reference_speed(s: PlaneState) -> float:
    return s.computer.reference[1]


def _throttle(s: PlaneState) -> float:
    return s.computer.throttle


def _descent_flag(s: PlaneState) -> float:
    # Both vehicles call the phase DESCENT: the rocket's unpowered fall and the hopper's controlled descent.
    return 1.0 if s.computer.phase.value == Phase.DESCENT.value else 0.0


def _mission_phase(name: str) -> Callable[[PlaneState], float]:
    # The electric vehicle's phases (rocketsim.hop.mission.HopPhase), compared by name.
    return lambda s: 1.0 if s.computer.phase.value == name else 0.0


FIELDS: dict[str, Callable[[PlaneState], float]] = {
    "height": _height,
    "vertical_speed": _vertical_speed,
    "lateral_position": _lateral_position,
    "lateral_speed": _lateral_speed,
    "tilt": _tilt,
    "tilt_rate": _tilt_rate,
    "roll": _roll,
    "gimbal": _gimbal,
    "phase_boost": _phase_flag(Phase.BOOST),
    "phase_coast": _phase_flag(Phase.COAST),
    "phase_descent": _descent_flag,
    "phase_burn": _phase_flag(Phase.LANDING_BURN),
    "burn_progress": _burn_progress,
    "trigger_ignite": _trigger_ignite,
    "brake": _brake,
    "time_since_liftoff": _time_since_liftoff,
    "height_error": _height_error,
    "vertical_speed_error": _vertical_speed_error,
    "reference_speed": _reference_speed,
    "throttle": _throttle,
    "phase_ascent": _mission_phase("ASCENT"),
    "phase_hover": _mission_phase("HOVER"),
    "phase_landing": _mission_phase("LANDING"),
}


# What each field is, in its own unit before the training file's scale is applied (for model.json).
DESCRIPTIONS: dict[str, str] = {
    "height": "estimated height of the landing feet above the pad, m",
    "vertical_speed": "estimated vertical speed, m/s, positive up",
    "lateral_position": "estimated x (pitch plane run) or y (yaw plane run) from the pad, m",
    "lateral_speed": "estimated vx or vy, m/s",
    "tilt": "lean of the nose toward +x (pitch run) or +y (yaw run), rad",
    "tilt_rate": "rate of that lean, rad/s",
    "roll": "roll angle about the body axis, rad",
    "gimbal": "this plane's previous gimbal command from the network, rad",
    "phase_boost": "1 during the boost, else 0",
    "phase_coast": "1 during the coast, else 0",
    "phase_descent": "1 during the descent (rocket: unpowered fall; electric vehicle: controlled descent), else 0",
    "phase_burn": "1 during the landing burn, else 0",
    "burn_progress": "time since the landing igniter command over the motor burn time, 0 to 1",
    "trigger_ignite": "1 when the stopping distance table says light the landing motor now, else 0",
    "brake": "drag brake opening, 0 to 1",
    "time_since_liftoff": "seconds since liftoff",
    "height_error": "mission reference height minus estimated height, m",
    "vertical_speed_error": "mission reference vertical speed minus estimated vertical speed, m/s",
    "reference_speed": "mission reference vertical speed, m/s",
    "throttle": "throttle command of the previous step, 0 to 1",
    "phase_ascent": "1 while climbing to the target height, else 0",
    "phase_hover": "1 while holding the target height, else 0",
    "phase_landing": "1 during the final slow descent to touchdown, else 0",
}


class ObservationBuilder:
    """Turns the flight computer's state into the policy's input vector for one plane."""

    def __init__(self, fields: tuple[ObservationField, ...], burn_time: float) -> None:
        unknown = [f.name for f in fields if f.name not in FIELDS]
        if unknown:
            raise ValueError(f"unknown observation fields {unknown}; known: {sorted(FIELDS)}")
        self.fields = fields
        self.burn_time = burn_time
        self.functions = [FIELDS[f.name] for f in fields]
        self.scales = np.array([f.scale for f in fields], dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def build(self, computer: FlightComputer, plane: int, time: float, gimbal_command: float) -> np.ndarray:
        state = PlaneState(computer, plane, time, gimbal_command, self.burn_time)
        raw = np.array([function(state) for function in self.functions], dtype=np.float32)
        return raw / self.scales
