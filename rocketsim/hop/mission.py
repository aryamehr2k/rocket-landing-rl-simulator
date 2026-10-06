"""The flight plan: climb to a target height, hover, descend and land inside a radius.

`Guidance` turns the mission file into a reference height and vertical speed every control
step, which the controller (PID or policy) follows. `MissionScore` grades a flight on the true
state against the mission's pass criteria. The firmware runs the same guidance in C.
"""

import math
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from rocketsim.yaml_section import Section, read_yaml_mapping

ARRIVAL_DISTANCE = 0.05  # m below the target where the climb reference snaps to the target
TIME_TOLERANCE = 1e-6  # s; a time summed from many control steps is not exact, so comparisons allow this much


class HopPhase(str, Enum):
    PAD = "PAD"
    ASCENT = "ASCENT"
    HOVER = "HOVER"
    DESCENT = "DESCENT"
    LANDING = "LANDING"
    LANDED = "LANDED"
    ABORT = "ABORT"


FLYING_PHASES = (HopPhase.ASCENT, HopPhase.HOVER, HopPhase.DESCENT, HopPhase.LANDING)


@dataclass(frozen=True)
class MissionConfig:
    name: str
    target_altitude: float
    hover_time: float
    hover_margin: float
    climb_speed: float
    climb_acceleration: float
    descent_speed: float
    final_height: float
    final_speed: float
    altitude_tolerance: float
    landing_radius: float
    max_flight_time: float
    source: str

    @property
    def planned_duration(self) -> float:
        """Rough length of the plan from launch to touchdown, for scaling observations."""
        climb = self.target_altitude / self.climb_speed + self.climb_speed / self.climb_acceleration
        descent = (self.target_altitude - self.final_height) / self.descent_speed + self.final_height / self.final_speed
        return climb + self.hover_time + self.hover_margin + descent


def load_mission_config(path: str | Path) -> MissionConfig:
    """Load and validate a mission YAML."""
    data, source = read_yaml_mapping(path)
    root = Section(data, source)
    root.only_keys("name", "ascent", "hover", "descent", "criteria", "max_flight_time_s")
    ascent, hover, descent, criteria = root.sub("ascent"), root.sub("hover"), root.sub("descent"), root.sub("criteria")
    ascent.only_keys("target_altitude_m", "climb_speed_mps", "climb_acceleration_mps2")
    hover.only_keys("time_s", "extra_time_s")
    descent.only_keys("descent_speed_mps", "final_height_m", "final_speed_mps")
    criteria.only_keys("altitude_tolerance_m", "landing_radius_m")
    return MissionConfig(
        name=root.string("name"),
        target_altitude=ascent.number("target_altitude_m", above=0.0),
        hover_time=hover.number("time_s", minimum=0.0),
        hover_margin=hover.number("extra_time_s", minimum=0.0),
        climb_speed=ascent.number("climb_speed_mps", above=0.0),
        climb_acceleration=ascent.number("climb_acceleration_mps2", above=0.0),
        descent_speed=descent.number("descent_speed_mps", above=0.0),
        final_height=descent.number("final_height_m", above=0.0),
        final_speed=descent.number("final_speed_mps", above=0.0),
        altitude_tolerance=criteria.number("altitude_tolerance_m", above=0.0),
        landing_radius=criteria.number("landing_radius_m", above=0.0),
        max_flight_time=root.number("max_flight_time_s", above=0.0),
        source=source,
    )


def with_targets(mission: MissionConfig, altitude: float | None = None, hover_time: float | None = None) -> MissionConfig:
    """The mission with a different target height or hover time."""
    if altitude is not None:
        mission = replace(mission, target_altitude=altitude, final_height=min(mission.final_height, altitude / 2))
    if hover_time is not None:
        mission = replace(mission, hover_time=hover_time)
    return mission


class Guidance:
    """Reference height and vertical speed for the feet, stepped once per control step."""

    def __init__(self, mission: MissionConfig) -> None:
        self.mission = mission
        self.reset()

    def reset(self) -> None:
        self.phase = HopPhase.PAD
        self.height = 0.0
        self.speed = 0.0
        self.phase_start = 0.0
        self.launch_time: float | None = None

    def start(self, t: float) -> None:
        self.launch_time = t
        self._enter(HopPhase.ASCENT, t)

    def hover_time_left(self, t: float) -> float:
        m = self.mission
        if self.phase == HopPhase.ASCENT:
            return m.hover_time + m.hover_margin
        if self.phase == HopPhase.HOVER:
            return max(0.0, m.hover_time + m.hover_margin - (t - self.phase_start))
        return 0.0

    def update(self, t: float, dt: float) -> tuple[HopPhase, float, float]:
        """Advance the reference by one control step. Returns the phase, reference height and speed."""
        m = self.mission
        if self.phase == HopPhase.ASCENT:
            room = max(m.target_altitude - self.height, 0.0)
            self.speed = min(m.climb_speed, self.speed + m.climb_acceleration * dt, math.sqrt(2.0 * m.climb_acceleration * room))
            self.height += self.speed * dt
            if m.target_altitude - self.height <= ARRIVAL_DISTANCE:
                self.height, self.speed = m.target_altitude, 0.0
                self._enter(HopPhase.HOVER, t)
        elif self.phase == HopPhase.HOVER:
            if t - self.phase_start + TIME_TOLERANCE >= m.hover_time + m.hover_margin:
                self._enter(HopPhase.DESCENT, t)
        elif self.phase == HopPhase.DESCENT:
            self.speed = max(-m.descent_speed, self.speed - m.climb_acceleration * dt)
            self.height += self.speed * dt
            if self.height <= m.final_height:
                self._enter(HopPhase.LANDING, t)
        elif self.phase == HopPhase.LANDING:
            # Slow down to the final speed and keep going until the legs touch.
            self.speed = min(-m.final_speed, self.speed + m.climb_acceleration * dt)
            self.height = max(0.0, self.height + self.speed * dt)
        return self.phase, self.height, self.speed

    def _enter(self, phase: HopPhase, t: float) -> None:
        self.phase = phase
        self.phase_start = t


@dataclass(frozen=True)
class MissionResult:
    reached_altitude: bool
    max_height: float
    hover_held: float
    hover_ok: bool
    landed: bool
    miss_distance: float
    inside_radius: bool
    touchdown_speed: float
    aborted: bool

    @property
    def success(self) -> bool:
        return self.reached_altitude and self.hover_ok and self.landed and self.inside_radius and not self.aborted

    def summary(self) -> str:
        def mark(ok: bool) -> str:
            return "pass" if ok else "FAIL"
        return (
            f"altitude {self.max_height:.1f} m ({mark(self.reached_altitude)}), hover held {self.hover_held:.1f} s "
            f"({mark(self.hover_ok)}), landing {mark(self.landed)} at {self.touchdown_speed:.2f} m/s, "
            f"{self.miss_distance:.1f} m from the pad ({mark(self.inside_radius)})"
            + ("; ABORTED" if self.aborted else "")
        )


class MissionScore:
    """Tracks a flight's true height and position against the mission criteria."""

    def __init__(self, mission: MissionConfig) -> None:
        self.mission = mission
        self.reset()

    def reset(self) -> None:
        self.max_height = 0.0
        self.streak_start: float | None = None
        self.best_hold = 0.0

    def update(self, t: float, height: float) -> None:
        m = self.mission
        self.max_height = max(self.max_height, height)
        if abs(height - m.target_altitude) <= m.altitude_tolerance:
            if self.streak_start is None:
                self.streak_start = t
            self.best_hold = max(self.best_hold, t - self.streak_start)
        else:
            self.streak_start = None

    def result(self, landed: bool, miss_distance: float, touchdown_speed: float, aborted: bool) -> MissionResult:
        m = self.mission
        return MissionResult(
            reached_altitude=self.max_height >= m.target_altitude - m.altitude_tolerance,
            max_height=self.max_height,
            hover_held=self.best_hold,
            hover_ok=self.best_hold >= m.hover_time,
            landed=landed,
            miss_distance=miss_distance,
            inside_radius=miss_distance <= m.landing_radius,
            touchdown_speed=touchdown_speed,
            aborted=aborted,
        )
