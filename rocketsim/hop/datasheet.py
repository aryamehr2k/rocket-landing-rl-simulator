"""Read an electric vehicle data sheet and work out what the config files need beyond its own numbers.

The sheet is in the units people measure (g, mm, ms, deg); everything in `Derived` is SI.
"""

import difflib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from rocketsim.hop.datasheet_checks import leg_touchdown_speed, sheet_problems
from rocketsim.hop.datasheet_fields import GROUPS, INTEGER, TEXT, Entry, find_entry
from rocketsim.hop.mission import MissionConfig
from rocketsim.simconfig import SimConfig
from rocketsim.units import G_PER_KG, MM_PER_M, MS_PER_S, deg_to_rad
from rocketsim.yaml_section import ConfigError

ATTITUDE_FREQUENCY = 6.0  # rad/s, natural frequency of the closed tilt loop (about 1 Hz)
ATTITUDE_DAMPING = 0.6
ROLL_FREQUENCY = 10.0  # rad/s, the roll loop's frequency when the roll actuator is fast enough for it
ROLL_DAMPING = 0.5
ROLL_LINEAR_ANGLE_DEG = 15.0  # roll gain is capped so the torque saturates no earlier than this error
HELD_COMMAND_DELAY = 0.5  # control periods: a command held for a period acts on average half a period late
BISECTION_STEPS = 50
HORN_TRAVEL_DEG = 60.0  # servo speeds are quoted as the time for 60 degrees of horn travel
FLIGHT_TIME_MARGIN_S = 40.0  # flight time limit beyond the planned mission
FLIGHT_TIME_STEP_S = 10.0  # the limit is rounded up to a multiple of this
HALF = 0.5
SOLID_CYLINDER_PITCH_DIVISOR = 12.0
TO_SI = {"g": 1.0 / G_PER_KG, "mm": 1.0 / MM_PER_M, "ms": 1.0 / MS_PER_S, "deg": deg_to_rad(1.0)}
YAML_HINT = "each entry needs a space after its colon, and text holding ':' or '#' needs quotes"


class DatasheetError(ConfigError):
    """A data sheet with empty required entries or bad values; the message lists every one."""


@dataclass(frozen=True)
class Datasheet:
    values: dict[str, dict[str, Any]]  # group -> entry -> value in the sheet's unit, defaults filled in
    source: str

    def __getitem__(self, group: str) -> dict[str, Any]:
        return self.values[group]

    def si(self, group: str, key: str) -> float:
        """An entry converted from its sheet unit to SI."""
        return float(self.values[group][key]) * TO_SI.get(find_entry(group, key).unit, 1.0)

    @property
    def has_gps(self) -> bool:
        return self.values["sensors"]["gps_rate_hz"] > 0.0


@dataclass(frozen=True)
class Derived:
    """What the config files need beyond the sheet's own numbers, in SI units."""

    mass: float  # kg, ready to fly
    dry_mass: float  # kg, everything but the fan unit
    cg: float  # m from the top, ready to fly
    dry_cg: float  # m from the top, without the fan unit
    pitch_inertia: float  # kg m2, whole vehicle about its CG
    roll_inertia: float  # kg m2, whole vehicle about its long axis
    dry_pitch_inertia: float  # kg m2, without the fan unit, about the dry CG
    dry_roll_inertia: float  # kg m2, without the fan unit
    lever_arm: float  # m from the CG down to the gimbal pivot
    authority: float  # rad/s2 of pitch acceleration per rad of gimbal at hover thrust
    attitude_kp: float  # rad of gimbal per rad of tilt error
    attitude_kd: float  # rad of gimbal per rad/s of tilt rate
    roll_kp: float  # N m per rad
    roll_kd: float  # N m per rad/s
    roll_frequency: float  # rad/s the roll loop reaches within the torque limit and the roll lag
    roll_lag: float  # s, roll actuator response plus the command delay
    gimbal_rate: float  # rad/s of thrust angle the servos can move
    touchdown_speed: float  # m/s the legs survive, from the drop height
    centre_of_pressure: float  # m from the top
    mission: MissionConfig  # the flight plan, with the flight time limit filled in


def load_datasheet(path: str | Path) -> Datasheet:
    """Read a data sheet, fill in the defaults of empty optional entries and report every problem at once."""
    path = Path(path)
    if not path.is_file():
        raise DatasheetError(f"{path}: file not found")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        line = f" line {mark.line + 1}" if mark is not None else ""
        problem = getattr(error, "problem", None) or "not readable"
        raise DatasheetError(f"{path}{line}: {problem}; {YAML_HINT}") from error
    if not isinstance(data, dict):
        raise DatasheetError(f"{path}: the sheet must be groups of named entries")
    missing: list[str] = []
    groups = [group.key for group in GROUPS]
    problems = [_unknown(str(key), groups, "group") for key in data if key not in groups]
    values: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        given = data.get(group.key) or {}
        if not isinstance(given, dict):
            problems.append(f"{group.key}: must be a group of named entries")
            given = {}
        names = [entry.key for entry in group.entries]
        problems += [_unknown(f"{group.key}.{key}", names, "entry") for key in given if key not in names]
        values[group.key] = {}
        for entry in group.entries:
            raw = given.get(entry.key)
            if raw is None:
                if entry.required:
                    unit = f" [{entry.unit}]" if entry.unit else ""
                    missing.append(f"{group.key}.{entry.key}: {entry.what}{unit}")
                values[group.key][entry.key] = entry.default
                continue
            value, problem = _parse(entry, raw)
            if problem:
                problems.append(f"{group.key}.{entry.key} {problem}")
            values[group.key][entry.key] = value
    if missing or problems:
        raise DatasheetError(_report(path, missing, problems))
    return Datasheet(values, str(path))


def derive(sheet: Datasheet, sim: SimConfig) -> Derived:
    """Masses, inertias, gains and limits for the simulation settings `sim`. Raises DatasheetError on contradictions."""
    mission = planned_mission(sheet)
    problems = sheet_problems(sheet.values, mission, sim)
    if problems:
        raise DatasheetError(_report(Path(sheet.source), [], problems))
    gravity = sim.environment.gravity
    mass, fan_mass = sheet.si("vehicle", "total_mass_g"), sheet.si("fan", "unit_mass_g")
    cg, fan_position = sheet.si("vehicle", "balance_point_mm"), sheet.si("fan", "position_mm")
    length, radius = sheet.si("vehicle", "length_mm"), HALF * sheet.si("vehicle", "diameter_mm")
    dry_mass = mass - fan_mass
    dry_cg = (mass * cg - fan_mass * fan_position) / dry_mass
    # Parallel axis terms that move the dry inertia and the fan unit to the whole vehicle's CG.
    offsets = dry_mass * (dry_cg - cg) ** 2 + fan_mass * (fan_position - cg) ** 2
    fan_roll = HALF * fan_mass * (HALF * sheet.si("fan", "diameter_mm")) ** 2
    if sheet["vehicle"]["pitch_inertia_kgm2"] is None:
        dry_pitch = solid_cylinder_pitch_inertia(dry_mass, radius, length)
    else:
        dry_pitch = sheet["vehicle"]["pitch_inertia_kgm2"] - offsets
    if sheet["vehicle"]["roll_inertia_kgm2"] is None:
        dry_roll = solid_cylinder_roll_inertia(dry_mass, radius)
    else:
        dry_roll = sheet["vehicle"]["roll_inertia_kgm2"] - fan_roll
    if dry_pitch <= 0.0 or dry_roll <= 0.0:
        raise DatasheetError(
            f"{sheet.source}: vehicle.pitch_inertia_kgm2 or roll_inertia_kgm2 is smaller than the fan unit's share "
            f"alone ({offsets:.4f} and {fan_roll:.5f} kg m2); check the inertias and the fan position"
        )
    pitch, roll = dry_pitch + offsets, dry_roll + fan_roll
    lever_arm = sheet.si("thrust_vectoring", "pivot_mm") - cg
    authority = mass * gravity * lever_arm / pitch
    kp, kd = attitude_gains(authority)
    sensors = sheet["sensors"]
    roll_lag = sheet.si("roll_control", "response_ms") + (
        sensors["command_delay_steps"] + HELD_COMMAND_DELAY) / sensors["control_rate_hz"]
    roll_kp, roll_kd, roll_frequency = roll_gains(roll, sheet["roll_control"]["max_torque_nm"], roll_lag)
    vectoring = sheet["thrust_vectoring"]
    servo_rate = deg_to_rad(HORN_TRAVEL_DEG) / vectoring["servo_time_per_60deg_s"]
    cp = sheet["vehicle"]["centre_of_pressure_mm"]
    return Derived(
        mass=mass, dry_mass=dry_mass, cg=cg, dry_cg=dry_cg, pitch_inertia=pitch, roll_inertia=roll,
        dry_pitch_inertia=dry_pitch, dry_roll_inertia=dry_roll, lever_arm=lever_arm, authority=authority,
        attitude_kp=kp, attitude_kd=kd, roll_kp=roll_kp, roll_kd=roll_kd, roll_frequency=roll_frequency,
        roll_lag=roll_lag, gimbal_rate=servo_rate * vectoring["linkage_ratio"],
        touchdown_speed=leg_touchdown_speed(sheet["legs"]["max_drop_height_mm"], gravity),
        centre_of_pressure=HALF * length if cp is None else cp / MM_PER_M, mission=mission,
    )


def solid_cylinder_pitch_inertia(mass: float, radius: float, length: float) -> float:
    """Inertia of a solid cylinder about a diameter through its middle."""
    return mass * (3.0 * radius ** 2 + length ** 2) / SOLID_CYLINDER_PITCH_DIVISOR


def solid_cylinder_roll_inertia(mass: float, radius: float) -> float:
    """Inertia of a solid cylinder about its axis."""
    return HALF * mass * radius ** 2


def attitude_gains(authority: float) -> tuple[float, float]:
    """Tilt PID kp and kd (s) that put the closed loop at ATTITUDE_FREQUENCY with ATTITUDE_DAMPING."""
    return ATTITUDE_FREQUENCY ** 2 / authority, 2.0 * ATTITUDE_DAMPING * ATTITUDE_FREQUENCY / authority


def roll_gains(inertia: float, max_torque: float, lag: float) -> tuple[float, float, float]:
    """Roll kp (N m/rad), kd (N m s/rad) and the frequency of the roll loop they give.

    The loop is the roll inertia behind a first order lag (actuator response plus command delay). The gains
    place a pole pair with ROLL_DAMPING and a real pole, the pair no faster than the lag allows and kp no
    larger than the torque limit allows.
    """
    frequency = roll_frequency_limit(lag)
    kp_limit = max_torque / deg_to_rad(ROLL_LINEAR_ANGLE_DEG)
    if _roll_kp(inertia, lag, frequency) > kp_limit:
        low, high = 0.0, frequency
        for _ in range(BISECTION_STEPS):
            middle = HALF * (low + high)
            low, high = (middle, high) if _roll_kp(inertia, lag, middle) <= kp_limit else (low, middle)
        frequency = low
    real_pole = 1.0 / lag - 2.0 * ROLL_DAMPING * frequency
    kd = inertia * lag * (frequency ** 2 + 2.0 * ROLL_DAMPING * frequency * real_pole)
    return _roll_kp(inertia, lag, frequency), kd, frequency


def roll_frequency_limit(lag: float) -> float:
    """Fastest roll loop for a roll lag: ROLL_FREQUENCY, or the frequency that gives the stiffest loop for the lag."""
    return min(ROLL_FREQUENCY, 1.0 / (3.0 * ROLL_DAMPING * lag))


def planned_mission(sheet: Datasheet, name: str = "") -> MissionConfig:
    """The mission from the sheet; an empty flight time limit is the plan plus a margin, rounded up."""
    m = sheet["mission"]
    mission = MissionConfig(
        name=name, target_altitude=m["target_height_m"], hover_time=m["hover_time_s"],
        hover_margin=m["extra_hover_s"], climb_speed=m["climb_speed_mps"],
        climb_acceleration=m["climb_acceleration_mps2"], descent_speed=m["descent_speed_mps"],
        final_height=m["final_height_m"], final_speed=m["final_speed_mps"],
        altitude_tolerance=m["height_tolerance_m"], landing_radius=m["landing_radius_m"],
        max_flight_time=m["max_flight_time_s"] or 0.0, source=sheet.source,
    )
    if m["max_flight_time_s"] is None:
        steps = math.ceil((mission.planned_duration + FLIGHT_TIME_MARGIN_S) / FLIGHT_TIME_STEP_S)
        mission = replace(mission, max_flight_time=steps * FLIGHT_TIME_STEP_S)
    return mission


def _roll_kp(inertia: float, lag: float, frequency: float) -> float:
    # Matching I lag s^3 + I s^2 + kd s + kp to (s + p)(s^2 + 2 zeta w s + w^2) with p = 1/lag - 2 zeta w.
    return inertia * frequency ** 2 * (1.0 - 2.0 * ROLL_DAMPING * frequency * lag)


def _parse(entry: Entry, raw: Any) -> tuple[Any, str]:
    """The value in its type, and what is wrong with it ('' when nothing is)."""
    if entry.kind == TEXT:
        return " ".join(str(raw).split()), ""  # one line, since it ends up in a YAML comment
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return raw, f"must be a number, got {raw!r}"
    if entry.kind == INTEGER and not float(raw).is_integer():
        return raw, f"must be a whole number, got {raw!r}"
    value = int(raw) if entry.kind == INTEGER else float(raw)
    if entry.choices and value not in entry.choices:
        return value, f"must be one of {', '.join(str(c) for c in entry.choices)}, got {raw!r}"
    if entry.minimum is not None and (value < entry.minimum or (value == entry.minimum and not entry.inclusive)):
        relation = "at least" if entry.inclusive else "more than"
        return value, f"must be {relation} {entry.minimum:g}, got {raw!r}"
    if entry.maximum is not None and value > entry.maximum:
        return value, f"must be at most {entry.maximum:g}, got {raw!r}"
    return value, ""


def _unknown(name: str, known: list[str], kind: str) -> str:
    close = difflib.get_close_matches(name.rsplit(".", 1)[-1], known, n=1)
    return f"{name} is not a known {kind}" + (f" (did you mean {close[0]}?)" if close else "")


def _report(path: Path, missing: list[str], problems: list[str]) -> str:
    lines = [f"{path} is not ready:"]
    if missing:
        lines.append(f"  {len(missing)} required {'entry is' if len(missing) == 1 else 'entries are'} empty:")
        lines += [f"    {item}" for item in missing]
    if problems:
        lines.append(f"  {len(problems)} entries need fixing:" if len(problems) > 1 else "  1 entry needs fixing:")
        lines += [f"    {item}" for item in problems]
    return "\n".join(lines)
