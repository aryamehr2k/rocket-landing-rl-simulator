"""Motor files, thrust curves and propellant flow.

Two formats are read: RASP .eng files as published on ThrustCurve.org and simple YAML
files with time and thrust pairs. See docs/conventions.md for the ignition semantics.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from rocketsim.units import to_si

ENG_COMMENT_PREFIX = ";"
ENG_HEADER_FIELDS = 7
ENG_NAME, ENG_DIAMETER_MM, ENG_LENGTH_MM, ENG_PROPELLANT_KG, ENG_TOTAL_KG = 0, 1, 2, 4, 5
ENG_SUFFIXES = (".eng",)
YAML_SUFFIXES = (".yaml", ".yml")


class MotorFileError(ValueError):
    """Raised when a motor file cannot be parsed or fails validation."""


@dataclass(frozen=True)
class MotorSpec:
    """A motor's static data: masses, thrust curve and derived quantities."""

    name: str
    propellant_mass: float
    total_mass: float
    times: np.ndarray
    thrusts: np.ndarray
    diameter: float = 0.0
    length: float = 0.0
    throttleable: bool = False
    throttle_lag: float = 0.0
    ignition_delay_mean: float | None = None
    ignition_delay_spread: float | None = None
    source: str = ""
    cumulative_impulse: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def __post_init__(self) -> None:
        impulse = np.concatenate(
            [[0.0], np.cumsum(np.diff(self.times) * (self.thrusts[1:] + self.thrusts[:-1]) / 2.0)]
        )
        object.__setattr__(self, "cumulative_impulse", impulse)

    @property
    def burn_time(self) -> float:
        return float(self.times[-1])

    @property
    def total_impulse(self) -> float:
        return float(self.cumulative_impulse[-1])

    @property
    def case_mass(self) -> float:
        return self.total_mass - self.propellant_mass

    @property
    def exhaust_velocity(self) -> float:
        """Effective exhaust velocity: total impulse per kilogram of propellant."""
        return self.total_impulse / self.propellant_mass

    def thrust_at(self, time_since_ignition: float) -> float:
        """Thrust from the curve, zero before ignition and after burnout."""
        if time_since_ignition < 0.0 or time_since_ignition > self.burn_time:
            return 0.0
        return float(np.interp(time_since_ignition, self.times, self.thrusts))

    def impulse_at(self, time_since_ignition: float) -> float:
        """Impulse delivered so far, clipped to the burn."""
        if time_since_ignition <= 0.0:
            return 0.0
        if time_since_ignition >= self.burn_time:
            return self.total_impulse
        return float(np.interp(time_since_ignition, self.times, self.cumulative_impulse))

    def propellant_burned_at(self, time_since_ignition: float) -> float:
        """Propellant mass burned so far, assuming mass flow follows the thrust curve."""
        return self.propellant_mass * self.impulse_at(time_since_ignition) / self.total_impulse


def load_motor(path: str | Path) -> MotorSpec:
    """Read a motor file, choosing the parser by file extension."""
    path = Path(path)
    if not path.is_file():
        raise MotorFileError(f"{path}: motor file not found")
    if path.suffix.lower() in ENG_SUFFIXES:
        return parse_eng(path.read_text(), source=str(path))
    if path.suffix.lower() in YAML_SUFFIXES:
        return parse_motor_yaml(path.read_text(), source=str(path))
    raise MotorFileError(f"{path}: unknown motor file extension, expected .eng or .yaml")


def parse_eng(text: str, source: str = "<eng>", name: str | None = None) -> MotorSpec:
    """Parse RASP .eng text. With several motors in one file, `name` selects one."""
    header: list[str] | None = None
    points: list[tuple[float, float]] = []
    motors: list[MotorSpec] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(ENG_COMMENT_PREFIX):
            continue
        if header is None:
            header = line.split()
            if len(header) < ENG_HEADER_FIELDS:
                raise MotorFileError(f"{source}: bad .eng header line: {line!r}")
            continue
        values = line.split()
        if len(values) % 2 != 0:
            raise MotorFileError(f"{source}: expected time thrust pairs, got {line!r}")
        for i in range(0, len(values), 2):
            points.append((float(values[i]), float(values[i + 1])))
        if points[-1][1] == 0.0:
            motors.append(_motor_from_eng(header, points, source))
            header, points = None, []
    if header is not None:
        raise MotorFileError(f"{source}: thrust curve for {header[ENG_NAME]} does not end with zero thrust")
    if not motors:
        raise MotorFileError(f"{source}: no motor found in .eng file")
    if name is None:
        return motors[0]
    for motor in motors:
        if motor.name == name:
            return motor
    raise MotorFileError(f"{source}: no motor named {name!r} in file")


def _motor_from_eng(header: list[str], points: list[tuple[float, float]], source: str) -> MotorSpec:
    diameter_mm, length_mm = float(header[ENG_DIAMETER_MM]), float(header[ENG_LENGTH_MM])
    propellant_mass, total_mass = float(header[ENG_PROPELLANT_KG]), float(header[ENG_TOTAL_KG])
    times, thrusts = _curve_arrays(points, source)
    return _validated(
        MotorSpec(
            name=header[ENG_NAME],
            propellant_mass=propellant_mass,
            total_mass=total_mass,
            times=times,
            thrusts=thrusts,
            diameter=to_si("diameter_mm", diameter_mm),
            length=to_si("length_mm", length_mm),
            source=source,
        )
    )


def parse_motor_yaml(text: str, source: str = "<yaml>") -> MotorSpec:
    """Parse a YAML motor description with a `thrust_curve` list of [time_s, thrust_n] pairs."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise MotorFileError(f"{source}: motor YAML must be a mapping")
    for key in ("name", "propellant_mass_g", "total_mass_g", "thrust_curve"):
        if key not in data:
            raise MotorFileError(f"{source}: missing required key {key!r}")
    curve = data["thrust_curve"]
    if not isinstance(curve, list) or any(len(p) != 2 for p in curve):
        raise MotorFileError(f"{source}: thrust_curve must be a list of [time_s, thrust_n] pairs")
    times, thrusts = _curve_arrays([(float(t), float(f)) for t, f in curve], source)
    delay = data.get("ignition_delay_s")
    delay_mean = float(delay["mean"]) if delay else None
    delay_spread = float(delay["spread"]) if delay else None
    return _validated(
        MotorSpec(
            name=str(data["name"]),
            propellant_mass=to_si("propellant_mass_g", float(data["propellant_mass_g"])),
            total_mass=to_si("total_mass_g", float(data["total_mass_g"])),
            times=times,
            thrusts=thrusts,
            diameter=to_si("diameter_mm", float(data.get("diameter_mm", 0.0))),
            length=to_si("length_mm", float(data.get("length_mm", 0.0))),
            throttleable=bool(data.get("throttleable", False)),
            throttle_lag=float(data.get("throttle_lag_s", 0.0)),
            ignition_delay_mean=delay_mean,
            ignition_delay_spread=delay_spread,
            source=source,
        )
    )


def _curve_arrays(points: list[tuple[float, float]], source: str) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        raise MotorFileError(f"{source}: thrust curve is empty")
    if points[0][0] > 0.0:
        # RASP curves start at the first sample after ignition; thrust is zero at t = 0.
        points = [(0.0, 0.0)] + points
    times = np.array([p[0] for p in points], dtype=float)
    thrusts = np.array([p[1] for p in points], dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise MotorFileError(f"{source}: thrust curve times must strictly increase")
    if np.any(thrusts < 0.0):
        raise MotorFileError(f"{source}: thrust values must not be negative")
    return times, thrusts


def _validated(motor: MotorSpec) -> MotorSpec:
    if motor.propellant_mass <= 0.0:
        raise MotorFileError(f"{motor.source}: propellant mass must be positive")
    if motor.total_mass <= motor.propellant_mass:
        raise MotorFileError(f"{motor.source}: total mass must exceed propellant mass")
    if motor.total_impulse <= 0.0:
        raise MotorFileError(f"{motor.source}: thrust curve has zero total impulse")
    if motor.throttle_lag < 0.0:
        raise MotorFileError(f"{motor.source}: throttle_lag_s must not be negative")
    if motor.ignition_delay_mean is not None and motor.ignition_delay_mean < 0.0:
        raise MotorFileError(f"{motor.source}: ignition delay mean must not be negative")
    if motor.ignition_delay_spread is not None and motor.ignition_delay_spread < 0.0:
        raise MotorFileError(f"{motor.source}: ignition delay spread must not be negative")
    return motor
