"""Rocket configuration: YAML loading, unit conversion and validation.

Values are converted to SI on load using the key suffixes described in
docs/conventions.md. Every problem raises ConfigError with the file and key path.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from rocketsim.motors import MotorFileError, MotorSpec, load_motor
from rocketsim.units import MM_PER_M, rad_to_deg
from rocketsim.yaml_section import ConfigError, Section, read_yaml_mapping

MOTOR_ROLES = ("ascent", "landing")
SERVO_NAMES = ("pitch", "yaw")
SERVO_SIGNS = (-1, 1)
MAX_GIMBAL_ANGLE_DEG = 90.0
MAX_TOUCHDOWN_TILT_DEG = 90.0
MIN_LEG_COUNT = 3
HALF = 0.5


@dataclass(frozen=True)
class AirframeConfig:
    dry_mass: float
    dry_cg: float
    dry_pitch_inertia: float
    dry_roll_inertia: float
    length: float
    reference_diameter: float


@dataclass(frozen=True)
class AeroConfig:
    drag_coefficient: float
    normal_force_slope: float
    cp: float


@dataclass(frozen=True)
class MotorConfig:
    role: str
    spec: MotorSpec
    position: float
    gimbaled: bool
    ignition_delay_mean: float
    ignition_delay_spread: float


@dataclass(frozen=True)
class ServoCalibration:
    """Gimbal angle to servo pulse width. Angles in radians, pulses in microseconds."""

    center_us: float
    pulse_us_per_rad: float
    sign: int
    linkage_ratio: float
    pulse_resolution_us: float
    min_us: float
    max_us: float

    def pulse_for_gimbal(self, gimbal_angle: float) -> float:
        """Unquantised pulse width for a gimbal angle."""
        servo_angle = gimbal_angle / self.linkage_ratio
        return self.center_us + self.sign * servo_angle * self.pulse_us_per_rad

    def quantize(self, pulse_us: float) -> float:
        """Round half up to the pulse resolution, then clip. The C firmware does the same."""
        steps = math.floor(pulse_us / self.pulse_resolution_us + HALF)
        return min(max(steps * self.pulse_resolution_us, self.min_us), self.max_us)

    def gimbal_for_pulse(self, pulse_us: float) -> float:
        """Gimbal angle the servo actually holds for a pulse width."""
        servo_angle = self.sign * (pulse_us - self.center_us) / self.pulse_us_per_rad
        return servo_angle * self.linkage_ratio


@dataclass(frozen=True)
class GimbalConfig:
    pivot: float
    max_angle: float
    servo_rate_limit: float
    servo_delay: float
    servo_deadband: float
    pitch_servo: ServoCalibration
    yaw_servo: ServoCalibration


@dataclass(frozen=True)
class LegsConfig:
    span: float
    height: float
    count: int
    max_vertical_speed: float
    max_lateral_speed: float
    max_tilt: float


@dataclass(frozen=True)
class ControlConfig:
    control_rate_hz: float
    action_delay_steps: int


@dataclass(frozen=True)
class RocketConfig:
    name: str
    airframe: AirframeConfig
    aero: AeroConfig
    motors: tuple[MotorConfig, ...]
    gimbal: GimbalConfig
    legs: LegsConfig
    control: ControlConfig
    source: str

    def motor(self, role: str) -> MotorConfig:
        for motor in self.motors:
            if motor.role == role:
                return motor
        raise KeyError(role)

    @property
    def feet_station(self) -> float:
        """Station of the leg feet plane, measured from the nose tip."""
        return self.airframe.length + self.legs.height

    @property
    def reference_area(self) -> float:
        return math.pi * self.airframe.reference_diameter ** 2 / 4.0


def load_rocket_config(path: str | Path) -> RocketConfig:
    """Load and validate a rocket YAML. Motor file paths are relative to the rocket file."""
    data, source = read_yaml_mapping(path)
    root = Section(data, source)
    root.only_keys("name", "airframe", "aero", "motors", "gimbal", "legs", "control")
    airframe = _airframe(root.sub("airframe"))
    length_mm = airframe.length * MM_PER_M
    aero_section = root.sub("aero")
    aero_section.only_keys("drag_coefficient", "normal_force_slope_per_rad", "cp_from_nose_mm")
    aero = AeroConfig(
        drag_coefficient=aero_section.number("drag_coefficient", above=0.0),
        normal_force_slope=aero_section.number("normal_force_slope_per_rad", minimum=0.0),
        cp=aero_section.number("cp_from_nose_mm", above=0.0, maximum=length_mm),
    )
    motors_section = root.sub("motors")
    motors_section.only_keys(*MOTOR_ROLES)
    motors = tuple(
        _motor(motors_section.sub(role), role, Path(source).parent, length_mm) for role in MOTOR_ROLES
    )
    gimbal = _gimbal(root.sub("gimbal"), length_mm)
    legs = _legs(root.sub("legs"))
    control_section = root.sub("control")
    control_section.only_keys("control_rate_hz", "action_delay_steps")
    control = ControlConfig(
        control_rate_hz=control_section.number("control_rate_hz", above=0.0),
        action_delay_steps=control_section.integer("action_delay_steps", minimum=0),
    )
    return RocketConfig(root.string("name"), airframe, aero, motors, gimbal, legs, control, source)


def _airframe(section: Section) -> AirframeConfig:
    section.only_keys(
        "dry_mass_g", "dry_cg_from_nose_mm", "dry_pitch_inertia_kgm2", "dry_roll_inertia_kgm2",
        "length_mm", "reference_diameter_mm",
    )
    length = section.number("length_mm", above=0.0)
    return AirframeConfig(
        dry_mass=section.number("dry_mass_g", above=0.0),
        dry_cg=section.number("dry_cg_from_nose_mm", above=0.0, maximum=length * MM_PER_M),
        dry_pitch_inertia=section.number("dry_pitch_inertia_kgm2", above=0.0),
        dry_roll_inertia=section.number("dry_roll_inertia_kgm2", above=0.0),
        length=length,
        reference_diameter=section.number("reference_diameter_mm", above=0.0),
    )


def _motor(section: Section, role: str, base_dir: Path, length_mm: float) -> MotorConfig:
    section.only_keys("file", "position_from_nose_mm", "gimbaled", "ignition_delay_s")
    motor_path = base_dir / section.string("file")
    try:
        spec = load_motor(motor_path)
    except MotorFileError as error:
        raise ConfigError(f"{section.where('file')}: {error}") from error
    if section.has("ignition_delay_s"):
        delay = section.sub("ignition_delay_s")
        delay.only_keys("mean", "spread")
        mean = delay.number("mean", minimum=0.0)
        spread = delay.number("spread", minimum=0.0)
    elif spec.ignition_delay_mean is not None and spec.ignition_delay_spread is not None:
        mean, spread = spec.ignition_delay_mean, spec.ignition_delay_spread
    else:
        raise ConfigError(
            f"{section.where('ignition_delay_s')} is required because the motor file has no default"
        )
    return MotorConfig(
        role=role,
        spec=spec,
        position=section.number("position_from_nose_mm", above=0.0, maximum=length_mm),
        gimbaled=section.boolean("gimbaled", default=True),
        ignition_delay_mean=mean,
        ignition_delay_spread=spread,
    )


def _servo(cal: Section, max_angle: float) -> ServoCalibration:
    cal.only_keys(
        "center_us", "pulse_us_per_deg", "sign", "linkage_ratio", "pulse_resolution_us", "min_us", "max_us"
    )
    sign = cal.integer("sign")
    if sign not in SERVO_SIGNS:
        raise ConfigError(f"{cal.where('sign')} must be 1 or -1, got {sign}")
    calibration = ServoCalibration(
        center_us=cal.number("center_us", above=0.0),
        pulse_us_per_rad=cal.number("pulse_us_per_deg", above=0.0),
        sign=sign,
        linkage_ratio=cal.number("linkage_ratio", above=0.0),
        pulse_resolution_us=cal.number("pulse_resolution_us", above=0.0),
        min_us=cal.number("min_us", above=0.0),
        max_us=cal.number("max_us", above=0.0),
    )
    if not calibration.min_us < calibration.center_us < calibration.max_us:
        raise ConfigError(f"{cal.where('center_us')} must lie between min_us and max_us")
    for key in ("center_us", "min_us", "max_us"):
        if getattr(calibration, key) % calibration.pulse_resolution_us != 0.0:
            raise ConfigError(f"{cal.where(key)} must be a multiple of pulse_resolution_us")
    for angle in (-max_angle, max_angle):
        pulse = calibration.pulse_for_gimbal(angle)
        if not calibration.min_us <= pulse <= calibration.max_us:
            raise ConfigError(
                f"{cal.where('min_us')}: gimbal angle {rad_to_deg(angle):.1f} deg needs a pulse of "
                f"{pulse:.0f} us, outside min_us..max_us"
            )
    return calibration


def _gimbal(section: Section, length_mm: float) -> GimbalConfig:
    section.only_keys(
        "pivot_from_nose_mm", "max_angle_deg", "servo_rate_limit_deg_per_s", "servo_delay_s",
        "servo_deadband_deg", "servos",
    )
    max_angle = section.number("max_angle_deg", above=0.0, maximum=MAX_GIMBAL_ANGLE_DEG)
    servos = section.sub("servos")
    servos.only_keys(*SERVO_NAMES)
    gimbal = GimbalConfig(
        pivot=section.number("pivot_from_nose_mm", above=0.0, maximum=length_mm),
        max_angle=max_angle,
        servo_rate_limit=section.number("servo_rate_limit_deg_per_s", above=0.0),
        servo_delay=section.number("servo_delay_s", minimum=0.0),
        servo_deadband=section.number("servo_deadband_deg", minimum=0.0),
        pitch_servo=_servo(servos.sub("pitch"), max_angle),
        yaw_servo=_servo(servos.sub("yaw"), max_angle),
    )
    if gimbal.servo_deadband >= max_angle:
        raise ConfigError(f"{section.where('servo_deadband_deg')} must be smaller than max_angle_deg")
    return gimbal


def _legs(section: Section) -> LegsConfig:
    section.only_keys(
        "span_mm", "height_mm", "count", "max_touchdown_vertical_speed_mps",
        "max_touchdown_lateral_speed_mps", "max_touchdown_tilt_deg",
    )
    return LegsConfig(
        span=section.number("span_mm", above=0.0),
        height=section.number("height_mm", minimum=0.0),
        count=section.integer("count", minimum=MIN_LEG_COUNT),
        max_vertical_speed=section.number("max_touchdown_vertical_speed_mps", above=0.0),
        max_lateral_speed=section.number("max_touchdown_lateral_speed_mps", above=0.0),
        max_tilt=section.number("max_touchdown_tilt_deg", above=0.0, maximum=MAX_TOUCHDOWN_TILT_DEG),
    )
