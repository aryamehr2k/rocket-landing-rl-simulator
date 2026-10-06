"""Electric hopper vehicle file: airframe, motor, gimbal, roll control, legs, sensors and controller gains.

The physical part is a RocketConfig with one motor, so the six degree of freedom physics,
sensors, servos and touchdown grading are shared with the solid rocket.
"""

from dataclasses import dataclass
from pathlib import Path

from rocketsim.config import RocketConfig, _airframe, _gimbal, _legs, _motor
from rocketsim.config import AeroConfig, ControlConfig
from rocketsim.guidance_config import POLICY, PID, EstimatorConfig, PidConfig, _estimator, _pid
from rocketsim.sensors import SensorsConfig, load_sensors_config
from rocketsim.units import MM_PER_M
from rocketsim.yaml_section import ConfigError, Section, read_yaml_mapping

MAIN = "main"
OWNERS = (PID, POLICY)
DIRECT, RESIDUAL = "direct", "residual"
MODES = (DIRECT, RESIDUAL)
MAX_ANGLE_DEG = 90.0


@dataclass(frozen=True)
class RollControlConfig:
    max_torque: float
    time_constant: float
    kp: float
    kd: float


@dataclass(frozen=True)
class AltitudePidConfig:
    height_gain: float
    max_speed_correction: float
    speed_gain: float
    speed_integral_gain: float
    max_integral_accel: float


@dataclass(frozen=True)
class HopControllers:
    """Who flies the vehicle: the PID or a trained policy, separately for steering and throttle."""

    steering: str = PID
    throttle: str = PID
    mode: str = DIRECT  # direct: the policy's command is used; residual: it is added to the PID's


@dataclass(frozen=True)
class HopSafetyConfig:
    max_gimbal_rate: float
    max_throttle_rate: float
    abort_tilt: float
    geofence_radius: float


@dataclass(frozen=True)
class HopVehicleConfig:
    body: RocketConfig
    sensors: SensorsConfig
    estimator: EstimatorConfig
    attitude: PidConfig
    altitude: AltitudePidConfig
    roll: RollControlConfig
    controllers: HopControllers
    safety: HopSafetyConfig
    source: str

    @property
    def name(self) -> str:
        return self.body.name

    @property
    def motor(self):
        return self.body.motor(MAIN)

    @property
    def mass(self) -> float:
        return self.body.airframe.dry_mass + self.motor.spec.total_mass

    @property
    def max_thrust(self) -> float:
        return self.motor.spec.max_thrust


def is_vehicle_file(path: str | Path) -> bool:
    """True for an electric vehicle YAML (it has a single `motor`), False for a rocket YAML (`motors`)."""
    data, _ = read_yaml_mapping(path)
    return "motor" in data and "motors" not in data


def load_vehicle_config(path: str | Path) -> HopVehicleConfig:
    """Load and validate an electric vehicle YAML. File paths inside it are relative to it."""
    data, source = read_yaml_mapping(path)
    root = Section(data, source)
    root.only_keys(
        "name", "airframe", "aero", "motor", "gimbal", "roll_control", "legs", "control", "sensors", "estimator",
        "attitude_pid", "altitude_pid", "controllers", "safety",
    )
    base = Path(source).parent
    airframe = _airframe(root.sub("airframe"))
    length_mm = airframe.length * MM_PER_M
    aero = root.sub("aero")
    aero.only_keys("drag_coefficient", "normal_force_slope_per_rad", "cp_from_nose_mm")
    motor_section = root.sub("motor")
    motor = _motor(motor_section, MAIN, base, length_mm)
    if not motor.spec.electric:
        raise ConfigError(f"{motor_section.where('file')} must be an electric motor (type: electric)")
    control = root.sub("control")
    control.only_keys("control_rate_hz", "action_delay_steps")
    body = RocketConfig(
        name=root.string("name"),
        airframe=airframe,
        aero=AeroConfig(
            drag_coefficient=aero.number("drag_coefficient", above=0.0),
            normal_force_slope=aero.number("normal_force_slope_per_rad", minimum=0.0),
            cp=aero.number("cp_from_nose_mm", above=0.0, maximum=length_mm),
        ),
        motors=(motor,),
        gimbal=_gimbal(root.sub("gimbal"), length_mm),
        legs=_legs(root.sub("legs")),
        control=ControlConfig(
            control_rate_hz=control.number("control_rate_hz", above=0.0),
            action_delay_steps=control.integer("action_delay_steps", minimum=0),
        ),
        computer=None,
        source=source,
    )
    sensors = root.sub("sensors")
    sensors.only_keys("file")
    try:
        sensors_config = load_sensors_config(base / sensors.string("file"))
    except ConfigError as error:
        raise ConfigError(f"{sensors.where('file')}: {error}") from error
    vehicle = HopVehicleConfig(
        body=body,
        sensors=sensors_config,
        estimator=_estimator(root.sub("estimator")),
        attitude=_pid(root.sub("attitude_pid")),
        altitude=_altitude(root.sub("altitude_pid")),
        roll=_roll(root.sub("roll_control")),
        controllers=load_hop_controllers(root.sub("controllers")),
        safety=_safety(root.sub("safety")),
        source=source,
    )
    weight = vehicle.mass * 9.80665
    if vehicle.max_thrust <= weight:
        raise ConfigError(
            f"{source}: the motor's {vehicle.max_thrust:.1f} N cannot lift the {vehicle.mass:.2f} kg vehicle "
            f"({weight:.1f} N); check motor.max_thrust_n and the masses"
        )
    return vehicle


def load_hop_controllers(section: Section) -> HopControllers:
    """`steering` and `throttle`: pid or policy; `mode`: direct or residual."""
    section.only_keys("steering", "throttle", "mode")
    steering = section.string("steering", default=PID)
    throttle = section.string("throttle", default=PID)
    mode = section.string("mode", default=DIRECT)
    for key, value, choices in (("steering", steering, OWNERS), ("throttle", throttle, OWNERS), ("mode", mode, MODES)):
        if value not in choices:
            raise ConfigError(f"{section.where(key)} must be one of {choices}, got {value!r}")
    return HopControllers(steering, throttle, mode)


def _altitude(section: Section) -> AltitudePidConfig:
    section.only_keys(
        "height_gain_per_s", "max_speed_correction_mps", "speed_gain_per_s", "speed_integral_gain_per_s2",
        "max_integral_accel_mps2",
    )
    return AltitudePidConfig(
        height_gain=section.number("height_gain_per_s", minimum=0.0),
        max_speed_correction=section.number("max_speed_correction_mps", minimum=0.0),
        speed_gain=section.number("speed_gain_per_s", above=0.0),
        speed_integral_gain=section.number("speed_integral_gain_per_s2", minimum=0.0),
        max_integral_accel=section.number("max_integral_accel_mps2", minimum=0.0),
    )


def _roll(section: Section) -> RollControlConfig:
    section.only_keys("max_torque_nm", "time_constant_s", "kp_nm_per_rad", "kd_nm_s_per_rad")
    return RollControlConfig(
        max_torque=section.number("max_torque_nm", minimum=0.0),
        time_constant=section.number("time_constant_s", minimum=0.0),
        kp=section.number("kp_nm_per_rad", minimum=0.0),
        kd=section.number("kd_nm_s_per_rad", minimum=0.0),
    )


def _safety(section: Section) -> HopSafetyConfig:
    section.only_keys("max_gimbal_rate_deg_per_s", "max_throttle_rate_per_s", "abort_tilt_deg", "geofence_radius_m")
    return HopSafetyConfig(
        max_gimbal_rate=section.number("max_gimbal_rate_deg_per_s", above=0.0),
        max_throttle_rate=section.number("max_throttle_rate_per_s", above=0.0),
        abort_tilt=section.number("abort_tilt_deg", above=0.0, maximum=MAX_ANGLE_DEG),
        geofence_radius=section.number("geofence_radius_m", above=0.0),
    )
