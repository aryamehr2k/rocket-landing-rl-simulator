"""Flight computer settings from the rocket YAML: sensors, estimator, phases, trigger, PID, safety.

These sections describe what runs on the board, so one rocket file holds everything the
firmware needs. rocketsim.config calls load_flight_computer_config while loading a rocket.
"""

from dataclasses import dataclass
from pathlib import Path

from rocketsim.dragdevice import FULLY_OPEN
from rocketsim.sensors import SensorsConfig, load_sensors_config
from rocketsim.yaml_section import ConfigError, Section

SECTION_KEYS = ("sensors", "estimator", "phases", "landing_trigger", "pid", "safety", "brake", "controllers")
PID, POLICY, TRIGGER = "pid", "policy", "trigger"
STEERING_CHOICES = (PID, POLICY)
IGNITION_CHOICES = (TRIGGER, POLICY)
MAX_TILT_DEG = 90.0
MAX_GAIN_FRACTION = 1.0
MAX_THRUST_MARGIN = 1.10  # the table may assume a motor up to this much stronger than the curve


@dataclass(frozen=True)
class EstimatorConfig:
    pad_average_time: float
    baro_altitude_gain: float
    baro_velocity_gain: float
    gps_position_gain: float = 0.0
    gps_velocity_gain: float = 0.0


@dataclass(frozen=True)
class PhaseConfig:
    liftoff_accel: float
    burnout_accel: float
    min_boost_time: float
    apogee_vz: float
    abort_tilt: float


@dataclass(frozen=True)
class LandingTriggerConfig:
    target_speed: float
    target_height: float
    thrust_margin: float
    calibrate_drag_in_flight: bool = False


@dataclass(frozen=True)
class BrakeConfig:
    """Flight computer rules for the drag device: when to open it, when to shut it, how far to hold it in the burn."""

    deploy_descent_speed: float
    retract_descent_speed: float
    burn_fraction: float


@dataclass(frozen=True)
class PidGains:
    kp: float
    ki: float
    kd: float
    max_integral: float


@dataclass(frozen=True)
class PidConfig:
    attitude: PidGains
    position_kp: float
    position_kd: float
    max_tilt_command: float


@dataclass(frozen=True)
class SafetyConfig:
    landing_ignition_lockout: float
    max_landing_ignition_tilt: float
    max_landing_ignition_height: float
    max_gimbal_rate: float | None = None  # rad/s the commanded gimbal may change at; None means unlimited


@dataclass(frozen=True)
class ControllersConfig:
    """Who is in charge in each controlled phase: the PID or an outside policy, and who lights the landing motor."""

    boost: str = PID
    landing_burn: str = PID
    landing_ignition: str = TRIGGER

    def steering(self, phase_name: str) -> str:
        return {"BOOST": self.boost, "LANDING_BURN": self.landing_burn}.get(phase_name, PID)


@dataclass(frozen=True)
class FlightComputerConfig:
    sensors: SensorsConfig
    estimator: EstimatorConfig
    phases: PhaseConfig
    landing_trigger: LandingTriggerConfig
    pid: PidConfig
    safety: SafetyConfig
    brake: BrakeConfig | None = None
    controllers: ControllersConfig = ControllersConfig()


def load_flight_computer_config(root: Section, base_dir: Path) -> FlightComputerConfig:
    """Read the flight computer sections of a rocket YAML. The sensor file path is relative to it."""
    sensors = root.sub("sensors")
    sensors.only_keys("file")
    try:
        sensors_config = load_sensors_config(base_dir / sensors.string("file"))
    except ConfigError as error:
        raise ConfigError(f"{sensors.where('file')}: {error}") from error
    return FlightComputerConfig(
        sensors=sensors_config,
        estimator=_estimator(root.sub("estimator")),
        phases=_phases(root.sub("phases")),
        landing_trigger=_landing_trigger(root.sub("landing_trigger")),
        pid=_pid(root.sub("pid")),
        safety=_safety(root.sub("safety")),
        brake=_brake(root.sub("brake")) if root.has("brake") else None,
        controllers=load_controllers(root.sub("controllers")) if root.has("controllers") else ControllersConfig(),
    )


def load_controllers(section: Section) -> ControllersConfig:
    """Read a `controllers` mapping: BOOST and LANDING_BURN are pid or policy, landing_ignition trigger or policy."""
    section.only_keys("BOOST", "LANDING_BURN", "landing_ignition")
    boost = section.string("BOOST", default=PID)
    landing_burn = section.string("LANDING_BURN", default=PID)
    ignition = section.string("landing_ignition", default=TRIGGER)
    for key, value, choices in (("BOOST", boost, STEERING_CHOICES), ("LANDING_BURN", landing_burn, STEERING_CHOICES), ("landing_ignition", ignition, IGNITION_CHOICES)):
        if value not in choices:
            raise ConfigError(f"{section.where(key)} must be one of {choices}, got {value!r}")
    return ControllersConfig(boost, landing_burn, ignition)


def _estimator(section: Section) -> EstimatorConfig:
    section.only_keys(
        "pad_average_time_s", "baro_altitude_gain", "baro_velocity_gain_per_s", "gps_position_gain", "gps_velocity_gain"
    )
    return EstimatorConfig(
        pad_average_time=section.number("pad_average_time_s", above=0.0),
        baro_altitude_gain=section.number("baro_altitude_gain", minimum=0.0, maximum=MAX_GAIN_FRACTION),
        baro_velocity_gain=section.number("baro_velocity_gain_per_s", minimum=0.0),
        gps_position_gain=section.number("gps_position_gain", minimum=0.0, maximum=MAX_GAIN_FRACTION, default=0.0),
        gps_velocity_gain=section.number("gps_velocity_gain", minimum=0.0, maximum=MAX_GAIN_FRACTION, default=0.0),
    )


def _phases(section: Section) -> PhaseConfig:
    section.only_keys("liftoff_accel_mps2", "burnout_accel_mps2", "min_boost_time_s", "apogee_vz_mps", "abort_tilt_deg")
    config = PhaseConfig(
        liftoff_accel=section.number("liftoff_accel_mps2", above=0.0),
        burnout_accel=section.number("burnout_accel_mps2", minimum=0.0),
        min_boost_time=section.number("min_boost_time_s", minimum=0.0),
        apogee_vz=section.number("apogee_vz_mps", maximum=0.0),
        abort_tilt=section.number("abort_tilt_deg", above=0.0, maximum=MAX_TILT_DEG),
    )
    if config.burnout_accel >= config.liftoff_accel:
        raise ConfigError(f"{section.where('burnout_accel_mps2')} must be below liftoff_accel_mps2")
    return config


def _landing_trigger(section: Section) -> LandingTriggerConfig:
    section.only_keys("target_speed_mps", "target_height_m", "thrust_margin", "calibrate_drag_in_flight")
    return LandingTriggerConfig(
        target_speed=section.number("target_speed_mps", minimum=0.0),
        target_height=section.number("target_height_m", minimum=0.0),
        thrust_margin=section.number("thrust_margin", above=0.0, maximum=MAX_THRUST_MARGIN),
        calibrate_drag_in_flight=section.boolean("calibrate_drag_in_flight", default=False),
    )


def _brake(section: Section) -> BrakeConfig:
    section.only_keys("deploy_descent_speed_mps", "retract_descent_speed_mps", "burn_fraction")
    config = BrakeConfig(
        deploy_descent_speed=section.number("deploy_descent_speed_mps", minimum=0.0),
        retract_descent_speed=section.number("retract_descent_speed_mps", minimum=0.0),
        burn_fraction=section.number("burn_fraction", minimum=0.0, maximum=FULLY_OPEN),
    )
    if config.retract_descent_speed >= config.deploy_descent_speed:
        raise ConfigError(f"{section.where('retract_descent_speed_mps')} must be below deploy_descent_speed_mps")
    return config


def _pid(section: Section) -> PidConfig:
    section.only_keys("attitude", "position")
    attitude = section.sub("attitude")
    attitude.only_keys("kp", "ki_per_s", "kd_s", "max_integral_deg")
    position = section.sub("position")
    position.only_keys("kp_deg_per_m", "kd_deg_per_mps", "max_tilt_command_deg")
    return PidConfig(
        attitude=PidGains(
            kp=attitude.number("kp", minimum=0.0),
            ki=attitude.number("ki_per_s", minimum=0.0),
            kd=attitude.number("kd_s", minimum=0.0),
            max_integral=attitude.number("max_integral_deg", minimum=0.0),
        ),
        position_kp=position.number("kp_deg_per_m", minimum=0.0),
        position_kd=position.number("kd_deg_per_mps", minimum=0.0),
        max_tilt_command=position.number("max_tilt_command_deg", above=0.0, maximum=MAX_TILT_DEG),
    )


def _safety(section: Section) -> SafetyConfig:
    section.only_keys(
        "landing_ignition_lockout_s", "max_landing_ignition_tilt_deg", "max_landing_ignition_height_m",
        "max_gimbal_rate_deg_per_s",
    )
    return SafetyConfig(
        landing_ignition_lockout=section.number("landing_ignition_lockout_s", minimum=0.0),
        max_landing_ignition_tilt=section.number("max_landing_ignition_tilt_deg", above=0.0, maximum=MAX_TILT_DEG),
        max_landing_ignition_height=section.number("max_landing_ignition_height_m", above=0.0),
        max_gimbal_rate=section.number("max_gimbal_rate_deg_per_s", above=0.0) if section.has("max_gimbal_rate_deg_per_s") else None,
    )
