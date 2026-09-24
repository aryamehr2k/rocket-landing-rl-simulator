"""Flight computer settings from the rocket YAML: sensors, estimator, phases, trigger, PID, safety.

These sections describe what runs on the board, so one rocket file holds everything the
firmware needs. rocketsim.config calls load_flight_computer_config while loading a rocket.
"""

from dataclasses import dataclass
from pathlib import Path

from rocketsim.sensors import SensorsConfig, load_sensors_config
from rocketsim.yaml_section import ConfigError, Section

SECTION_KEYS = ("sensors", "estimator", "phases", "landing_trigger", "pid", "safety")
MAX_TILT_DEG = 90.0
MAX_GAIN_FRACTION = 1.0


@dataclass(frozen=True)
class EstimatorConfig:
    pad_average_time: float
    baro_altitude_gain: float
    baro_velocity_gain: float


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


@dataclass(frozen=True)
class FlightComputerConfig:
    sensors: SensorsConfig
    estimator: EstimatorConfig
    phases: PhaseConfig
    landing_trigger: LandingTriggerConfig
    pid: PidConfig
    safety: SafetyConfig


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
    )


def _estimator(section: Section) -> EstimatorConfig:
    section.only_keys("pad_average_time_s", "baro_altitude_gain", "baro_velocity_gain_per_s")
    return EstimatorConfig(
        pad_average_time=section.number("pad_average_time_s", above=0.0),
        baro_altitude_gain=section.number("baro_altitude_gain", minimum=0.0, maximum=MAX_GAIN_FRACTION),
        baro_velocity_gain=section.number("baro_velocity_gain_per_s", minimum=0.0),
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
    section.only_keys("target_speed_mps", "target_height_m", "thrust_margin")
    return LandingTriggerConfig(
        target_speed=section.number("target_speed_mps", minimum=0.0),
        target_height=section.number("target_height_m", minimum=0.0),
        thrust_margin=section.number("thrust_margin", above=0.0, maximum=MAX_GAIN_FRACTION),
    )


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
    section.only_keys("landing_ignition_lockout_s", "max_landing_ignition_tilt_deg", "max_landing_ignition_height_m")
    return SafetyConfig(
        landing_ignition_lockout=section.number("landing_ignition_lockout_s", minimum=0.0),
        max_landing_ignition_tilt=section.number("max_landing_ignition_tilt_deg", above=0.0, maximum=MAX_TILT_DEG),
        max_landing_ignition_height=section.number("max_landing_ignition_height_m", above=0.0),
    )
