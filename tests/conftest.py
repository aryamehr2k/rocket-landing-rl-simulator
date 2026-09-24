"""Shared fixtures: a rocket with round numbers so expected values can be worked out by hand."""

import math
from pathlib import Path

import numpy as np
import pytest

from rocketsim.config import (
    AeroConfig,
    AirframeConfig,
    ControlConfig,
    GimbalConfig,
    LegsConfig,
    MotorConfig,
    RocketConfig,
    ServoCalibration,
)
from rocketsim.guidance_config import (
    EstimatorConfig,
    FlightComputerConfig,
    LandingTriggerConfig,
    PhaseConfig,
    PidConfig,
    PidGains,
    SafetyConfig,
)
from rocketsim.motors import MotorSpec
from rocketsim.sensors import BarometerConfig, ImuChannelConfig, ImuConfig, SensorsConfig
from rocketsim.simconfig import EnvironmentConfig, WindConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_ROCKET = REPO_ROOT / "configs" / "rockets" / "example_tvc.yaml"
EXAMPLE_SIM = REPO_ROOT / "configs" / "training" / "default.yaml"
ROTATE_Y = np.array([0.0, 1.0, 0.0])
ROTATE_X = np.array([1.0, 0.0, 0.0])


def constant_thrust_motor(name: str, thrust: float, burn_time: float, propellant: float, total: float) -> MotorSpec:
    return MotorSpec(
        name=name,
        propellant_mass=propellant,
        total_mass=total,
        times=np.array([0.0, burn_time]),
        thrusts=np.array([thrust, thrust]),
    )


def make_servo() -> ServoCalibration:
    return ServoCalibration(
        center_us=1500.0,
        pulse_us_per_rad=math.degrees(10.0),
        sign=1,
        linkage_ratio=0.5,
        pulse_resolution_us=1.0,
        min_us=1000.0,
        max_us=2000.0,
    )


def make_sensors(noise: bool = False) -> SensorsConfig:
    """200 Hz IMU and 50 Hz barometer, perfect unless noise is asked for."""
    scale = 1.0 if noise else 0.0
    return SensorsConfig(
        name="test",
        imu=ImuConfig(
            rate_hz=200.0, lag=0.0,
            accelerometer=ImuChannelConfig(noise_std=0.05 * scale, bias_std=0.02 * scale, range=156.0),
            gyroscope=ImuChannelConfig(noise_std=math.radians(0.1) * scale, bias_std=math.radians(0.5) * scale, range=math.radians(2000.0)),
        ),
        barometer=BarometerConfig(rate_hz=50.0, lag=0.0, noise_std=0.3 * scale, bias_std=1.0 * scale),
        source="<test>",
    )


def make_computer(noise: bool = False) -> FlightComputerConfig:
    return FlightComputerConfig(
        sensors=make_sensors(noise),
        estimator=EstimatorConfig(pad_average_time=0.5, baro_altitude_gain=0.1, baro_velocity_gain=0.5),
        phases=PhaseConfig(liftoff_accel=12.0, burnout_accel=3.0, min_boost_time=0.5, apogee_vz=-1.0, abort_tilt=math.radians(40.0)),
        landing_trigger=LandingTriggerConfig(target_speed=1.0, target_height=0.5, thrust_margin=0.95),
        pid=PidConfig(
            attitude=PidGains(kp=0.6, ki=0.0, kd=0.12, max_integral=math.radians(3.0)),
            position_kp=math.radians(1.5), position_kd=math.radians(5.0), max_tilt_command=math.radians(8.0),
        ),
        safety=SafetyConfig(landing_ignition_lockout=2.5, max_landing_ignition_tilt=math.radians(25.0), max_landing_ignition_height=120.0),
    )


def make_rocket(
    drag_coefficient: float = 0.5,
    normal_force_slope: float = 2.0,
    cp: float = 0.7,
    gimbaled: bool = True,
    dry_mass: float = 1.0,
) -> RocketConfig:
    """1 kg dry body at 0.5 m, 20 N ascent motor at 0.9 m, 10 N landing motor at 0.8 m, 4 legs."""
    return RocketConfig(
        name="test",
        airframe=AirframeConfig(
            dry_mass=dry_mass, dry_cg=0.5, dry_pitch_inertia=0.1, dry_roll_inertia=0.002, length=1.0,
            reference_diameter=0.1,
        ),
        aero=AeroConfig(drag_coefficient=drag_coefficient, normal_force_slope=normal_force_slope, cp=cp),
        motors=(
            MotorConfig(
                role="ascent", spec=constant_thrust_motor("A", 20.0, 2.0, 0.1, 0.2), position=0.9,
                gimbaled=gimbaled, ignition_delay_mean=0.0, ignition_delay_spread=0.0,
            ),
            MotorConfig(
                role="landing", spec=constant_thrust_motor("L", 10.0, 1.0, 0.05, 0.1), position=0.8,
                gimbaled=gimbaled, ignition_delay_mean=0.0, ignition_delay_spread=0.0,
            ),
        ),
        gimbal=GimbalConfig(
            pivot=0.95, max_angle=math.radians(10.0), servo_rate_limit=math.radians(300.0), servo_delay=0.02,
            servo_deadband=math.radians(0.2), pitch_servo=make_servo(), yaw_servo=make_servo(),
        ),
        legs=LegsConfig(
            span=0.3, height=0.1, count=4, max_vertical_speed=2.0, max_lateral_speed=1.0,
            max_tilt=math.radians(10.0),
        ),
        control=ControlConfig(control_rate_hz=50.0, action_delay_steps=1),
        computer=make_computer(),
        source="<test>",
    )


def make_environment(gravity: float = 10.0, density: float = 1.0) -> EnvironmentConfig:
    """Uniform atmosphere: a huge scale height makes density constant at every altitude."""
    return EnvironmentConfig(site_elevation=0.0, gravity=gravity, sea_level_density=density, scale_height=1e12)


@pytest.fixture
def rocket() -> RocketConfig:
    return make_rocket()


@pytest.fixture
def environment() -> EnvironmentConfig:
    return make_environment()


@pytest.fixture
def no_wind() -> WindConfig:
    return WindConfig(steady=0.0, steady_direction=0.0, gust_std=0.0, gust_time_constant=1.0)
