"""Servo model order of operations and the igniter delay."""

import math

import numpy as np

from rocketsim.actuators import Igniter, Servo
from rocketsim.config import RocketConfig
from rocketsim.physics import Inputs, MotorInputs

DT = 0.005
CENTER_US = 1500.0
US_PER_SERVO_DEG = 10.0
LINKAGE = 0.5


def make_servo(rocket: RocketConfig) -> Servo:
    return Servo(rocket.gimbal.pitch_servo, rocket.gimbal, DT)


def test_command_is_clipped_to_the_gimbal_limit(rocket: RocketConfig) -> None:
    servo = make_servo(rocket)
    pulse = servo.command(math.radians(20.0))
    servo_deg = 10.0 / LINKAGE  # 10 deg gimbal limit through a 0.5 linkage
    assert pulse == CENTER_US + servo_deg * US_PER_SERVO_DEG


def test_small_change_inside_the_deadband_is_ignored(rocket: RocketConfig) -> None:
    servo = make_servo(rocket)
    assert servo.command(math.radians(0.1)) == CENTER_US
    servo.command(math.radians(1.0))
    assert servo.command(math.radians(1.1)) == servo.command(math.radians(1.0))


def test_pulse_is_quantised_half_up(rocket: RocketConfig) -> None:
    servo = make_servo(rocket)
    pulse = servo.command(math.radians(0.33))  # 0.66 deg servo travel, 6.6 us, rounds to 7
    assert pulse == CENTER_US + 7.0
    servo.command(math.radians(0.325))  # 6.5 us rounds up
    assert servo.pulse == CENTER_US + 7.0


def test_delay_then_rate_limit(rocket: RocketConfig) -> None:
    servo = make_servo(rocket)
    servo.command(math.radians(5.0))
    delay_steps = round(rocket.gimbal.servo_delay / DT)
    for _ in range(delay_steps):
        assert servo.step() == 0.0
    per_step = rocket.gimbal.servo_rate_limit * DT
    angles = [servo.step() for _ in range(4)]
    assert np.allclose(angles[:3], [per_step, 2 * per_step, 3 * per_step])
    assert math.isclose(angles[3], math.radians(5.0), rel_tol=1e-9)
    assert servo.step() == angles[3]


def test_igniter_delay_and_single_use(rocket: RocketConfig) -> None:
    motor = rocket.motors[1]
    motor = type(motor)(**{**motor.__dict__, "ignition_delay_mean": 0.2, "ignition_delay_spread": 0.05})
    igniter = Igniter(1, motor, np.random.default_rng(3))
    assert 0.15 <= igniter.delay <= 0.25
    inputs = Inputs(motors=[MotorInputs(), MotorInputs()])
    assert igniter.command(4.0, inputs)
    assert inputs.motors[1].ignition_time == 4.0 + igniter.delay
    assert not igniter.command(4.5, inputs)
    assert inputs.motors[1].ignition_time == 4.0 + igniter.delay
