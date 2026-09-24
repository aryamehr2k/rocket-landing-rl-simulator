"""Phases, PID signs, landing trigger table and safety refusals."""

import math

import numpy as np
import pytest

from tests.conftest import constant_thrust_motor, make_environment, make_rocket
from rocketsim.commands import ControlCommand
from rocketsim.config import MotorConfig, RocketConfig
from rocketsim.estimator import Estimate
from rocketsim.landing_trigger import LandingTrigger
from rocketsim.phases import Phase, PhaseMachine
from rocketsim.pid import PlanePid, world_to_servo
from rocketsim.quaternion import IDENTITY, from_axis_angle
from rocketsim.safety import Safety

G = 10.0
DT = 0.02


def estimate(velocity: tuple[float, float, float] = (0.0, 0.0, 0.0), tilt_x: float = 0.0) -> Estimate:
    q = from_axis_angle(np.array([0.0, 1.0, 0.0]), tilt_x) if tilt_x else IDENTITY
    return Estimate(np.zeros(3), np.array(velocity), q, np.zeros(3))


def test_phase_sequence(rocket: RocketConfig) -> None:
    machine = PhaseMachine(rocket.computer.phases)
    assert not machine.imu_update(0.5, 9.8) and machine.phase == Phase.PAD
    assert machine.imu_update(1.0, 15.0) and machine.phase == Phase.BOOST
    assert machine.update(1.2, 1.0, estimate()) == Phase.BOOST  # too early for burnout
    assert machine.update(1.6, 1.0, estimate()) == Phase.COAST
    assert machine.update(3.0, 1.0, estimate(velocity=(0.0, 0.0, 2.0))) == Phase.COAST
    assert machine.update(4.0, 1.0, estimate(velocity=(0.0, 0.0, -2.0))) == Phase.DESCENT
    machine.landing_commanded(6.0)
    assert machine.phase == Phase.LANDING_BURN
    machine.landed(8.0)
    assert [phase for _, phase in machine.history] == [
        Phase.PAD, Phase.BOOST, Phase.COAST, Phase.DESCENT, Phase.LANDING_BURN, Phase.LANDED,
    ]


def test_large_tilt_during_boost_aborts(rocket: RocketConfig) -> None:
    machine = PhaseMachine(rocket.computer.phases)
    machine.imu_update(1.0, 15.0)
    assert machine.update(1.1, 15.0, estimate(tilt_x=math.radians(45.0))) == Phase.ABORT
    machine.landed(5.0)
    assert machine.phase == Phase.ABORT


def test_pid_steers_the_nose_back_upright_and_toward_the_pad(rocket: RocketConfig) -> None:
    pid = PlanePid(rocket.computer.pid)
    leaning = pid.update(0.0, 0.0, math.radians(5.0), 0.0, DT, hold_position=False)
    assert leaning < 0.0  # nose toward +x, gimbal must lean it back toward -x
    rotating = pid.update(0.0, 0.0, 0.0, 0.5, DT, hold_position=False)
    assert rotating < 0.0  # falling toward +x, damp it
    off_pad = pid.update(3.0, 0.0, 0.0, 0.0, DT, hold_position=True)
    assert pid.tilt_command < 0.0 and off_pad < 0.0  # lean toward -x to come back
    far_off = pid.update(100.0, 0.0, 0.0, 0.0, DT, hold_position=True)
    assert math.isclose(pid.tilt_command, -rocket.computer.pid.max_tilt_command)
    assert far_off < off_pad


def test_world_commands_rotate_by_roll() -> None:
    assert world_to_servo(1.0, 2.0, 0.0) == (1.0, 2.0)
    pitch, yaw = world_to_servo(1.0, 0.0, math.radians(90.0))
    assert math.isclose(pitch, 0.0, abs_tol=1e-12) and math.isclose(yaw, -1.0)


def trigger_rocket() -> RocketConfig:
    rocket = make_rocket()
    landing = MotorConfig(
        role="landing", spec=constant_thrust_motor("L", 30.0, 2.0, 0.1, 0.2), position=0.8, gimbaled=True,
        ignition_delay_mean=0.0, ignition_delay_spread=0.0,
    )
    return type(rocket)(**{**rocket.__dict__, "motors": (rocket.motors[0], landing)})


def test_stopping_distance_matches_constant_deceleration() -> None:
    rocket = trigger_rocket()
    trigger = LandingTrigger(rocket, rocket.computer.landing_trigger, make_environment(gravity=G, density=0.0))
    mass = 1.0 + 0.1 + 0.2  # dry, empty ascent case, full landing motor
    decel = 30.0 * 0.95 / mass - G
    expected = (10.0 ** 2 - 1.0 ** 2) / (2.0 * decel)
    assert math.isclose(trigger.stopping_distance(10.0), expected, rel_tol=0.05)
    assert trigger.stopping_distance(20.0) > trigger.stopping_distance(10.0)
    assert trigger.can_reach_target_from > 20.0


def test_trigger_fires_at_the_required_height_after_the_igniter_delay() -> None:
    rocket = trigger_rocket()
    trigger = LandingTrigger(rocket, rocket.computer.landing_trigger, make_environment(gravity=G, density=0.0))
    required = trigger.required_height(10.0)
    assert math.isclose(required, trigger.stopping_distance(10.0) + 0.5)
    assert not trigger.should_ignite(required + 1.0, 10.0)
    assert trigger.should_ignite(required - 0.1, 10.0)
    trigger.ignition_delay = 0.1
    later_speed = 10.0 + G * 0.1
    fallen = 10.0 * 0.1 + 0.5 * G * 0.1 ** 2
    assert trigger.should_ignite(trigger.required_height(later_speed) + fallen - 0.01, 10.0)
    assert not trigger.should_ignite(trigger.required_height(later_speed) + fallen + 0.01, 10.0)


@pytest.fixture
def safety(rocket: RocketConfig) -> Safety:
    return Safety(rocket.computer.safety, rocket.gimbal)


def test_safety_clips_gimbal_and_refuses_out_of_phase_ignitions(safety: Safety, rocket: RocketConfig) -> None:
    wanted = ControlCommand(gimbal_pitch=1.0, gimbal_yaw=-1.0, ignite_ascent=True, ignite_landing=True)
    command, refused = safety.filter(wanted, Phase.BOOST, 1.0, 0.5, estimate(), 10.0)
    limit = rocket.gimbal.max_angle
    assert (command.gimbal_pitch, command.gimbal_yaw) == (limit, -limit)
    assert not command.ignite_ascent and not command.ignite_landing
    assert len(refused) == 2


def test_safety_landing_ignition_rules(safety: Safety) -> None:
    wanted = ControlCommand(ignite_landing=True)
    ok, refused = safety.filter(wanted, Phase.DESCENT, 10.0, 1.0, estimate(), 30.0)
    assert ok.ignite_landing and refused == ()
    for phase, t, tilt, height in (
        (Phase.DESCENT, 2.0, 0.0, 30.0),  # within the lockout
        (Phase.DESCENT, 10.0, math.radians(30.0), 30.0),  # too tilted
        (Phase.DESCENT, 10.0, 0.0, 500.0),  # too high
        (Phase.BOOST, 10.0, 0.0, 30.0),  # wrong phase
    ):
        command, refused = safety.filter(wanted, phase, t, 1.0, estimate(tilt_x=tilt), height)
        assert not command.ignite_landing and len(refused) == 1


def test_abort_zeroes_everything(safety: Safety) -> None:
    wanted = ControlCommand(gimbal_pitch=0.1, gimbal_yaw=0.1, ignite_landing=True)
    command, _ = safety.filter(wanted, Phase.ABORT, 10.0, 1.0, estimate(), 30.0)
    assert command == ControlCommand()
