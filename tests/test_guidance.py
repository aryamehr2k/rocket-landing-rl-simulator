"""Phases, PID signs, landing trigger table and safety refusals."""

import math
from dataclasses import replace

import numpy as np
import pytest

from tests.conftest import EXAMPLE_ROCKET, EXAMPLE_SIM, constant_thrust_motor, make_environment, make_rocket
from rocketsim.commands import ControlCommand
from rocketsim.config import MotorConfig, RocketConfig, load_rocket_config
from rocketsim.dragdevice import DragDeviceConfig
from rocketsim.estimator import Estimate
from rocketsim.flightcomputer import FlightComputer
from rocketsim.landing_trigger import LandingTrigger
from rocketsim.phases import Phase, PhaseMachine
from rocketsim.pid import PlanePid, world_to_servo
from rocketsim.quaternion import IDENTITY, from_axis_angle
from rocketsim.safety import Safety
from rocketsim.simconfig import load_sim_config

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


def example_computer() -> FlightComputer:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    return FlightComputer(rocket, load_sim_config(EXAMPLE_SIM).environment, DT, 2.0)


def test_brake_rules_open_hold_and_shut_in_order() -> None:
    computer = example_computer()
    assert computer.brake_rule(Phase.COAST, 12.0) == 0.0
    assert computer.brake_rule(Phase.DESCENT, 5.0) == 0.0  # not yet falling fast enough
    assert computer.brake_rule(Phase.DESCENT, 9.0) == 1.0
    assert computer.brake_rule(Phase.DESCENT, 7.0) == 1.0  # stays open once opened
    assert computer.brake_rule(Phase.LANDING_BURN, 15.0) == 1.0  # burn_fraction of the example
    assert computer.brake_rule(Phase.LANDING_BURN, 2.0) == 0.0  # hand-over: shut
    assert computer.brake_rule(Phase.LANDING_BURN, 6.0) == 0.0  # and stays shut
    fresh = example_computer()
    assert fresh.brake_rule(Phase.LANDING_BURN, 10.0) == 0.0  # never deployed, never opened in the burn


def test_safety_brake_refusals() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    nose = Safety(rocket.computer.safety, rocket.gimbal, rocket.drag_device, rocket.descent_cg)
    wanted = ControlCommand(brake_fraction=1.0)
    for phase, allowed in ((Phase.COAST, False), (Phase.DESCENT, True), (Phase.LANDING_BURN, True), (Phase.BOOST, False)):
        command, refused = nose.filter(wanted, phase, 10.0, 1.0, estimate(), 30.0)
        assert (command.brake_fraction == 1.0) is allowed, phase
        assert (refused == ()) is allowed
    tail_device = DragDeviceConfig(0.05, 0.96, 0.5, 0.3)
    tail = Safety(rocket.computer.safety, rocket.gimbal, tail_device, rocket.descent_cg)
    command, refused = tail.filter(wanted, Phase.DESCENT, 10.0, 1.0, estimate(), 30.0)
    assert command.brake_fraction == 0.0 and "behind the centre of gravity" in refused[0]
    command, _ = tail.filter(wanted, Phase.LANDING_BURN, 10.0, 1.0, estimate(), 30.0)
    assert command.brake_fraction == 1.0
    command, _ = nose.filter(ControlCommand(brake_fraction=3.0), Phase.DESCENT, 10.0, 1.0, estimate(), 30.0)
    assert command.brake_fraction == 1.0  # clipped


def test_example_trigger_table_with_the_brake() -> None:
    trigger = example_computer().trigger
    assert trigger.can_reach_target_from == pytest.approx(21.5)
    assert trigger.expected_arrival_speed == pytest.approx(20.0, abs=0.1)
    assert 12.5 < trigger.stopping_distance(20.0) < 13.5
    assert trigger.drag_factor_at(1.0) > 20.0 * trigger.drag_factor_at(0.0)
    assert trigger.drag_factor == trigger.drag_factor_at(1.0)  # burn_fraction 1 in the example


def test_delay_prediction_counts_the_drag() -> None:
    trigger = example_computer().trigger
    trigger.ignition_delay = 0.18
    speed, height = 20.0, 30.0
    accel = trigger.gravity - trigger.drag_factor_at(1.0) * speed ** 2 / trigger.mass_at_ignition
    assert abs(accel) < 0.2  # at terminal speed the descent barely accelerates
    later_speed = speed + accel * 0.18
    fallen = 0.5 * (speed + later_speed) * 0.18
    required = float(trigger.required_height(later_speed))
    assert trigger.should_ignite(required + fallen - 0.01, speed, brake_fraction=1.0)
    assert not trigger.should_ignite(required + fallen + 0.01, speed, brake_fraction=1.0)
    assert trigger.should_ignite(required + fallen + 0.01, speed, brake_fraction=0.0)  # shut: gravity's full 1.8 m/s
    decisions = trigger.should_ignite(np.array([height, 5.0]), np.array([speed, speed]), 1.0)
    assert list(decisions) == [False, True]


def test_drag_fit_from_the_accelerometer_rebuilds_the_table() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    config = replace(rocket.computer.landing_trigger, calibrate_drag_in_flight=True)
    trigger = LandingTrigger(rocket, config, load_sim_config(EXAMPLE_SIM).environment)
    before, reachable_before = trigger.heights.copy(), trigger.reachable.copy()
    true_total = 1.1 * trigger.drag_factor_at(1.0)  # the real brake has 10 % more drag than the file says
    t, speed = 0.0, 14.0
    while not trigger.observe_descent(t, speed, true_total * speed ** 2 / trigger.mass_at_ignition, 0.0):
        t += DT
        speed += 0.05  # still speeding up: the fit does not need terminal speed
        assert t < 2.0
    assert trigger.calibrated
    assert trigger.drag_factor_device == pytest.approx(true_total - trigger.drag_factor_body, rel=1e-6)
    near = (np.abs(trigger.speeds - trigger.expected_arrival_speed) <= 4.9) & reachable_before
    far = (np.abs(trigger.speeds - trigger.expected_arrival_speed) >= 5.1) & reachable_before
    assert near.sum() > 10 and np.all(trigger.heights[near] < before[near])  # more drag: shorter stopping distances
    assert np.array_equal(trigger.heights[far], before[far])
    assert not trigger.observe_descent(t + DT, speed, 9.0, 0.0)  # done once
    trigger.reset()
    assert not trigger.calibrated and np.array_equal(trigger.heights, before)


def test_required_height_stays_at_the_edge_beyond_the_table_reach() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    trigger = LandingTrigger(rocket, rocket.computer.landing_trigger, load_sim_config(EXAMPLE_SIM).environment)
    edge = trigger.can_reach_target_from
    assert 21.0 <= edge <= 22.0
    at_edge = float(trigger.required_height(edge))
    assert float(trigger.required_height(edge + 0.5)) == at_edge
    assert float(trigger.required_height(40.0)) == at_edge
    assert float(trigger.required_height(edge - 0.5)) < at_edge
    whole_burn = float(trigger.burn(edge + 0.5).distance) + trigger.config.target_height
    assert whole_burn > at_edge + 10.0  # what an uncapped table would ask for: tens of metres more


def test_drag_fit_skips_slow_and_tilted_samples_and_corrects_for_tilt() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    config = replace(rocket.computer.landing_trigger, calibrate_drag_in_flight=True)
    env = load_sim_config(EXAMPLE_SIM).environment
    trigger = LandingTrigger(rocket, config, env)
    total = trigger.drag_factor_at(1.0)
    assert not trigger.observe_descent(0.0, 5.0, 1.0, 0.0) and trigger._fit_start is None  # too slow
    assert not trigger.observe_descent(0.1, 20.0, 9.0, math.radians(30.0)) and trigger._fit_start is None
    tilt, t = math.radians(15.0), 0.2
    while not trigger.observe_descent(t, 20.0, total * 400.0 * math.cos(tilt) / trigger.mass_at_ignition, tilt):
        t += DT
    assert trigger.drag_factor_device == pytest.approx(trigger._nominal[0], rel=1e-6)
    plain = LandingTrigger(rocket, rocket.computer.landing_trigger, env)
    assert not plain.observe_descent(0.0, 21.0, 9.8, 0.0) and not plain.observe_descent(1.0, 21.0, 9.8, 0.0)


def test_warning_when_the_terminal_speed_is_near_the_table_reach() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    small = replace(rocket, drag_device=replace(rocket.drag_device, drag_area=0.048))  # terminal speed about 21.6 m/s
    with pytest.warns(UserWarning, match="within"):
        LandingTrigger(small, small.computer.landing_trigger, load_sim_config(EXAMPLE_SIM).environment)


def test_safety_rate_limits_the_gimbal_command() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    config = replace(rocket.computer.safety, max_gimbal_rate=math.radians(100.0))
    safety = Safety(config, rocket.gimbal, control_dt=0.02)
    big = ControlCommand(gimbal_pitch=math.radians(7.0), gimbal_yaw=-math.radians(7.0))
    first, _ = safety.filter(big, Phase.LANDING_BURN, 5.0, 0.0, estimate(), 10.0)
    assert first.gimbal_pitch == pytest.approx(math.radians(2.0))
    assert first.gimbal_yaw == pytest.approx(-math.radians(2.0))
    for _ in range(3):
        last, _ = safety.filter(big, Phase.LANDING_BURN, 5.0, 0.0, estimate(), 10.0)
    assert last.gimbal_pitch == pytest.approx(math.radians(7.0))
    safety.reset()
    again, _ = safety.filter(big, Phase.LANDING_BURN, 5.0, 0.0, estimate(), 10.0)
    assert again.gimbal_pitch == pytest.approx(math.radians(2.0))
    unlimited = Safety(replace(config, max_gimbal_rate=None), rocket.gimbal, control_dt=0.02)
    free, _ = unlimited.filter(big, Phase.LANDING_BURN, 5.0, 0.0, estimate(), 10.0)
    assert free.gimbal_pitch == pytest.approx(math.radians(7.0))
