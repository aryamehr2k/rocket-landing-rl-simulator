"""The closed loop end to end: a PID landing, repeatability, the open loop action path and the log."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rocketsim.guidance_config import POLICY, ControllersConfig
from rocketsim.commands import PlaneAction
from rocketsim.config import RocketConfig, load_rocket_config
from rocketsim.flightlog import ESTIMATE_COLUMNS, SENSOR_COLUMNS, read_flight_log
from rocketsim.phases import Phase
from rocketsim.simconfig import SimConfig, load_sim_config
from rocketsim.aero import air_density
from rocketsim.dragdevice import drag_factor
from rocketsim.simulation import Simulation, with_fixed_errors, with_wind
from tests.conftest import EXAMPLE_ROCKET, EXAMPLE_SIM

LANDING_SEED = 1
EXPECTED_PHASES = [Phase.PAD, Phase.BOOST, Phase.COAST, Phase.DESCENT, Phase.LANDING_BURN, Phase.LANDED]


@pytest.fixture(scope="module")
def example() -> tuple[RocketConfig, SimConfig]:
    return load_rocket_config(EXAMPLE_ROCKET), load_sim_config(EXAMPLE_SIM)


def test_pid_lands_the_example_rocket_in_calm_air(example: tuple[RocketConfig, SimConfig]) -> None:
    rocket, sim_config = example
    sim = Simulation(rocket, sim_config, seed=LANDING_SEED)
    sim.run()
    touchdown = sim.flight.touchdown
    assert touchdown is not None and touchdown.success, touchdown
    assert touchdown.vertical_speed < rocket.legs.max_vertical_speed
    assert [phase for _, phase in sim.flight_computer.phases.history] == EXPECTED_PHASES
    assert sim.flight_computer.refusals == ()
    assert sim.flight.apogee > 50.0


def test_same_seed_gives_the_same_flight(example: tuple[RocketConfig, SimConfig]) -> None:
    rocket, sim_config = example
    sim = Simulation(rocket, sim_config, seed=7)
    sim.run()
    first = (sim.t, sim.flight.y.copy(), sim.igniters[1].delay)
    sim.reset(seed=7)
    sim.run()
    assert sim.t == first[0] and sim.igniters[1].delay == first[2]
    assert np.array_equal(sim.flight.y, first[1])


def test_open_loop_action_path_and_safety(example: tuple[RocketConfig, SimConfig]) -> None:
    rocket, sim_config = example
    # Let a policy steer the boost and decide the landing ignition, as the controllers section allows.
    owners = ControllersConfig(boost=POLICY, landing_burn=POLICY, landing_ignition=POLICY)
    rocket = replace(rocket, computer=replace(rocket.computer, controllers=owners))
    sim = Simulation(rocket, sim_config, seed=3)
    angle = math.radians(2.0)
    while sim.flight_computer.phase != Phase.BOOST or sim.t < sim_config.simulation.pad_hold_time + 0.6:
        sim.control_step(PlaneAction(angle, -angle, ignite_landing=True))
        assert not sim.igniters[1].commanded  # landing ignition during PAD or BOOST is refused
    assert sim.flight_computer.refusals
    assert math.isclose(sim.servos[0].angle, angle, abs_tol=math.radians(0.3))
    assert math.isclose(sim.servos[1].angle, -angle, abs_tol=math.radians(0.3))
    while not sim.done:
        sim.control_step(PlaneAction(0.0, 0.0, ignite_landing=False))
    assert not sim.igniters[1].commanded  # the policy owns the ignition and never asked for it
    assert sim.flight.touchdown is not None and not sim.flight.touchdown.success


def test_log_has_sensors_estimate_and_phases(example: tuple[RocketConfig, SimConfig], tmp_path: Path) -> None:
    rocket, sim_config = example
    sim = Simulation(rocket, sim_config, seed=LANDING_SEED, log_path=tmp_path / "flight.csv")
    sim.run()
    sim.close()
    log = read_flight_log(tmp_path / "flight.csv")
    assert set(np.unique(log["phase"])) == {phase.value for phase in EXPECTED_PHASES}
    for column in SENSOR_COLUMNS + ESTIMATE_COLUMNS:
        assert np.isfinite(log[column][-1]), column
    in_flight = log["phase"] == Phase.DESCENT.value
    assert np.max(np.abs(log["est_z_m"][in_flight] - log["true_z_m"][in_flight])) < 2.0
    assert np.all(log["servo_pitch_us"][in_flight] == rocket.gimbal.pitch_servo.center_us)  # gimbal centred when falling
    assert np.nanmax(log["landing_thrust_n"]) > 0.0 and np.nanmax(log["ascent_thrust_n"]) > 0.0


def test_wind_override_changes_the_flight(example: tuple[RocketConfig, SimConfig]) -> None:
    rocket, sim_config = example
    windy = with_wind(sim_config, steady=4.0, direction=0.0, gust_std=None)
    assert windy.steady == 4.0 and windy.gust_std == sim_config.wind.gust_std
    sim = Simulation(rocket, sim_config, seed=LANDING_SEED, wind=windy)
    sim.run()
    assert sim.flight.touchdown is not None
    assert abs(sim.flight.y[0]) > 1.0  # blown downrange


def test_hidden_errors_are_drawn_per_seed_and_hidden_from_the_flight_computer(example: tuple[RocketConfig, SimConfig]) -> None:
    rocket, sim_config = example
    sim = Simulation(rocket, sim_config, seed=5)
    first = sim.draw
    ranges = sim_config.randomize
    assert ranges is not None
    assert ranges.landing_thrust_scale[0] <= first.landing_thrust_scale <= ranges.landing_thrust_scale[1]
    assert ranges.dry_mass_offset[0] <= first.dry_mass_offset <= ranges.dry_mass_offset[1]
    assert sim.flight.inputs.motors[1].thrust_scale == first.landing_thrust_scale
    assert sim.dynamics.rocket.airframe.dry_mass == pytest.approx(rocket.airframe.dry_mass + first.dry_mass_offset)
    assert sim.dynamics.rocket.drag_device.drag_area == pytest.approx(rocket.drag_device.drag_area * first.device_drag_area_scale)
    sim.reset(seed=5)
    assert sim.draw == first
    sim.reset(seed=6)
    assert sim.draw != first
    # The flight computer keeps the nominal rocket: same mass and same table as a trigger built from the file.
    trigger = sim.flight_computer.trigger
    assert trigger.mass_at_ignition == pytest.approx(rocket.descent_mass)
    from rocketsim.landing_trigger import LandingTrigger
    nominal = LandingTrigger(rocket, rocket.computer.landing_trigger, sim_config.environment, trigger.ignition_delay - rocket.motor("landing").ignition_delay_mean)
    assert np.array_equal(nominal.heights, trigger.heights)


def test_brake_opens_in_the_descent_and_shuts_for_the_tail(example: tuple[RocketConfig, SimConfig], tmp_path: Path) -> None:
    rocket, sim_config = example
    sim = Simulation(rocket, sim_config, seed=LANDING_SEED, log_path=tmp_path / "brake.csv")
    sim.run()
    sim.close()
    log = read_flight_log(tmp_path / "brake.csv")
    descent, burn = log["phase"] == Phase.DESCENT.value, log["phase"] == Phase.LANDING_BURN.value
    assert np.max(log["brake_fraction"][descent]) == pytest.approx(1.0)
    assert np.all(log["brake_fraction"][log["phase"] == Phase.BOOST.value] == 0.0)
    assert np.max(log["device_drag_n"][descent]) > 12.0
    assert log["brake_fraction"][-1] == 0.0 and np.min(log["brake_fraction"][burn]) == 0.0
    speed_at_ignition = -log["true_vz_mps"][burn][0]
    assert 19.0 < speed_at_ignition < 21.0  # terminal speed, not a free fall
    summary = sim.burn_summary
    assert summary.stop_height is not None and 0.0 < summary.stop_height < 3.0
    assert summary.thrust_start_speed is not None and 19.0 < summary.thrust_start_speed < 21.5
    assert summary.burn_time_left is not None and summary.burn_time_left > 2.0
    assert summary.height_estimate_error is not None and abs(summary.height_estimate_error) < 1.0
    assert "stopped at" in summary.describe()


def test_policy_brake_action_overrides_the_rules_and_is_refused_during_boost(example: tuple[RocketConfig, SimConfig]) -> None:
    rocket, sim_config = example
    sim = Simulation(rocket, sim_config, seed=3)
    while sim.flight_computer.phase != Phase.BOOST:
        sim.control_step(PlaneAction(0.0, 0.0, False, brake=1.0))
    sim.control_step(PlaneAction(0.0, 0.0, False, brake=1.0))
    assert any("brake refused" in reason for reason in sim.flight_computer.refusals)
    while sim.flight_computer.phase != Phase.DESCENT:
        sim.control_step(PlaneAction(0.0, 0.0, False))
    sim.control_step(PlaneAction(0.0, 0.0, False, brake=0.4))
    sim.control_step(PlaneAction(0.0, 0.0, False, brake=0.4))
    assert sim.applied.brake_fraction == 0.4 and sim.flight_computer.refusals == ()
    assert sim.brake_servo is not None and sim.brake_servo.target == 0.4


def test_fixed_errors_pin_one_hidden_error(example: tuple[RocketConfig, SimConfig]) -> None:
    from rocketsim.simulation import with_fixed_errors
    rocket, sim_config = example
    pinned = with_fixed_errors(sim_config, landing_thrust_scale=1.05, dry_mass_offset=None)
    sim = Simulation(rocket, pinned, seed=2)
    assert sim.draw.landing_thrust_scale == 1.05
    assert sim.draw.dry_mass_offset != 0.0  # the YAML's range still applies to the others


def test_in_flight_drag_fit_runs_before_the_burn_and_lands_near_the_truth(
    example: tuple[RocketConfig, SimConfig]
) -> None:
    rocket, sim_config = example
    trigger_config = replace(rocket.computer.landing_trigger, calibrate_drag_in_flight=True)
    calibrating = replace(rocket, computer=replace(rocket.computer, landing_trigger=trigger_config))
    sim = Simulation(calibrating, with_fixed_errors(sim_config, 1.0, 0.0, 1.05), seed=LANDING_SEED)
    calibrated_in_phase = None
    while not sim.done:
        sim.control_step()
        if calibrated_in_phase is None and sim.flight_computer.trigger.calibrated:
            calibrated_in_phase = sim.flight_computer.phase
    assert calibrated_in_phase == Phase.DESCENT
    truth = drag_factor(sim.dynamics.rocket.drag_device, 1.0, air_density(0.0, sim_config.environment))
    assert sim.flight_computer.trigger.drag_factor_device == pytest.approx(truth, rel=0.03)
    assert sim.flight.touchdown is not None and sim.flight.touchdown.success
    sim.reset(seed=LANDING_SEED + 1)
    assert not sim.flight_computer.trigger.calibrated  # a new flight starts from the file again
