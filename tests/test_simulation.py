"""The closed loop end to end: a PID landing, repeatability, the open loop action path and the log."""

import math
from pathlib import Path

import numpy as np
import pytest

from rocketsim.commands import PlaneAction
from rocketsim.config import RocketConfig, load_rocket_config
from rocketsim.flightlog import ESTIMATE_COLUMNS, SENSOR_COLUMNS, read_flight_log
from rocketsim.phases import Phase
from rocketsim.simconfig import SimConfig, load_sim_config
from rocketsim.simulation import Simulation, with_wind
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
    assert not sim.igniters[1].commanded  # the trigger is off when a policy acts
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
