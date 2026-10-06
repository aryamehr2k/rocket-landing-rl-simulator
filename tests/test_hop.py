import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from rocketsim.actuators import RollActuator, Throttle
from rocketsim.hop.env import HopEpisode
from rocketsim.hop.mission import Guidance, HopPhase, MissionScore, load_mission_config, with_targets
from rocketsim.hop.policy import plane_action
from rocketsim.hop.simulation import HopSimulation
from rocketsim.hop.training import load_hop_training_config
from rocketsim.hop.vehicle import is_vehicle_file, load_vehicle_config
from rocketsim.motors import load_motor
from rocketsim.physics import IBURNED, IW, RocketDynamics
from rocketsim.simconfig import load_sim_config
from rocketsim.vecenv import PlanePairVecEnv
from rocketsim.yaml_section import ConfigError
from tests.conftest import REPO_ROOT

VEHICLE = REPO_ROOT / "configs" / "vehicles" / "electric_hopper.yaml"
MISSION = REPO_ROOT / "configs" / "missions" / "hop_50m.yaml"
TRAINING = REPO_ROOT / "configs" / "training" / "hop.yaml"
CALM = REPO_ROOT / "configs" / "training" / "calm_exact.yaml"
DT = 0.02


def pid_vehicle():
    vehicle = load_vehicle_config(VEHICLE)
    return replace(vehicle, controllers=replace(vehicle.controllers, steering="pid", throttle="pid"))


def test_electric_motor_file() -> None:
    motor = load_motor(REPO_ROOT / "configs" / "motors" / "example_edf_90mm.yaml")
    assert motor.electric and motor.throttleable
    assert motor.max_thrust == pytest.approx(38.0)
    assert motor.total_mass == pytest.approx(0.38) and motor.propellant_mass == 0.0
    assert motor.throttle_lag == pytest.approx(0.10)
    assert math.isinf(motor.exhaust_velocity)


def test_electric_thrust_follows_throttle_and_burns_nothing() -> None:
    vehicle = load_vehicle_config(VEHICLE)
    sim = load_sim_config(CALM)
    dynamics = RocketDynamics(vehicle.body, sim.environment)
    y = dynamics.initial_state()
    inputs = dynamics.new_inputs()
    assert dynamics.motor_thrust(0, 1.0, y, inputs) == 0.0  # not started
    inputs.ignite(0, 0.0)
    inputs.motors[0].throttle = 0.5
    inputs.motors[0].thrust_scale = 1.1
    assert dynamics.motor_thrust(0, 1.0, y, inputs) == pytest.approx(38.0 * 0.5 * 1.1)
    assert dynamics.derivatives(1.0, y, inputs)[IBURNED] == 0.0
    inputs.motors[0].throttle = 2.0  # clipped to full throttle
    assert dynamics.motor_thrust(0, 1.0, y, inputs) == pytest.approx(38.0 * 1.1)


def test_roll_torque_and_fan_reaction() -> None:
    vehicle = load_vehicle_config(VEHICLE)
    dynamics = RocketDynamics(vehicle.body, load_sim_config(CALM).environment)
    y = dynamics.initial_state()
    inputs = dynamics.new_inputs()
    inputs.roll_torque = 0.1
    roll_inertia = dynamics.mass_properties(y).roll_inertia
    assert dynamics.derivatives(0.0, y, inputs)[IW][2] == pytest.approx(0.1 / roll_inertia)
    inputs.roll_torque = 0.0
    inputs.ignite(0, 0.0)
    inputs.motors[0].throttle = 1.0
    reaction = vehicle.motor.spec.reaction_torque_per_n * 38.0
    assert dynamics.derivatives(0.5, y, inputs)[IW][2] == pytest.approx(reaction / roll_inertia)


def test_throttle_and_roll_actuators() -> None:
    throttle = Throttle(time_constant=0.1, dt=0.005)
    throttle.command(1.0)
    for _ in range(20):  # one time constant
        level = throttle.step()
    assert level == pytest.approx(1.0 - math.exp(-1.0), abs=0.02)
    assert throttle.command(1.5) == 1.0 and throttle.command(-1.0) == 0.0
    roll = RollActuator(max_torque=0.15, time_constant=0.0, dt=0.005)
    roll.command(1.0)
    assert roll.step() == pytest.approx(0.15)


def test_vehicle_file_validation(tmp_path: Path) -> None:
    assert is_vehicle_file(VEHICLE) and not is_vehicle_file(REPO_ROOT / "configs" / "rockets" / "example_tvc.yaml")
    data = yaml.safe_load(VEHICLE.read_text())
    data["motor"]["file"] = str(VEHICLE.parent / data["motor"]["file"])
    data["sensors"]["file"] = str(VEHICLE.parent / data["sensors"]["file"])
    data["airframe"]["dry_mass_g"] = 5000.0
    heavy = tmp_path / "heavy.yaml"
    heavy.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="cannot lift"):
        load_vehicle_config(heavy)
    data["airframe"]["dry_mass_g"] = 1800.0
    data["controllers"]["mode"] = "sideways"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="mode must be one of"):
        load_vehicle_config(bad)


def test_guidance_climbs_hovers_descends_and_lands() -> None:
    mission = load_mission_config(MISSION)
    guidance = Guidance(mission)
    guidance.start(0.0)
    t, phases, max_speed = 0.0, {}, 0.0
    while guidance.phase != HopPhase.LANDING or guidance.height > 0.0:
        t += DT
        phase, height, speed = guidance.update(t, DT)
        phases.setdefault(phase, t)
        max_speed = max(max_speed, speed)
        assert height <= mission.target_altitude + 1e-9
        if t > 200.0:
            raise AssertionError("the reference never reached the ground")
    hover = phases[HopPhase.DESCENT] - phases[HopPhase.HOVER]
    assert hover == pytest.approx(mission.hover_time + mission.hover_margin, abs=2 * DT)
    assert max_speed == pytest.approx(mission.climb_speed)
    assert guidance.speed == pytest.approx(-mission.final_speed)


def test_mission_score_criteria() -> None:
    mission = with_targets(load_mission_config(MISSION), altitude=20.0, hover_time=5.0)
    score = MissionScore(mission)
    for step in range(600):
        score.update(step * 0.01, 20.5)  # 6 s inside the 2 m band
    result = score.result(landed=True, miss_distance=3.0, touchdown_speed=0.8, aborted=False)
    assert result.reached_altitude and result.hover_ok and result.success
    assert not score.result(landed=True, miss_distance=12.0, touchdown_speed=0.8, aborted=False).success
    assert not score.result(landed=False, miss_distance=1.0, touchdown_speed=3.0, aborted=False).success


def test_pid_flies_the_mission_in_calm_air() -> None:
    sim = replace(load_sim_config(CALM), simulation=replace(load_sim_config(CALM).simulation, max_flight_time=120.0))
    simulation = HopSimulation(pid_vehicle(), load_mission_config(MISSION), sim, seed=0)
    result = simulation.run()
    assert result.success, result.summary()
    assert result.miss_distance < 5.0


def test_gps_keeps_the_position_estimate_honest() -> None:
    sim = replace(load_sim_config(CALM), simulation=replace(load_sim_config(CALM).simulation, max_flight_time=120.0))
    mission = with_targets(load_mission_config(MISSION), altitude=20.0, hover_time=5.0)
    simulation = HopSimulation(pid_vehicle(), mission, sim, seed=2)
    while not simulation.done:
        simulation.control_step()
    estimate = simulation.computer.estimate
    error = np.hypot(*(estimate.position[:2] - simulation.flight.y[:2]))
    assert error < 3.0


def test_plane_action_mapping() -> None:
    vehicle = pid_vehicle()
    simulation = HopSimulation(replace(vehicle, controllers=replace(vehicle.controllers, steering="policy", throttle="policy")),
                               load_mission_config(MISSION), load_sim_config(CALM), seed=1)
    training = load_hop_training_config(TRAINING)
    computer = simulation.computer
    action = plane_action([np.array([0.5, 0.0]), np.array([-1.0, 0.0])], computer, training.action)
    assert action.delta_x == pytest.approx(0.5 * vehicle.body.gimbal.max_angle)
    assert action.delta_y == pytest.approx(-vehicle.body.gimbal.max_angle)
    assert action.throttle == pytest.approx(computer.hover_throttle)  # upright, zero vote: hover throttle
    missing = plane_action([np.array([np.nan, 1.0]), None], computer, training.action)
    assert missing.delta_x is None and missing.delta_y is None and missing.throttle is None
    residual = replace(computer.controllers, mode="residual")
    computer.controllers = residual
    action = plane_action([np.array([1.0, 1.0]), np.array([1.0, 1.0])], computer, training.action)
    assert action.delta_x == pytest.approx(training.action.residual_gimbal)
    assert action.throttle == pytest.approx(training.action.throttle_range)


def test_hop_episode_and_vectorised_training_env() -> None:
    training = load_hop_training_config(TRAINING)
    vehicle, mission = load_vehicle_config(training.vehicle_path), load_mission_config(training.mission_path)
    sim = load_sim_config(TRAINING)
    venv = PlanePairVecEnv(lambda i: HopEpisode(vehicle, mission, sim, training, seed=20 + i), n_sims=2, workers=1)
    obs = venv.reset()
    assert obs.shape == (4, len(training.observation))
    for _ in range(200):
        obs, rewards, dones, infos = venv.step(np.zeros((4, 2), dtype=np.float32))
        assert np.all(np.isfinite(obs)) and np.all(np.isfinite(rewards))
    venv.close()
