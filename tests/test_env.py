import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocketsim.config import RocketConfig, load_rocket_config
from rocketsim.curriculum import CurriculumLevel, CurriculumStage, apply_level, level_for
from rocketsim.commands import PlaneAction
from rocketsim.env import PlaneEpisode, RocketLandingEnv, rocket_for_training
from rocketsim.guidance_config import POLICY, ControllersConfig
from rocketsim.observation import PITCH_PLANE, YAW_PLANE, ObservationBuilder, ObservationField
from rocketsim.policy import MlpPolicy, PolicyController, load_policy
from rocketsim.rewards import RewardCalculator, RewardConfig
from rocketsim.simconfig import SimConfig, load_sim_config
from rocketsim.simulation import Simulation
from rocketsim.touchdown import TouchdownResult
from rocketsim.training_config import TrainingConfig, load_training_config
from rocketsim.vecenv import PlanePairVecEnv
from rocketsim.yaml_section import ConfigError
from tests.conftest import EXAMPLE_ROCKET, EXAMPLE_SIM

SHORT_FLIGHT_STEPS = 40  # control steps; enough to check the worker plumbing without a whole flight


@pytest.fixture(scope="module")
def setup() -> tuple[RocketConfig, SimConfig, TrainingConfig]:
    return load_rocket_config(EXAMPLE_ROCKET), load_sim_config(EXAMPLE_SIM), load_training_config(EXAMPLE_SIM)


def test_training_config_loads(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    _, _, training = setup
    assert training.observation[0].name == "height" and training.observation[0].scale == 100.0
    assert training.ppo.hidden_layers == (32, 32)
    assert training.curriculum[0].level.wind_max == 0.0
    assert training.curriculum[-1].level.wind_max == 5.0
    assert training.rewards.crash < 0.0 < training.rewards.landed


def test_training_config_rejects_unknown_field(tmp_path: Path) -> None:
    text = EXAMPLE_SIM.read_text().replace("{name: height, scale: 100.0}", "{name: altitude, scale: 100.0}")
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="not a known observation field"):
        load_training_config(path)


def test_gym_interface_and_one_episode(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, sim, training = setup
    env = RocketLandingEnv(rocket, sim, training, seed=1)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (len(training.observation),) and obs.dtype == np.float32
    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert math.isfinite(reward)
        done = terminated or truncated
        steps += 1
    assert steps > 100 and "landed" in info and not info["timeout"]


def test_observation_fields_read_the_estimate(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, sim, _ = setup
    fields = (ObservationField("height", 10.0), ObservationField("lateral_position", 1.0), ObservationField("phase_boost", 1.0))
    observer = ObservationBuilder(fields, burn_time=8.0)
    simulation = Simulation(rocket, sim, seed=0)
    for _ in range(150):
        simulation.control_step()
    computer = simulation.flight_computer
    obs = observer.build(computer, PITCH_PLANE, simulation.t, 0.0)
    assert obs[0] == pytest.approx(computer.height / 10.0, rel=1e-5)
    assert obs[1] == pytest.approx(float(computer.estimate.position[0]), rel=1e-5)
    assert obs[2] == (1.0 if computer.phase.value == "BOOST" else 0.0)
    obs_y = observer.build(computer, YAW_PLANE, simulation.t, 0.0)
    assert obs_y[1] == pytest.approx(float(computer.estimate.position[1]), rel=1e-5)
    with pytest.raises(ValueError, match="unknown observation fields"):
        ObservationBuilder((ObservationField("nope", 1.0),), burn_time=8.0)


def test_reward_terms() -> None:
    config = RewardConfig(
        touchdown_vertical_speed=-1.0, touchdown_lateral_speed=-2.0, touchdown_tilt_deg=-0.5, miss_distance=-0.1,
        crash=-10.0, landed=10.0, timeout=-7.0, tilt_step=-1.0, gimbal_change_step=-1.0, gimbal_step=-2.0,
    )
    rewards = RewardCalculator(config)
    assert rewards.step(True, 0.1, 0.0, 0.0, 0.2, 0.0, gimbal=0.05) == pytest.approx(-0.1 - 0.04 - 0.1)
    assert rewards.step(False, 0.1, 0.0, 0.0, 0.2, 0.0, gimbal=0.05) == 0.0
    # Touchdown totals (lateral 0.9 m/s, miss 5 m) must not be used: the plane's own 0.5 m/s and 2 m are.
    good = TouchdownResult(1.0, 5.0, 1.0, 0.9, math.radians(2.0), 0.3, 0.17, ())
    assert rewards.terminal(good, 0.5, 2.0) == pytest.approx(-1.0 - 1.0 - 1.0 - 0.2 + 10.0)
    bad = replace(good, failures=("tilt",))
    assert rewards.terminal(bad, 0.5, 2.0) == pytest.approx(-1.0 - 1.0 - 1.0 - 0.2 - 10.0)
    assert rewards.terminal(None, 0.0, 0.0) == -7.0


def test_curriculum_levels_and_scaling(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, sim, _ = setup
    stages = (CurriculumStage(0.3, CurriculumLevel(0.0, 0.0, 0.5, 0.0)), CurriculumStage(1.0, CurriculumLevel()))
    assert level_for(stages, 0.1).wind_max == 0.0
    assert level_for(stages, 0.5) == CurriculumLevel()
    assert level_for((), 0.5) == CurriculumLevel()
    windy = replace(sim, wind=replace(sim.wind, steady=4.0, gust_std=1.5))
    scaled_rocket, scaled_sim = apply_level(rocket, windy, CurriculumLevel(2.0, 0.0, 0.5, 0.0))
    assert scaled_sim.wind.steady == 2.0 and scaled_sim.wind.gust_std == 0.0
    assert apply_level(rocket, windy, CurriculumLevel())[1].wind.steady == 4.0
    assert scaled_rocket.computer.sensors.barometer.noise_std == pytest.approx(rocket.computer.sensors.barometer.noise_std * 0.5)
    assert scaled_sim.randomize is not None
    assert scaled_sim.randomize.landing_thrust_scale == pytest.approx((1.0, 1.0))
    assert scaled_sim.randomize.dry_mass_offset == pytest.approx((0.0, 0.0))


def test_vecenv_steps_both_planes_and_resets(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, sim, training = setup
    venv = PlanePairVecEnv(lambda i: PlaneEpisode(rocket, sim, training, seed=10 + i), n_sims=2, workers=1)
    obs = venv.reset()
    assert obs.shape == (4, len(training.observation))
    # Sub-environments 2k and 2k+1 are the two planes of flight k: same height, own lateral field.
    height, lateral = [f.name for f in training.observation].index("height"), [f.name for f in training.observation].index("lateral_position")
    assert obs[0, height] == obs[1, height] and obs[2, height] == obs[3, height]
    finished = 0
    for _ in range(1200):
        obs, rewards, dones, infos = venv.step(np.zeros((4, 2), dtype=np.float32))
        assert rewards.shape == (4,) and dones.shape == (4,)
        for done, info in zip(dones, infos):
            if done:
                finished += 1
                assert "landed" in info
        if finished >= 4:
            break
    assert finished >= 4
    venv.set_level(CurriculumLevel(0.0, 0.0, 0.0, 0.0))
    assert venv.reset().shape == (4, len(training.observation))
    venv.close()


class ShortFlight(PlaneEpisode):
    """A flight cut off after SHORT_FLIGHT_STEPS that reports its curriculum level in every info."""

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.steps = 0
        return super().reset(seed)

    def step(self, actions: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        observations, rewards, done, info = super().step(actions)
        self.steps += 1
        return observations, rewards, done or self.steps >= SHORT_FLIGHT_STEPS, {**info, "level": self.level}


def test_vecenv_workers_match_a_single_process(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, sim, training = setup
    calm = CurriculumLevel(0.0, 0.0, 0.0, 0.0)
    single, pooled = (
        PlanePairVecEnv(lambda i: ShortFlight(rocket, sim, training, seed=30 + i), n_sims=2, workers=workers)
        for workers in (1, 2)
    )
    actions = np.zeros((single.num_envs, single.action_size), dtype=np.float32)
    try:
        assert np.array_equal(single.reset(), pooled.reset())
        single.set_level(calm)
        pooled.set_level(calm)
        restarted = np.zeros(pooled.num_envs, dtype=bool)
        for _ in range(2 * SHORT_FLIGHT_STEPS):
            expected, result = single.step(actions), pooled.step(actions)
            for a, b in zip(expected[:3], result[:3]):
                assert np.array_equal(a, b)
            # The level set mid-flight applies from the next flight on.
            assert all(info["level"] == calm for info, again in zip(result[3], restarted) if again)
            if restarted.all():
                break
            restarted |= result[2]
        assert restarted.all()
    finally:
        single.close()
        pooled.close()


def random_policy(size: int, names: tuple[str, ...]) -> MlpPolicy:
    rng = np.random.default_rng(0)
    return MlpPolicy(
        weights=(rng.normal(size=(8, size)).astype(np.float32), rng.normal(size=(2, 8)).astype(np.float32)),
        biases=(np.zeros(8, dtype=np.float32), np.zeros(2, dtype=np.float32)),
        activation="tanh", obs_mean=np.zeros(size, dtype=np.float32), obs_std=np.ones(size, dtype=np.float32),
        obs_clip=10.0, field_names=names,
    )


def test_policy_controller_runs_and_saves(setup: tuple[RocketConfig, SimConfig, TrainingConfig], tmp_path: Path) -> None:
    rocket, sim, training = setup
    observer = ObservationBuilder(training.observation, rocket.motor("landing").spec.burn_time)
    policy = random_policy(observer.size, observer.names)
    policy.save(tmp_path / "policy.npz")
    loaded = load_policy(tmp_path / "policy.npz")
    obs = np.linspace(-1.0, 1.0, observer.size).astype(np.float32)
    assert loaded.forward(obs) == pytest.approx(policy.forward(obs))
    assert np.all(np.abs(loaded.forward(obs)) <= 1.0)
    controller = PolicyController(loaded, observer, rocket.gimbal.max_angle, 0.0)
    simulation = Simulation(rocket, sim, seed=4)
    while not simulation.done:
        action = controller.action(simulation.flight_computer, simulation.t)
        assert action.delta_x is not None and abs(action.delta_x) <= rocket.gimbal.max_angle
        simulation.control_step(action)
    assert simulation.flight.touchdown is not None
    with pytest.raises(ValueError, match="trained on fields"):
        PolicyController(loaded, ObservationBuilder(training.observation[:3], 8.0), 0.1, 0.0)


def test_invalid_policy_output_falls_back_to_the_pid(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, sim, _ = setup
    owners = ControllersConfig(boost=POLICY, landing_burn=POLICY, landing_ignition=POLICY)
    rocket = replace(rocket, computer=replace(rocket.computer, controllers=owners))
    simulation = Simulation(rocket, sim, seed=5)
    while simulation.flight_computer.phase.value != "BOOST":
        simulation.control_step(PlaneAction(float("nan"), float("nan"), False))
    for _ in range(20):
        simulation.control_step(PlaneAction(float("nan"), float("nan"), False))
    assert simulation.flight_computer.policy_fallbacks > 0
    assert math.isfinite(simulation.servos[0].angle)


def test_pid_owns_phases_the_config_gives_it(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, sim, _ = setup
    simulation = Simulation(rocket, sim, seed=6)  # example file: BOOST pid, LANDING_BURN policy
    angle = math.radians(5.0)
    while simulation.flight_computer.phase.value != "BOOST" or simulation.t < sim.simulation.pad_hold_time + 0.6:
        simulation.control_step(PlaneAction(angle, angle, None))
    assert abs(simulation.servos[0].angle) < math.radians(1.0)  # the PID kept the boost vertical


def test_training_controllers_override_is_shared(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, _, training = setup
    owners = ControllersConfig(boost=POLICY, landing_burn=POLICY, landing_ignition=POLICY)
    trained = replace(training, controllers=owners)
    assert rocket_for_training(rocket, trained).computer.controllers == owners
    assert rocket_for_training(rocket, training).computer.controllers == rocket.computer.controllers


def test_nan_actions_hand_the_plane_to_the_pid(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, sim, training = setup
    episode = PlaneEpisode(rocket, sim, training, seed=2)
    episode.reset()
    nan_action = np.array([float("nan"), float("nan")], dtype=np.float32)
    for _ in range(300):
        obs, rewards, done, _ = episode.step([nan_action, nan_action])
        assert np.all(np.isfinite(obs)) and np.all(np.isfinite(rewards))
        if done:
            break


def test_training_wind_is_drawn_per_flight(setup: tuple[RocketConfig, SimConfig, TrainingConfig]) -> None:
    rocket, sim, training = setup
    episode = PlaneEpisode(rocket, sim, training, seed=9)
    episode.set_level(CurriculumLevel(wind_max=5.0, gust_std=0.0))
    speeds = []
    for _ in range(6):
        episode.reset()
        speeds.append(float(np.linalg.norm(episode.sim.wind.steady)))
    assert max(speeds) <= 5.0 and max(speeds) - min(speeds) > 0.5
