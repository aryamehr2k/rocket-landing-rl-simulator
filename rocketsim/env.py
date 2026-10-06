"""Gymnasium environment around the closed loop simulation.

One flight is one episode. The policy sees the flight computer's estimate through the fields
named in the training YAML and commands the gimbal of one plane plus the landing igniter. The
same policy runs once for the pitch plane (x) and once for the yaw plane (y), as it will on the
flight computer, so `PlaneEpisode` exposes one flight as two plane views and the training
vectorised environment (rocketsim.vecenv) stacks them. `RocketLandingEnv` is the plain
Gymnasium view of one plane with the PID flying the other, for tests and experiments.
"""

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np

from rocketsim.commands import PlaneAction
from rocketsim.config import RocketConfig
from rocketsim.curriculum import FULL_LEVEL, CurriculumLevel, apply_level
from rocketsim.guidance_config import POLICY
from rocketsim.observation import PITCH_PLANE, PLANES, ObservationBuilder
from rocketsim.physics import IBURNED, IPOS, IVEL
from rocketsim.rewards import RewardCalculator
from rocketsim.simconfig import SimConfig
from rocketsim.simulation import Simulation
from rocketsim.training_config import TrainingConfig
from rocketsim.units import rad_to_deg

ACTION_SIZE = 2  # gimbal for the plane, ignite output
GIMBAL, IGNITE = 0, 1
ACTION_LOW, ACTION_HIGH = -1.0, 1.0
SEED_LIMIT = 2 ** 31 - 1


def rocket_for_training(rocket: RocketConfig, training: TrainingConfig) -> RocketConfig:
    """The rocket with the training YAML's controllers override applied, if it has one."""
    if training.controllers is None:
        return rocket
    return replace(rocket, computer=replace(rocket.computer, controllers=training.controllers))


class PlaneEpisode:
    """One simulated flight seen as two plane views, with rewards and the curriculum applied."""

    def __init__(self, rocket: RocketConfig, sim: SimConfig, training: TrainingConfig, seed: int) -> None:
        rocket = rocket_for_training(rocket, training)
        self.base_rocket = rocket
        self.base_sim = sim
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.rewards = RewardCalculator(training.rewards)
        self.observer = ObservationBuilder(training.observation, rocket.motor("landing").spec.burn_time)
        self.max_angle = rocket.gimbal.max_angle
        self.last_command = np.zeros(len(PLANES))
        self.propellant_burned = 0.0
        self.level = FULL_LEVEL
        self.pending_level: CurriculumLevel | None = None
        self.sim = self._build(self.level)

    def _build(self, level: CurriculumLevel) -> Simulation:
        rocket, sim = apply_level(self.base_rocket, self.base_sim, level)
        return Simulation(rocket, sim, seed=self._next_seed())

    def _next_seed(self) -> int:
        return int(self.rng.integers(SEED_LIMIT))

    def set_level(self, level: CurriculumLevel) -> None:
        """Change the curriculum level; the flight in progress finishes first, the next reset applies it."""
        if level != self.level:
            self.pending_level = level

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Start a new flight and return the two plane observations, shape (2, fields)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.pending_level is not None:
            self.level, self.pending_level = self.pending_level, None
            self.sim = self._build(self.level)
        self.sim.reset(self._next_seed())
        self._draw_wind()
        self.last_command[:] = 0.0
        self.propellant_burned = self._burned()
        return self.observe()

    def _draw_wind(self) -> None:
        # The flight's steady wind: the level's maximum, or anything up to it, in any direction.
        episode = self.training.episode
        speed = float(np.linalg.norm(self.sim.wind.config.steady * np.array([1.0, 0.0])))
        direction = float(self.sim.wind.config.steady_direction)
        if episode.random_wind_speed:
            speed = float(self.rng.uniform(0.0, speed))
        if episode.random_wind_direction:
            direction = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.sim.wind.steady = speed * np.array([np.cos(direction), np.sin(direction)])

    def observe(self) -> np.ndarray:
        computer, t = self.sim.flight_computer, self.sim.t
        return np.stack([self.observer.build(computer, plane, t, self.last_command[plane]) for plane in PLANES])

    def step(self, actions: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        """Apply one action per plane (None hands that plane to the PID) and run one control step."""
        gimbal: list[float | None] = [None, None]
        ignite_votes = []
        for plane, action in zip(PLANES, actions):
            # A missing or non-finite action hands the plane to the PID for this step.
            if action is None or not np.all(np.isfinite(action)):
                continue
            gimbal[plane] = float(np.clip(action[GIMBAL], ACTION_LOW, ACTION_HIGH)) * self.max_angle
            ignite_votes.append(float(np.clip(action[IGNITE], ACTION_LOW, ACTION_HIGH)))
        ignite = bool(np.mean(ignite_votes) > self.training.action.ignite_threshold) if ignite_votes else None
        self.sim.control_step(PlaneAction(gimbal[0], gimbal[1], ignite))
        computer = self.sim.flight_computer
        controlled = computer.controllers.steering(computer.phase.value) == POLICY
        estimate = computer.estimate
        burned = self._burned()
        propellant = burned - self.propellant_burned
        self.propellant_burned = burned
        rewards = np.zeros(len(PLANES))
        for plane in PLANES:
            command = gimbal[plane] if gimbal[plane] is not None else self.last_command[plane]
            rewards[plane] = self.rewards.step(
                controlled, estimate.tilt[plane], float(estimate.velocity[plane]), float(estimate.position[plane]),
                command - self.last_command[plane], propellant, gimbal=command,
            )
            self.last_command[plane] = command
        done = self.sim.done
        info: dict[str, Any] = {}
        if done:
            y = self.sim.flight.y
            for plane in PLANES:
                rewards[plane] += self.rewards.terminal(self.sim.flight.touchdown, float(y[IVEL][plane]), float(y[IPOS][plane]))
            info = episode_info(self.sim)
        return self.observe(), rewards, done, info

    def _burned(self) -> float:
        return float(np.sum(self.sim.flight.y[IBURNED:]))


def episode_info(sim: Simulation) -> dict[str, Any]:
    """What happened in a finished flight, for logs and evaluation."""
    touchdown = sim.flight.touchdown
    info: dict[str, Any] = {
        "landed": bool(touchdown is not None and touchdown.success),
        "timeout": touchdown is None,
        "apogee": float(sim.flight.apogee),
        "policy_fallbacks": sim.flight_computer.policy_fallbacks,
        "flight_time": float(sim.t),
    }
    if touchdown is not None:
        info.update(
            {
                "vertical_speed": touchdown.vertical_speed,
                "lateral_speed": touchdown.lateral_speed,
                "tilt_deg": rad_to_deg(touchdown.tilt),
                "miss": touchdown.miss_distance,
                "failures": touchdown.failures,
            }
        )
    return info


class RocketLandingEnv(gym.Env):
    """One plane of the flight as a Gymnasium environment; the PID flies the other plane."""

    metadata = {"render_modes": []}

    def __init__(
        self, rocket: RocketConfig, sim: SimConfig, training: TrainingConfig, plane: int = PITCH_PLANE, seed: int = 0
    ) -> None:
        super().__init__()
        self.episode = PlaneEpisode(rocket, sim, training, seed)
        self.plane = plane
        size = self.episode.observer.size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(ACTION_LOW, ACTION_HIGH, shape=(ACTION_SIZE,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observations = self.episode.reset(seed)
        return observations[self.plane].copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        actions: list[np.ndarray | None] = [None, None]
        actions[self.plane] = np.asarray(action, dtype=np.float32)
        observations, rewards, done, info = self.episode.step(actions)
        terminated = done and not info.get("timeout", False)
        truncated = done and info.get("timeout", False)
        return observations[self.plane].copy(), float(rewards[self.plane]), terminated, truncated, info
