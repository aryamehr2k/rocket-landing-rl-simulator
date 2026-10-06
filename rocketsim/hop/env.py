"""One mission flight of the electric vehicle as a training episode, seen as two control planes.

Same interface as rocketsim.env.PlaneEpisode, so rocketsim.vecenv runs many of them at once and
one network learns from both planes. Every reset draws the hidden vehicle errors, the mission
variation and the wind of the flight.
"""

from dataclasses import replace
from typing import Any

import numpy as np

from rocketsim.curriculum import FULL_LEVEL, CurriculumLevel, level_wind, scale_range, scale_sensors
from rocketsim.guidance_config import POLICY
from rocketsim.hop.mission import FLYING_PHASES, MissionConfig, with_targets
from rocketsim.hop.policy import plane_action
from rocketsim.hop.rewards import HopRewardCalculator
from rocketsim.hop.simulation import HopErrors, HopSimulation
from rocketsim.hop.training import HopTrainingConfig
from rocketsim.hop.vehicle import HopVehicleConfig
from rocketsim.observation import PLANES, ObservationBuilder
from rocketsim.simconfig import SimConfig
from rocketsim.units import g_to_kg

SEED_LIMIT = 2 ** 31 - 1


def vehicle_for_training(vehicle: HopVehicleConfig, training: HopTrainingConfig) -> HopVehicleConfig:
    """The vehicle with the training file's controllers override applied, if it has one."""
    if training.controllers is None:
        return vehicle
    return replace(vehicle, controllers=training.controllers)


def flight_info(sim: HopSimulation) -> dict[str, Any]:
    """What happened in a finished flight, for training logs and evaluation."""
    result = sim.result
    return {
        "landed": result.landed, "success": result.success, "timeout": sim.flight.touchdown is None and not result.aborted,
        "aborted": result.aborted, "max_height": result.max_height, "hover_held": result.hover_held,
        "miss": result.miss_distance, "vertical_speed": result.touchdown_speed, "flight_time": sim.t,
        "policy_fallbacks": sim.computer.policy_fallbacks, "target_altitude": sim.mission.target_altitude,
    }


class HopEpisode:
    def __init__(
        self, vehicle: HopVehicleConfig, mission: MissionConfig, sim: SimConfig, training: HopTrainingConfig, seed: int
    ) -> None:
        self.base_vehicle = vehicle_for_training(vehicle, training)
        self.base_mission = mission
        self.base_sim = sim
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.rewards = HopRewardCalculator(training.rewards)
        self.observer = ObservationBuilder(training.observation, mission.planned_duration)
        self.level = FULL_LEVEL
        self.pending_level: CurriculumLevel | None = None
        self.last_gimbal = np.zeros(len(PLANES))
        self.last_throttle = 0.0
        self.sim = self._build(self.level)

    def _build(self, level: CurriculumLevel) -> HopSimulation:
        vehicle = replace(self.base_vehicle, sensors=scale_sensors(self.base_vehicle.sensors, level.sensor_noise))
        sim = replace(self.base_sim, wind=level_wind(self.base_sim.wind, level))
        return HopSimulation(vehicle, self.base_mission, sim, seed=self._next_seed())

    def _next_seed(self) -> int:
        return int(self.rng.integers(SEED_LIMIT))

    def set_level(self, level: CurriculumLevel) -> None:
        if level != self.level:
            self.pending_level = level

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.pending_level is not None:
            self.level, self.pending_level = self.pending_level, None
            self.sim = self._build(self.level)
        self.sim.reset(self._next_seed(), mission=self._draw_mission(), errors=self._draw_errors())
        self._draw_wind()
        self.last_gimbal[:] = 0.0
        self.last_throttle = 0.0
        return self.observe()

    def _draw_mission(self) -> MissionConfig:
        variation = self.training.mission_variation
        altitude = float(self.rng.uniform(*variation.target_altitude)) if variation.target_altitude else None
        hover = float(self.rng.uniform(*variation.hover_time)) if variation.hover_time else None
        return with_targets(self.base_mission, altitude, hover)

    def _draw_errors(self) -> HopErrors:
        ranges, factor = self.training.hidden_errors, self.level.hidden_errors
        return HopErrors(
            thrust_scale=float(self.rng.uniform(*scale_range(ranges.thrust_scale, factor))),
            dry_mass_offset=g_to_kg(float(self.rng.uniform(*scale_range(ranges.dry_mass_offset, factor)))),
        )

    def _draw_wind(self) -> None:
        episode = self.training.episode
        config = self.sim.wind.config
        speed = float(self.rng.uniform(0.0, config.steady)) if episode.random_wind_speed else config.steady
        direction = float(self.rng.uniform(0.0, 2.0 * np.pi)) if episode.random_wind_direction else config.steady_direction
        self.sim.wind.steady = speed * np.array([np.cos(direction), np.sin(direction)])

    def observe(self) -> np.ndarray:
        computer, t = self.sim.computer, self.sim.t
        return np.stack([self.observer.build(computer, plane, t, self.last_gimbal[plane]) for plane in PLANES])

    def step(self, actions: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        computer = self.sim.computer
        action = plane_action(list(actions), computer, self.training.action)
        self.sim.control_step(action)
        flying = computer.phase in FLYING_PHASES
        estimate = computer.estimate
        controllers = computer.controllers
        rewards = np.zeros(len(PLANES))
        throttle = computer.throttle
        for plane, gimbal in zip(PLANES, (action.delta_x, action.delta_y)):
            gimbal = self.last_gimbal[plane] if gimbal is None else gimbal
            if flying and POLICY in (controllers.steering, controllers.throttle):
                rewards[plane] = self.rewards.step(
                    computer.reference[0] - computer.height, computer.reference[1] - float(estimate.velocity[2]),
                    float(estimate.position[plane]), float(estimate.velocity[plane]), estimate.tilt[plane],
                    gimbal - self.last_gimbal[plane], throttle - self.last_throttle,
                )
            self.last_gimbal[plane] = gimbal
        self.last_throttle = throttle
        done = self.sim.done
        info: dict[str, Any] = {}
        if done:
            result = self.sim.result
            timed_out = self.sim.flight.touchdown is None and not result.aborted
            for plane in PLANES:
                rewards[plane] += self.rewards.terminal(result, timed_out, float(self.sim.flight.y[plane]))
            info = flight_info(self.sim)
        return self.observe(), rewards, done, info
