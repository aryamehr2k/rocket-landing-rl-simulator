"""Rewards of the electric vehicle task, per control plane, weighted from the training YAML (penalties negative)."""

import math

from rocketsim.hop.mission import HopPhase, MissionResult
from rocketsim.hop.training import HopRewardConfig

MAX_TRACKING_ERROR = 10.0  # m or m/s; larger errors are charged as this, so one bad moment cannot dominate
PLANNED_TOUCHDOWN_PHASES = (HopPhase.DESCENT, HopPhase.LANDING)


class HopRewardCalculator:
    def __init__(self, config: HopRewardConfig) -> None:
        self.config = config

    def step(
        self, height_error: float, speed_error: float, lateral_position: float, lateral_speed: float, tilt: float,
        gimbal_change: float, throttle_change: float,
    ) -> float:
        c = self.config
        reward = c.height_error_step * min(abs(height_error), MAX_TRACKING_ERROR)
        reward += c.speed_error_step * min(abs(speed_error), MAX_TRACKING_ERROR)
        reward += c.lateral_position_step * min(abs(lateral_position), MAX_TRACKING_ERROR)
        reward += c.lateral_speed_step * min(abs(lateral_speed), MAX_TRACKING_ERROR)
        reward += c.tilt_step * abs(tilt)
        reward += c.gimbal_change_step * gimbal_change ** 2
        reward += c.throttle_change_step * throttle_change ** 2
        return reward

    def terminal(
        self, result: MissionResult, timed_out: bool, plane_miss: float, touchdown_phase: HopPhase = HopPhase.LANDING,
    ) -> float:
        """Paid once at the end of the flight. `plane_miss` is the distance from the pad in this plane.

        A touchdown before the plan's descent counts as a crash however gentle it was, so that dropping
        back onto the pad never pays better than flying the mission.
        """
        c = self.config
        if timed_out:
            return c.timeout
        if result.aborted or not result.landed or touchdown_phase not in PLANNED_TOUCHDOWN_PHASES:
            reward = c.crash
        else:
            reward = c.landed
            if result.success:
                reward += c.mission_success
        if math.isfinite(result.touchdown_speed):
            reward += c.touchdown_vertical_speed * result.touchdown_speed
        reward += c.miss_distance * min(abs(plane_miss), MAX_TRACKING_ERROR * 2)
        if not result.inside_radius:
            reward += c.outside_radius
        return reward
