"""Rewards of the electric vehicle task, per control plane. Weights come from the training YAML.

Step terms charge for being off the mission's reference and for moving the actuators; the end of
the flight pays for landing, for meeting every mission criterion, and charges for a crash, an
abort or running out of time. Penalties are negative weights.
"""

import math

from rocketsim.hop.mission import MissionResult
from rocketsim.hop.training import HopRewardConfig

MAX_TRACKING_ERROR = 10.0  # m or m/s; larger errors are charged as this, so one bad moment cannot dominate


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

    def terminal(self, result: MissionResult, timed_out: bool, plane_miss: float) -> float:
        """Paid once at the end of the flight. `plane_miss` is the distance from the pad in this plane."""
        c = self.config
        if timed_out:
            return c.timeout
        if result.aborted or not result.landed:
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
