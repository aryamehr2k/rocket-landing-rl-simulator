"""Reward terms for the landing policy, per control plane, weighted from the training YAML (penalties negative)."""

import math
from dataclasses import dataclass

from rocketsim.touchdown import TouchdownResult
from rocketsim.units import rad_to_deg


@dataclass(frozen=True)
class RewardConfig:
    """Weights: touchdown terms are paid once, step terms every control step in a controlled phase."""

    touchdown_vertical_speed: float = 0.0
    touchdown_lateral_speed: float = 0.0
    touchdown_tilt_deg: float = 0.0
    miss_distance: float = 0.0
    crash: float = 0.0
    landed: float = 0.0
    timeout: float = 0.0
    tilt_step: float = 0.0
    lateral_speed_step: float = 0.0
    lateral_position_step: float = 0.0
    gimbal_change_step: float = 0.0
    gimbal_step: float = 0.0
    propellant_step: float = 0.0


class RewardCalculator:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def step(
        self, controlled: bool, tilt: float, lateral_speed: float, lateral_position: float,
        gimbal_change: float, propellant_burned: float, gimbal: float = 0.0,
    ) -> float:
        """Reward for one control step. Shaping terms only count while the policy steers."""
        c = self.config
        reward = c.propellant_step * propellant_burned
        if controlled:
            reward += c.tilt_step * abs(tilt)
            reward += c.lateral_speed_step * abs(lateral_speed)
            reward += c.lateral_position_step * abs(lateral_position)
            reward += c.gimbal_change_step * gimbal_change ** 2
            reward += c.gimbal_step * abs(gimbal)
        return reward

    def terminal(
        self, touchdown: TouchdownResult | None, plane_lateral_speed: float, plane_lateral_position: float,
    ) -> float:
        """Reward paid once when the flight ends. No touchdown means the time limit ran out."""
        c = self.config
        if touchdown is None:
            return c.timeout
        reward = c.touchdown_vertical_speed * touchdown.vertical_speed
        reward += c.touchdown_lateral_speed * abs(plane_lateral_speed)
        reward += c.touchdown_tilt_deg * rad_to_deg(touchdown.tilt)
        reward += c.miss_distance * abs(plane_lateral_position)
        reward += c.landed if touchdown.success else c.crash
        return reward if math.isfinite(reward) else c.crash
