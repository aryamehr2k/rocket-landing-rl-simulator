"""The PID baseline controller: lateral position and velocity to a tilt command, tilt to a gimbal command.

A positive world-plane command leans the nose toward the positive lateral axis of that plane (docs/conventions.md).
"""

import math

from rocketsim.estimator import Estimate
from rocketsim.guidance_config import PidConfig


class PlanePid:
    """Cascaded position and attitude loops for one plane."""

    def __init__(self, config: PidConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.tilt_command = 0.0

    def update(
        self, lateral: float, lateral_velocity: float, tilt: float, tilt_rate: float, dt: float, hold_position: bool
    ) -> float:
        """World-plane gimbal command in radians."""
        gains = self.config.attitude
        limit = self.config.max_tilt_command
        self.tilt_command = 0.0
        if hold_position:
            wanted = -(self.config.position_kp * lateral + self.config.position_kd * lateral_velocity)
            self.tilt_command = min(max(wanted, -limit), limit)
        error = self.tilt_command - tilt
        if gains.ki > 0.0:
            windup = gains.max_integral / gains.ki
            self.integral = min(max(self.integral + error * dt, -windup), windup)
        return gains.kp * error + gains.ki * self.integral - gains.kd * tilt_rate


class TvcController:
    """Two plane controllers sharing one configuration."""

    def __init__(self, config: PidConfig) -> None:
        self.pitch_plane = PlanePid(config)
        self.yaw_plane = PlanePid(config)

    def reset(self) -> None:
        self.pitch_plane.reset()
        self.yaw_plane.reset()

    def command(self, estimate: Estimate, dt: float, hold_position: bool) -> tuple[float, float]:
        """World-plane gimbal commands (toward +x, toward +y) from the estimate."""
        tilt_x, tilt_y = estimate.tilt
        rate_x, rate_y = estimate.tilt_rate
        position, velocity = estimate.position, estimate.velocity
        delta_x = self.pitch_plane.update(float(position[0]), float(velocity[0]), tilt_x, rate_x, dt, hold_position)
        delta_y = self.yaw_plane.update(float(position[1]), float(velocity[1]), tilt_y, rate_y, dt, hold_position)
        return delta_x, delta_y


def world_to_servo(delta_x: float, delta_y: float, roll: float) -> tuple[float, float]:
    """Rotate world-plane gimbal commands by -roll into pitch servo and yaw servo commands."""
    cos_roll, sin_roll = math.cos(-roll), math.sin(-roll)
    return cos_roll * delta_x - sin_roll * delta_y, sin_roll * delta_x + cos_roll * delta_y
