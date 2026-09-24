"""When to light a solid landing motor: a stopping distance table built from its thrust curve.

For every downward speed the table holds how far the rocket falls from ignition until the burn
has slowed it to the target speed (or the motor burns out). The trigger fires when the height
above the ground, corrected for the igniter delay, has come down to that distance.
"""

import numpy as np

from rocketsim.aero import air_density
from rocketsim.config import RocketConfig
from rocketsim.guidance_config import LandingTriggerConfig
from rocketsim.simconfig import EnvironmentConfig

TABLE_MAX_SPEED = 80.0
TABLE_SPEED_STEP = 0.5
TABLE_DT = 0.002
HALF = 0.5


class LandingTrigger:
    def __init__(
        self,
        rocket: RocketConfig,
        config: LandingTriggerConfig,
        environment: EnvironmentConfig,
        decision_latency: float = 0.0,
    ) -> None:
        """`decision_latency` is the control loop's own delay from deciding to the igniter command going out."""
        self.config = config
        self.gravity = environment.gravity
        # Drag at ground level air density; the few metres of the burn change it by nothing.
        self.drag_factor = HALF * air_density(0.0, environment) * rocket.aero.drag_coefficient * rocket.reference_area
        landing = rocket.motor("landing")
        self.spec = landing.spec
        self.ignition_delay = landing.ignition_delay_mean + decision_latency
        self.mass_at_ignition = (
            rocket.airframe.dry_mass + rocket.motor("ascent").spec.case_mass + landing.spec.total_mass
        )
        self.speeds = np.arange(0.0, TABLE_MAX_SPEED + TABLE_SPEED_STEP, TABLE_SPEED_STEP)
        self.heights = np.array([self.stopping_distance(v) for v in self.speeds]) + config.target_height

    @property
    def can_reach_target_from(self) -> float:
        """Largest downward speed the burn can bring down to the target speed before burnout."""
        reachable = [v for v in self.speeds if self._burn(v)[1]]
        return max(reachable) if reachable else 0.0

    def stopping_distance(self, speed_down: float) -> float:
        """Metres fallen from ignition until the target speed is reached or the motor burns out."""
        return self._burn(speed_down)[0]

    def required_height(self, speed_down: float) -> float:
        """Feet height above the ground at which the burn should already be producing thrust."""
        return float(np.interp(speed_down, self.speeds, self.heights))

    def should_ignite(self, height: float, speed_down: float) -> bool:
        """Compare the height when thrust would start with the height the burn needs from that speed."""
        delay = self.ignition_delay
        speed_at_ignition = speed_down + self.gravity * delay
        height_at_ignition = height - speed_down * delay - HALF * self.gravity * delay ** 2
        return height_at_ignition <= self.required_height(speed_at_ignition)

    def _burn(self, speed_down: float) -> tuple[float, bool]:
        # One dimensional burn with drag: (distance fallen, target speed reached before burnout).
        speed, distance, burned, t = speed_down, 0.0, 0.0, 0.0
        while t < self.spec.burn_time:
            curve_thrust = self.spec.thrust_at(t)
            thrust = curve_thrust * self.config.thrust_margin
            mass = self.mass_at_ignition - burned
            accel_down = self.gravity - (thrust + self.drag_factor * speed * abs(speed)) / mass
            next_speed = speed + accel_down * TABLE_DT
            distance += HALF * (speed + next_speed) * TABLE_DT
            burned += curve_thrust / self.spec.exhaust_velocity * TABLE_DT
            speed = next_speed
            t += TABLE_DT
            if speed <= self.config.target_speed:
                return max(distance, 0.0), True
        return max(distance, 0.0), False
