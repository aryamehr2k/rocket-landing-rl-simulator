"""When to light a solid landing motor: a stopping distance table built from its thrust curve.

It fires when the height, corrected for the igniter delay, has come down to the distance the burn needs.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np

from rocketsim.aero import air_density
from rocketsim.config import RocketConfig
from rocketsim.dragdevice import FULLY_OPEN, drag_factor, terminal_speed
from rocketsim.guidance_config import LandingTriggerConfig
from rocketsim.motors import MotorSpec
from rocketsim.simconfig import EnvironmentConfig

TABLE_MAX_SPEED = 80.0
TABLE_SPEED_STEP = 0.5
TABLE_DT = 0.002
HALF = 0.5
ARRIVAL_MARGIN = 1.0  # m/s between the expected arrival speed and the table's reach before a warning
CALIBRATION_WINDOW = 1.0  # s of accelerometer samples fitted before the drag is trusted
CALIBRATION_MIN_SPEED = 12.0  # m/s; slower than this the drag is too small against the sensor noise
CALIBRATION_MAX_TILT = math.radians(20.0)  # samples at a larger tilt are skipped: the drag is no longer axial
CALIBRATION_SPAN = 5.0  # m/s either side of the measured terminal speed that the table is rebuilt for


@dataclass(frozen=True)
class BurnResult:
    """Distance fallen, whether the target speed was reached, and when; arrays follow the inputs' shape."""

    distance: np.ndarray
    reached: np.ndarray
    time: np.ndarray


def integrate_burn(
    spec: MotorSpec,
    mass: float | np.ndarray,
    drag: float | np.ndarray,
    gravity: float,
    speed_down: float | np.ndarray,
    target_speed: float,
    thrust_scale: float | np.ndarray = 1.0,
    dt: float = TABLE_DT,
) -> BurnResult:
    """One dimensional burn from ignition: gravity, thrust times `thrust_scale`, drag `drag` v^2, mass flow.

    Every argument except the motor may be an array; they are broadcast together. The mass flow
    follows the curve, not the scaled thrust, as the physics does with the random thrust scale.
    """
    arrays = np.broadcast_arrays(speed_down, mass, drag, thrust_scale)
    speed, mass, drag, scale = (a.astype(float).copy() for a in arrays)
    distance, burned = np.zeros_like(speed), 0.0
    reached = np.zeros(speed.shape, dtype=bool)
    time = np.full(speed.shape, spec.burn_time)
    t = 0.0
    while t < spec.burn_time and not reached.all():
        curve_thrust = spec.thrust_at(t)
        current_mass = mass - burned
        accel_down = gravity - (curve_thrust * scale + drag * speed * np.abs(speed)) / current_mass
        next_speed = speed + accel_down * dt
        active = ~reached
        distance += np.where(active, HALF * (speed + next_speed) * dt, 0.0)
        speed = np.where(active, next_speed, speed)
        burned += curve_thrust / spec.exhaust_velocity * dt
        t += dt
        newly = active & (speed <= target_speed)
        reached |= newly
        time = np.where(newly, t, time)
    return BurnResult(np.maximum(distance, 0.0), reached, time)


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
        density = air_density(0.0, environment)
        self.drag_factor_body = HALF * density * rocket.aero.drag_coefficient * rocket.reference_area
        self.drag_factor_device = drag_factor(rocket.drag_device, FULLY_OPEN, density)
        brake = rocket.computer.brake
        self.burn_fraction = brake.burn_fraction if brake is not None else 0.0
        landing = rocket.motor("landing")
        self.spec = landing.spec
        self.ignition_delay = landing.ignition_delay_mean + decision_latency
        self.mass_at_ignition = rocket.descent_mass
        self.speeds = np.arange(0.0, TABLE_MAX_SPEED + TABLE_SPEED_STEP, TABLE_SPEED_STEP)
        self._distances, self.reachable = self._table(self.speeds)
        self._nominal = (self.drag_factor_device, self._distances.copy(), self.reachable.copy())
        self.reset()
        if brake is not None and self.can_reach_target_from - self.expected_arrival_speed < ARRIVAL_MARGIN:
            warnings.warn(
                f"landing trigger: the terminal speed {self.expected_arrival_speed:.1f} m/s is within "
                f"{ARRIVAL_MARGIN:g} m/s of the {self.can_reach_target_from:.1f} m/s the burn can stop; "
                "beyond it the burn leaves speed the tail cannot remove, so size the hard part for a higher speed",
                stacklevel=2,
            )

    def reset(self) -> None:
        """Forget an in-flight drag calibration: nominal device drag and table again."""
        self.drag_factor_device, distances, reachable = self._nominal
        self._distances, self.reachable = distances.copy(), reachable.copy()
        self.heights = self._capped(self._distances, self.reachable)
        self.calibrated = False
        self._fit_start: float | None = None
        self._fit_xy = self._fit_xx = 0.0

    @property
    def drag_factor(self) -> float:
        """Drag coefficient of speed squared assumed during the burn: body plus device at `burn_fraction`."""
        return self.drag_factor_at(self.burn_fraction)

    def drag_factor_at(self, brake_fraction: float) -> float:
        return self.drag_factor_body + self.drag_factor_device * brake_fraction

    @property
    def expected_arrival_speed(self) -> float:
        """Terminal speed with the device fully open, which a long enough descent reaches whatever the apogee."""
        return terminal_speed(self.mass_at_ignition, self.gravity, self.drag_factor_at(FULLY_OPEN))

    @property
    def can_reach_target_from(self) -> float:
        """Largest downward speed the burn can bring down to the target speed before burnout."""
        reachable = self.speeds[self.reachable]
        return float(reachable.max()) if reachable.size else 0.0

    def burn(self, speed_down: float | np.ndarray, thrust_scale: float | np.ndarray = 1.0) -> BurnResult:
        """The table's own one dimensional burn from `speed_down` with the flight computer's assumptions."""
        return integrate_burn(
            self.spec, self.mass_at_ignition, self.drag_factor, self.gravity, speed_down,
            self.config.target_speed, thrust_scale * self.config.thrust_margin,
        )

    def stopping_distance(self, speed_down: float) -> float:
        """Metres fallen from ignition until the target speed is reached or the motor burns out."""
        return float(self.burn(speed_down).distance)

    def required_height(self, speed_down: float | np.ndarray) -> float | np.ndarray:
        """Feet height above the ground at which the burn should already be producing thrust.

        Beyond `can_reach_target_from` the burn cannot bring the speed down to the target; the
        table then keeps the height of that edge, so the hard part still ends as low as it can
        with the least speed left, instead of jumping to the whole burn's fall distance.
        """
        return np.interp(speed_down, self.speeds, self.heights)

    def should_ignite(
        self, height: float | np.ndarray, speed_down: float | np.ndarray, brake_fraction: float = 0.0
    ) -> bool | np.ndarray:
        """Compare the height when thrust would start with the height the burn needs from that speed.

        The speed gained during the igniter delay counts gravity and the drag at the current
        device opening, which matters once the rocket falls at terminal speed.
        """
        delay = self.ignition_delay
        drag = self.drag_factor_at(brake_fraction)
        accel_down = self.gravity - drag * speed_down * np.abs(speed_down) / self.mass_at_ignition
        speed_at_ignition = speed_down + accel_down * delay
        height_at_ignition = height - HALF * (speed_down + speed_at_ignition) * delay
        decision = height_at_ignition <= self.required_height(speed_at_ignition)
        return decision if isinstance(decision, np.ndarray) else bool(decision)

    def observe_descent(self, t: float, speed_down: float, specific_force: float, tilt: float) -> bool:
        """Fit the drag from the accelerometer while the device is fully open, then rebuild the table.

        Falling tail first with no thrust, the body axis accelerometer reads the drag over the
        mass: `f = (k / m) v^2 cos(tilt)` with `v` the descent speed. A least squares fit of `f`
        against `v^2 cos(tilt)` over `CALIBRATION_WINDOW` seconds gives `k / m` directly, without
        waiting for terminal speed. Samples slower than `CALIBRATION_MIN_SPEED` or tilted more
        than `CALIBRATION_MAX_TILT` are skipped. Returns True on the step the calibration happened.
        """
        if self.calibrated or not self.config.calibrate_drag_in_flight:
            return False
        if speed_down < CALIBRATION_MIN_SPEED or abs(tilt) > CALIBRATION_MAX_TILT:
            return False
        if self._fit_start is None:
            self._fit_start = t
        x = speed_down ** 2 * math.cos(tilt)
        self._fit_xy += x * specific_force
        self._fit_xx += x * x
        if t - self._fit_start < CALIBRATION_WINDOW or self._fit_xx <= 0.0:
            return False
        self.calibrate_drag(self.mass_at_ignition * self._fit_xy / self._fit_xx)
        return True

    def calibrate_drag(self, total_drag_factor: float) -> None:
        """Set the device drag so the open rocket's total is `total_drag_factor`; rebuild the table near the arrival."""
        self.drag_factor_device = max(total_drag_factor - self.drag_factor_body, 0.0)
        near = np.abs(self.speeds - self.expected_arrival_speed) <= CALIBRATION_SPAN
        self._distances[near], self.reachable[near] = self._table(self.speeds[near])
        self.heights = self._capped(self._distances, self.reachable)
        self.calibrated = True

    def _table(self, speeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = self.burn(speeds)
        return result.distance + self.config.target_height, result.reached

    @staticmethod
    def _capped(distances: np.ndarray, reachable: np.ndarray) -> np.ndarray:
        """The required heights: the burn's own for reachable speeds, the edge value beyond them."""
        if not reachable.any():
            return distances.copy()
        return np.where(reachable, distances, distances[reachable].max())
