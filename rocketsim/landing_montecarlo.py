"""Many one dimensional landings at once: the descent, the brake rules, the trigger and the burn.

Vertical only, so no wind or tilt; the trigger and brake rules run on noisy estimates with the nominal rocket.
"""

from dataclasses import dataclass

import numpy as np

from rocketsim.config import RocketConfig
from rocketsim.dragdevice import FULLY_OPEN, SHUT
from rocketsim.landing_trigger import LandingTrigger

HALF = 0.5
MAX_TIME = 40.0  # s from apogee; longer than any landing, guards the loop


@dataclass(frozen=True)
class ErrorBudget:
    """Half widths of uniform hidden errors and standard deviations of the estimator's noise."""

    igniter_spread: float = 0.03
    thrust_error: float = 0.02
    mass_error: float = 0.012
    drag_area_error: float = 0.05
    height_noise: float = 0.12
    speed_noise: float = 0.3

    def without(self, name: str) -> "ErrorBudget":
        """The budget with the error `name` set to zero (the estimator's two count as one)."""
        drop = {"estimator": ("height_noise", "speed_noise")}.get(name, (name,))
        return ErrorBudget(**{field: (0.0 if field in drop else getattr(self, field)) for field in self.__dataclass_fields__})


@dataclass(frozen=True)
class Draws:
    igniter_delay: np.ndarray
    thrust_scale: np.ndarray
    mass_offset: np.ndarray
    drag_area_scale: np.ndarray
    height_error: np.ndarray
    speed_error: np.ndarray


def draw_errors(budget: ErrorBudget, igniter_mean: float, count: int, rng: np.random.Generator) -> Draws:
    return Draws(
        igniter_delay=np.maximum(rng.uniform(igniter_mean - budget.igniter_spread, igniter_mean + budget.igniter_spread, count), 0.0),
        thrust_scale=rng.uniform(1.0 - budget.thrust_error, 1.0 + budget.thrust_error, count),
        mass_offset=rng.uniform(-budget.mass_error, budget.mass_error, count),
        drag_area_scale=rng.uniform(1.0 - budget.drag_area_error, 1.0 + budget.drag_area_error, count),
        height_error=budget.height_noise * rng.standard_normal(count),
        speed_error=budget.speed_noise * rng.standard_normal(count),
    )


def nominal_draws(igniter_mean: float, thrust_scale: np.ndarray) -> Draws:
    """No errors except a given motor strength per draw, for sweeps."""
    zeros = np.zeros_like(thrust_scale)
    return Draws(np.full_like(thrust_scale, igniter_mean), thrust_scale, zeros, np.ones_like(thrust_scale), zeros, zeros)


@dataclass(frozen=True)
class LandingOutcome:
    """One entry per draw; nan where something never happened."""

    command_height: np.ndarray
    command_speed: np.ndarray
    thrust_on_time: np.ndarray
    thrust_on_height: np.ndarray
    thrust_on_speed: np.ndarray
    stop_height: np.ndarray
    stop_time: np.ndarray
    climb_after_stop: np.ndarray
    touchdown_speed: np.ndarray
    touchdown_time: np.ndarray
    landed: np.ndarray


def simulate_landings(
    rocket: RocketConfig, trigger: LandingTrigger, apogee: float, draws: Draws, control_dt: float, dt: float
) -> LandingOutcome:
    """Fly every draw from apogee to the ground in one dimension with the flight computer's rules."""
    spec, device, brake = trigger.spec, rocket.drag_device, rocket.computer.brake
    gravity, target_speed = trigger.gravity, trigger.config.target_speed
    count = draws.thrust_scale.size
    steps_per_control = max(int(round(control_dt / dt)), 1)
    action_delay = rocket.control.action_delay_steps * control_dt
    mass = rocket.descent_mass + draws.mass_offset
    device_drag = trigger.drag_factor_device * draws.drag_area_scale
    height, speed, fraction, burned = np.full(count, float(apogee)), np.zeros(count), np.zeros(count), np.zeros(count)
    target_fraction, commanded_fraction = np.zeros(count), np.zeros(count)
    deployed, retracted, commanded, touched = (np.zeros(count, dtype=bool) for _ in range(4))
    ignition_time = np.full(count, np.nan)
    nan = np.full(count, np.nan)
    out = {name: nan.copy() for name in ("command_height", "command_speed", "thrust_on_time", "thrust_on_height", "thrust_on_speed", "stop_height", "stop_time", "touchdown_speed", "touchdown_time")}
    peak = nan.copy()
    t, step = 0.0, 0
    while t < MAX_TIME and not touched.all():
        if step % steps_per_control == 0:
            height_est, speed_est = height + draws.height_error, speed + draws.speed_error
            descending = speed > 0.0
            if brake is not None and device is not None:
                deployed |= descending & ~commanded & (speed_est > brake.deploy_descent_speed)
                retracted |= commanded & (speed_est < brake.retract_descent_speed)
                in_burn = np.where(commanded, brake.burn_fraction, FULLY_OPEN)
                commanded_fraction = np.where(retracted, SHUT, np.where(deployed, in_burn, SHUT))
            fire = descending & ~commanded & trigger.should_ignite(height_est, speed_est, commanded_fraction)
            commanded |= fire
            ignition_time = np.where(fire, t + action_delay + draws.igniter_delay, ignition_time)
            out["command_height"] = np.where(fire, height, out["command_height"])
            out["command_speed"] = np.where(fire, speed, out["command_speed"])
            target_fraction = commanded_fraction
        if device is not None:
            opening = np.minimum(target_fraction, fraction + dt / device.deploy_time)
            closing = np.maximum(target_fraction, fraction - dt / device.retract_time)
            fraction = np.where(target_fraction > fraction, opening, closing)
        since = t - ignition_time
        burning = since >= 0.0
        curve = np.where(burning, np.interp(np.nan_to_num(since), spec.times, spec.thrusts), 0.0)
        drag = trigger.drag_factor_body + device_drag * fraction
        accel_down = gravity - (curve * draws.thrust_scale + drag * speed * np.abs(speed)) / (mass - burned)
        next_speed = speed + accel_down * dt
        next_height = height - HALF * (speed + next_speed) * dt
        burned += curve / spec.exhaust_velocity * dt
        first_thrust = burning & np.isnan(out["thrust_on_height"])
        out["thrust_on_time"] = np.where(first_thrust, t, out["thrust_on_time"])
        out["thrust_on_height"] = np.where(first_thrust, height, out["thrust_on_height"])
        out["thrust_on_speed"] = np.where(first_thrust, speed, out["thrust_on_speed"])
        stopping = burning & ~touched & np.isnan(out["stop_height"]) & (next_speed <= target_speed)
        out["stop_height"] = np.where(stopping, next_height, out["stop_height"])
        out["stop_time"] = np.where(stopping, t + dt - ignition_time, out["stop_time"])
        peak = np.where(stopping, next_height, np.where(~touched, np.fmax(peak, next_height), peak))
        landing_now = ~touched & (next_height <= 0.0)
        out["touchdown_speed"] = np.where(landing_now, next_speed, out["touchdown_speed"])
        out["touchdown_time"] = np.where(landing_now, t + dt, out["touchdown_time"])
        height = np.where(touched, height, next_height)
        speed = np.where(touched, speed, next_speed)
        touched |= landing_now
        t, step = t + dt, step + 1
    climb = np.where(np.isnan(out["stop_height"]), np.nan, peak - out["stop_height"])
    landed = out["touchdown_speed"] <= rocket.legs.max_vertical_speed
    return LandingOutcome(**out, climb_after_stop=climb, landed=landed)
