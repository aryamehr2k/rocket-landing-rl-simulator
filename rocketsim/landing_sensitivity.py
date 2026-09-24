"""How far the landing burn's stop point moves per error, and what that costs at touchdown.

Closed forms around the nominal one dimensional flight of rocketsim.landing_montecarlo, for the
design tool's sensitivity table, plus the single error draws that check them against that
model. The stop point is where the hard part of the burn has brought the descent down to the
target speed; the rest of the hard ramp lifts the rocket a little, then the tail sinks it.
"""

import math
from dataclasses import dataclass

import numpy as np

from rocketsim.landing_design import HALF, HARD_DROOP, BurnDesign, DescentSizing
from rocketsim.landing_montecarlo import Draws
from rocketsim.units import MS_PER_S

THRUST_STEP = 0.01  # relative motor strength error the sensitivity rows are quoted per
MASS_STEP = 0.01  # relative mass error the sensitivity rows are quoted per
DRAG_AREA_STEP = 0.10  # relative brake area error the sensitivity rows are quoted per
PERCENT = 100.0
MARGIN_SCAN_STEP = 0.01  # m steps when searching how high a stop the tail can still bring down
MARGIN_SCAN_LIMIT = 50.0  # m
ROW_NAMES = ("igniter delay", "thrust", "mass", "drag area", "speed estimate")


@dataclass(frozen=True)
class StopSensitivity:
    """How far the stop point moves for one unit of an error, and the touchdown speed that costs."""

    name: str
    shift: float
    touchdown_if_low: float
    touchdown_if_high: float


@dataclass(frozen=True)
class StopModel:
    """Closed forms around the nominal one dimensional flight: where it stops and how it sinks from there.

    After the stop the rest of the hard ramp lifts the rocket by `climb`; it then sinks from
    rest at `sink_accel`, gravity times one minus the tail thrust-to-weight, taken from the
    nominal flight's touchdown speed so the rising thrust-to-weight of a lightening rocket is
    included. A stop higher by `delta` sinks from `stop_height + delta + climb` for as long as
    the tail lasts, then falls freely. A stop lower by `delta` sinks from less height; once
    `delta` exceeds the stop height the hard part is still braking at `end_decel` when the feet
    touch, and the touchdown speed is sqrt(v_t^2 + 2 end_decel (delta - stop_height)).
    """

    target_speed: float
    stop_height: float
    climb: float
    sink_accel: float
    end_decel: float
    tail_time_left: float
    gravity: float

    @classmethod
    def from_nominal(
        cls, design: BurnDesign, gravity: float, target_speed: float, stop_height: float, climb: float,
        touchdown_speed: float, stop_time: float,
    ) -> "StopModel":
        sink_accel = touchdown_speed ** 2 / (2.0 * (stop_height + climb))
        end_decel = (design.hard_thrust - HARD_DROOP) / design.mass_at_tail_start - gravity
        return cls(target_speed, stop_height, climb, sink_accel, end_decel, design.tail_end - stop_time, gravity)

    def touchdown_if_low(self, delta: float) -> float:
        left = self.stop_height - delta
        if left >= 0.0:
            return math.sqrt(2.0 * self.sink_accel * (left + self.climb))
        return math.sqrt(self.target_speed ** 2 + 2.0 * self.end_decel * -left)

    def touchdown_if_high(self, delta: float) -> float:
        height, a = self.stop_height + delta + self.climb, self.sink_accel
        if a <= 0.0:
            return math.inf
        sink_time = math.sqrt(2.0 * height / a)
        if sink_time <= self.tail_time_left:
            return math.sqrt(2.0 * a * height)
        burnout_speed = a * self.tail_time_left
        fallen = HALF * a * self.tail_time_left ** 2
        return math.sqrt(burnout_speed ** 2 + 2.0 * self.gravity * (height - fallen))

    def margins(self, max_touchdown_speed: float) -> tuple[float, float]:
        """How far the stop may come out too low and too high before the touchdown speed exceeds the limit."""
        low = self.stop_height + (max_touchdown_speed ** 2 - self.target_speed ** 2) / (2.0 * self.end_decel)
        high = 0.0
        while self.touchdown_if_high(high + MARGIN_SCAN_STEP) <= max_touchdown_speed and high < MARGIN_SCAN_LIMIT:
            high += MARGIN_SCAN_STEP
        return low, high


def stop_sensitivities(
    design: BurnDesign, sizing: DescentSizing, speed: float, stopping_distance: float, model: StopModel,
    igniter_spread: float, speed_noise: float,
) -> list[StopSensitivity]:
    """Closed forms: shift = v dt (timing), d s (a + g) / a (thrust, mass), sigma_v v / a (speed estimate).

    `a` is the mean deceleration v^2 / (2 d) over the stopping distance d, `s` the relative
    error, and the rows follow `ROW_NAMES`.
    """
    mean_decel = speed ** 2 / (2.0 * stopping_distance)
    drag_at_arrival = sizing.drag_factor() * speed ** 2
    labels = (
        f"igniter delay +-{igniter_spread * MS_PER_S:.0f} ms", f"thrust {PERCENT * THRUST_STEP:.0f} %",
        f"mass {PERCENT * MASS_STEP:.0f} %", f"drag area {PERCENT * DRAG_AREA_STEP:.0f} %",
        f"speed estimate {speed_noise:.1f} m/s",
    )
    shifts = (
        speed * igniter_spread,
        stopping_distance * THRUST_STEP * design.hard_thrust / (design.mass_at_ignition * mean_decel),
        stopping_distance * MASS_STEP * (mean_decel + model.gravity) / mean_decel,
        stopping_distance * DRAG_AREA_STEP * HALF * drag_at_arrival / (design.mass_at_ignition * mean_decel),
        speed_noise * speed / mean_decel,
    )
    return [
        StopSensitivity(name, shift, model.touchdown_if_low(shift), model.touchdown_if_high(shift))
        for name, shift in zip(labels, shifts)
    ]


def single_error_draws(igniter_mean: float, descent_mass: float, igniter_spread: float, speed_noise: float) -> Draws:
    """Ten draws, one error each, in `ROW_NAMES` order: the direction that stops low, then the one that stops high.

    A late igniter, a weak motor, a heavy rocket, a small brake and a speed estimate that reads
    slow all stop the rocket lower than planned; their opposites stop it higher.
    """
    n = 2 * len(ROW_NAMES)
    igniter = np.full(n, igniter_mean)
    thrust, mass, area, height, speed = np.ones(n), np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n)
    igniter[0], igniter[1] = igniter_mean + igniter_spread, igniter_mean - igniter_spread
    thrust[2], thrust[3] = 1.0 - THRUST_STEP, 1.0 + THRUST_STEP
    mass[4], mass[5] = MASS_STEP * descent_mass, -MASS_STEP * descent_mass
    area[6], area[7] = 1.0 - DRAG_AREA_STEP, 1.0 + DRAG_AREA_STEP
    speed[8], speed[9] = -speed_noise, speed_noise
    return Draws(igniter, thrust, mass, area, height, speed)
