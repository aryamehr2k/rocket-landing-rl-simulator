"""Sizing a drag brake and a solid landing burn for a descent at terminal speed.

One dimensional, with the trigger's own burn integration (landing_trigger.integrate_burn), so design and table agree.
"""

from dataclasses import dataclass, replace

import numpy as np

from rocketsim.aero import air_density
from rocketsim.config import RocketConfig
from rocketsim.dragdevice import FULLY_OPEN, DragDeviceConfig, drag_area_for_terminal_speed, plate_area, terminal_speed
from rocketsim.guidance_config import BrakeConfig
from rocketsim.landing_montecarlo import LandingOutcome
from rocketsim.landing_trigger import integrate_burn
from rocketsim.motors import MotorSpec
from rocketsim.simconfig import EnvironmentConfig
from rocketsim.units import kg_to_g, mm_to_m

HALF = 0.5
PROFILE_DT = 0.005
RAMP_UP_TIME = 0.05  # s, the curve shape of the example motors
RAMP_UP_THRUST = 30.0  # N reached at RAMP_UP_TIME
HARD_START_TIME = 0.12  # s, full hard thrust from here
HARD_DROOP = 2.0  # N the hard part loses toward its end
HARD_TO_TAIL_RAMP = 0.15  # s from the hard part to the tail
TAIL_DROOP = 0.1  # N the tail loses toward burnout
BURNOUT_RAMP = 0.2  # s from tail thrust to zero
EXHAUST_VELOCITY = 1363.0  # m/s, the example landing motor's impulse per kilogram
CASE_MASS = 0.076  # kg of motor that is not propellant, the example landing motor's
SIZING_ITERATIONS = 8
SIZING_TOLERANCE = 0.005  # s
SIZING_STEP_WHEN_UNREACHED = 0.3  # s added to the hard part when the burn never reaches the target
TERMINAL_FRACTION = 0.95  # share of the terminal speed that counts as "arrived"
NO_BRAKE_AREA = 0.0  # a device Cd*A of zero means the rocket keeps falling freely
DEFAULT_STATION_MM = 150.0  # where a designed device sits when the rocket file has none: the nose
DEFAULT_DEPLOY_S, DEFAULT_RETRACT_S = 0.5, 0.3
DEFAULT_DEPLOY_SPEED, DEFAULT_RETRACT_SPEED = 8.0, 3.0  # m/s, the example rocket's brake rules


@dataclass(frozen=True)
class DescentSizing:
    mass: float
    weight: float
    density: float
    body_drag_area: float
    device_drag_area: float
    terminal_speed: float
    terminal_speed_low_drag: float
    terminal_speed_high_drag: float

    @property
    def total_drag_area(self) -> float:
        return self.body_drag_area + self.device_drag_area

    @property
    def plate_area(self) -> float:
        return plate_area(self.device_drag_area)

    def drag_factor(self, fraction: float = FULLY_OPEN, area_scale: float = 1.0) -> float:
        return HALF * self.density * (self.body_drag_area + self.device_drag_area * fraction * area_scale)


def size_descent(
    rocket: RocketConfig, environment: EnvironmentConfig, wanted_terminal_speed: float | None,
    device_drag_area: float | None, drag_area_error: float,
) -> DescentSizing:
    """Device Cd*A for a wanted terminal speed, or the terminal speed of a given device, with the error band."""
    mass, gravity = rocket.descent_mass, environment.gravity
    density = air_density(0.0, environment)
    body = rocket.aero.drag_coefficient * rocket.reference_area
    if device_drag_area is None:
        if wanted_terminal_speed is None:
            raise ValueError("give a terminal speed or a device drag area")
        device_drag_area = max(drag_area_for_terminal_speed(mass, gravity, density, wanted_terminal_speed) - body, 0.0)
    speeds = [
        terminal_speed(mass, gravity, HALF * density * (body + device_drag_area * scale))
        for scale in (1.0, 1.0 - drag_area_error, 1.0 + drag_area_error)
    ]
    return DescentSizing(mass, mass * gravity, density, body, device_drag_area, *speeds)


def ascent_apogee(rocket: RocketConfig, environment: EnvironmentConfig, dt: float = PROFILE_DT) -> float:
    """Feet height at apogee from a vertical one dimensional boost and coast with the body drag."""
    ascent, landing = rocket.motor("ascent").spec, rocket.motor("landing").spec
    drag = HALF * air_density(0.0, environment) * rocket.aero.drag_coefficient * rocket.reference_area
    height, speed, burned, t = 0.0, 0.0, 0.0, 0.0
    while True:
        thrust = ascent.thrust_at(t)
        mass = rocket.airframe.dry_mass + ascent.total_mass - burned + landing.total_mass
        accel = (thrust - drag * speed * abs(speed)) / mass - environment.gravity
        if height <= 0.0 and accel <= 0.0:
            burned += thrust / ascent.exhaust_velocity * dt
            t += dt
            continue
        next_speed = speed + accel * dt
        height += HALF * (speed + next_speed) * dt
        burned += thrust / ascent.exhaust_velocity * dt
        speed, t = next_speed, t + dt
        if speed <= 0.0 and t > ascent.burn_time:
            return height


def descent_profile(
    sizing: DescentSizing, gravity: float, apogee: float, device: DragDeviceConfig | None,
    deploy_speed: float | None, dt: float = PROFILE_DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Feet height and downward speed from apogee to the ground with the brake rule, no burn."""
    heights, speeds = [apogee], [0.0]
    height, speed, fraction = apogee, 0.0, 0.0
    while height > 0.0:
        if device is not None and deploy_speed is not None and (speed > deploy_speed or fraction > 0.0):
            fraction = min(FULLY_OPEN, fraction + dt / device.deploy_time)
        accel = gravity - sizing.drag_factor(fraction) * speed * abs(speed) / sizing.mass
        next_speed = speed + accel * dt
        height -= HALF * (speed + next_speed) * dt
        speed = next_speed
        heights.append(height)
        speeds.append(speed)
    return np.array(heights), np.array(speeds)


def speed_at_heights(heights: np.ndarray, speeds: np.ndarray, marks: list[float]) -> list[float]:
    """Downward speed when the feet pass each mark, heights decreasing along the profile."""
    return [float(np.interp(-mark, -heights, speeds)) for mark in marks]


def height_where_speed_reaches(heights: np.ndarray, speeds: np.ndarray, speed: float) -> float:
    index = int(np.argmax(speeds >= speed))
    return float(heights[index]) if speeds[index] >= speed else 0.0


def build_curve(
    hard_thrust: float, hard_end: float, tail_thrust: float, tail_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """The example motors' shape: fast ramp, hard part, short ramp, flat tail, burnout."""
    points = [
        (0.0, 0.0), (RAMP_UP_TIME, min(RAMP_UP_THRUST, hard_thrust)), (HARD_START_TIME, hard_thrust),
        (hard_end, hard_thrust - HARD_DROOP), (hard_end + HARD_TO_TAIL_RAMP, tail_thrust),
        (tail_end, tail_thrust - TAIL_DROOP), (tail_end + BURNOUT_RAMP, 0.0),
    ]
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def motor_from_curve(name: str, times: np.ndarray, thrusts: np.ndarray, spread: MotorSpec) -> MotorSpec:
    """A MotorSpec for a designed curve; propellant at the exhaust velocity, case mass and igniter of `spread`."""
    impulse = float(np.sum(np.diff(times) * (thrusts[1:] + thrusts[:-1]) * HALF))
    propellant = impulse / EXHAUST_VELOCITY
    return replace(
        spread, name=name, times=times, thrusts=thrusts, propellant_mass=propellant,
        total_mass=propellant + CASE_MASS,
    )


@dataclass(frozen=True)
class BurnDesign:
    motor: MotorSpec
    hard_thrust: float
    hard_end: float
    tail_thrust: float
    tail_end: float
    design_speed: float
    mass_at_ignition: float
    mass_at_tail_start: float

    @property
    def hard_impulse(self) -> float:
        return self.motor.impulse_at(self.hard_end + HARD_TO_TAIL_RAMP)


def size_burn(
    rocket: RocketConfig, sizing: DescentSizing, gravity: float, hard_thrust: float, tail_ratio: float,
    tail_duration: float, design_speed: float, target_speed: float, burn_fraction: float,
) -> BurnDesign:
    """Hard part that brings `design_speed` to `target_speed` exactly at its end; tail at `tail_ratio` of the weight."""
    template = rocket.motor("landing").spec
    mass = rocket.airframe.dry_mass + rocket.motor("ascent").spec.case_mass
    hard_end, tail_thrust = 1.0, tail_ratio * sizing.weight
    drag = sizing.drag_factor(burn_fraction)
    for _ in range(SIZING_ITERATIONS):
        times, thrusts = build_curve(hard_thrust, hard_end, tail_thrust, hard_end + HARD_TO_TAIL_RAMP + tail_duration)
        motor = motor_from_curve(template.name, times, thrusts, template)
        mass_at_ignition = mass + motor.total_mass
        tail_start = hard_end + HARD_TO_TAIL_RAMP
        mass_at_tail = mass_at_ignition - motor.propellant_burned_at(tail_start)
        tail_thrust = tail_ratio * mass_at_tail * gravity
        result = integrate_burn(motor, mass_at_ignition, drag, gravity, design_speed, target_speed)
        if not bool(result.reached):
            hard_end += SIZING_STEP_WHEN_UNREACHED
            continue
        reached_at = float(result.time)
        if abs(reached_at - hard_end) < SIZING_TOLERANCE:
            break
        hard_end = reached_at
    return BurnDesign(
        motor, hard_thrust, hard_end, tail_thrust, hard_end + HARD_TO_TAIL_RAMP + tail_duration, design_speed,
        mass_at_ignition, mass_at_tail,
    )


def design_from_motor(spec: MotorSpec, mass_at_ignition: float, design_speed: float) -> BurnDesign:
    """Describe an existing motor file in BurnDesign terms: hard part at its peak thrust, tail from the last plateau."""
    hard_end = float(spec.times[np.argmax(spec.thrusts)])
    for time, thrust in zip(spec.times, spec.thrusts):
        if thrust >= spec.thrusts.max() - HARD_DROOP:
            hard_end = float(time)
    tail_end = float(spec.times[-2])
    tail_thrust = float(spec.thrust_at(HALF * (hard_end + HARD_TO_TAIL_RAMP + tail_end)))
    mass_at_tail = mass_at_ignition - spec.propellant_burned_at(hard_end + HARD_TO_TAIL_RAMP)
    return BurnDesign(
        spec, float(spec.thrusts.max()), hard_end, tail_thrust, tail_end, design_speed, mass_at_ignition, mass_at_tail
    )


def burn_profile(
    spec: MotorSpec, mass: float, drag: float, gravity: float, height: float, speed: float, marks: list[float],
    dt: float = PROFILE_DT,
) -> list[tuple[float, float]]:
    """Downward speed when the feet pass each mark during a nominal burn from thrust on to the ground."""
    passed: list[tuple[float, float]] = []
    burned, t = 0.0, 0.0
    remaining = [m for m in marks if m < height]
    while height > 0.0 and remaining:
        thrust = spec.thrust_at(t)
        accel = gravity - (thrust + drag * speed * abs(speed)) / (mass - burned)
        next_speed = speed + accel * dt
        height -= HALF * (speed + next_speed) * dt
        burned += thrust / spec.exhaust_velocity * dt
        speed, t = next_speed, t + dt
        while remaining and height <= remaining[0]:
            passed.append((remaining.pop(0), speed))
    return passed


def curve_yaml(design: BurnDesign) -> str:
    """The thrust curve as lines for a motor YAML."""
    lines = [
        f"propellant_mass_g: {kg_to_g(design.motor.propellant_mass):.0f}",
        f"total_mass_g: {kg_to_g(design.motor.total_mass):.0f}",
        "thrust_curve:",
    ]
    lines += [f"  - [{t:.2f}, {f:.1f}]" for t, f in zip(design.motor.times, design.motor.thrusts)]
    return "\n".join(lines)


def design_rocket(rocket: RocketConfig, sizing: DescentSizing, design: BurnDesign | None) -> RocketConfig:
    """The rocket with the designed device, brake rules and motor filled in where the file had none."""
    motors = rocket.motors
    if design is not None:
        motors = tuple(replace(m, spec=design.motor) if m.role == "landing" else m for m in motors)
    if rocket.drag_device is None and sizing.device_drag_area <= NO_BRAKE_AREA:
        return replace(rocket, motors=motors)
    device = rocket.drag_device or DragDeviceConfig(
        sizing.device_drag_area, mm_to_m(DEFAULT_STATION_MM), DEFAULT_DEPLOY_S, DEFAULT_RETRACT_S
    )
    device = replace(device, drag_area=sizing.device_drag_area)
    brake = rocket.computer.brake or BrakeConfig(DEFAULT_DEPLOY_SPEED, DEFAULT_RETRACT_SPEED, FULLY_OPEN)
    return replace(rocket, drag_device=device, computer=replace(rocket.computer, brake=brake), motors=motors)
