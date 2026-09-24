"""Design the drag brake and the solid landing burn for a rocket, and estimate the landing rate.

The rocket falls at terminal speed once the brake is open, so the landing motor's hard part can
be sized for one arrival speed. This tool prints, with how each number was computed: the brake
area for a wanted terminal speed, the speed profile, the hard part and tail of the motor, the
trigger numbers, the sensitivity of the stop point to every error, the window of motor strength
that still lands, a one dimensional Monte Carlo landing rate, optionally real simulator flights,
and which three things to measure first.

Usage: python scripts/design_landing_burn.py --rocket configs/rockets/example_tvc.yaml --terminal-speed 20
A rocket without a drag_device section needs --terminal-speed (size a brake) or --drag-area-cm2
(0 checks the free fall as it is).
"""

import argparse
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np

from rocketsim.config import RocketConfig, load_rocket_config
from rocketsim.flightcomputer import HALF_STEP
from rocketsim.landing_design import (
    HARD_TO_TAIL_RAMP, NO_BRAKE_AREA, TERMINAL_FRACTION, BurnDesign, DescentSizing, LandingOutcome, ascent_apogee,
    burn_profile, curve_yaml, descent_profile, design_from_motor, design_rocket, height_where_speed_reaches, size_burn,
    size_descent, speed_at_heights,
)
from rocketsim.landing_montecarlo import ErrorBudget, draw_errors, nominal_draws, simulate_landings
from rocketsim.landing_sensitivity import StopModel, single_error_draws, stop_sensitivities
from rocketsim.landing_trigger import LandingTrigger
from rocketsim.motors import load_motor
from rocketsim.simconfig import SimConfig, load_sim_config
from rocketsim.simulation import Simulation
from rocketsim.units import CM2_PER_M2, MS_PER_S, cm2_to_m2, g_to_kg, kg_to_g

DEFAULT_SIM = "configs/training/default.yaml"
DEFAULT_WINDY = "configs/training/windy.yaml"
PROFILE_MARKS = [100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 15.0, 10.0, 5.0, 3.0, 2.0, 1.0]
SWEEP_PERCENT = np.arange(-8, 9)
ERROR_SOURCES = ("igniter_spread", "thrust_error", "mass_error", "drag_area_error", "estimator")
ERROR_LABELS = {
    "igniter_spread": "igniter delay spread (fire 20 igniters with your own circuit)",
    "thrust_error": "landing motor strength spread (load cell, several motors of the batch)",
    "mass_error": "rocket mass (weigh it before every flight and enter it)",
    "drag_area_error": "brake drag area (drop test or log the terminal speed)",
    "estimator": "height and speed estimate noise (barometer and IMU quality)",
}
PERCENT = 100.0
PER_THOUSAND = 1000.0
TOP_MEASUREMENTS = 3
SIM_SEED = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rocket", required=True)
    parser.add_argument("--sim", default=DEFAULT_SIM, help="training YAML for gravity and air density")
    parser.add_argument("--motor", help="use this landing motor file as it is instead of sizing one")
    parser.add_argument("--hard-n", type=float, default=31.0, help="hard part thrust when sizing a motor")
    parser.add_argument("--tail-ratio", type=float, default=0.92, help="tail thrust over the weight at tail start")
    parser.add_argument("--tail-s", type=float, default=6.5, help="tail duration")
    parser.add_argument("--terminal-speed", type=float,
                        help="wanted arrival speed; sizes the brake area if none is given")
    parser.add_argument("--drag-area-cm2", type=float,
                        help="brake Cd*A; default from the rocket file or the terminal speed; 0 means no brake")
    parser.add_argument("--design-margin-mps", type=float, default=0.3,
                        help="added to the terminal speed when sizing the hard part")
    parser.add_argument("--igniter-spread-s", type=float, default=0.03)
    parser.add_argument("--thrust-error", type=float, default=0.02,
                        help="half width of the motor strength error, as a fraction")
    parser.add_argument("--mass-error-g", type=float, default=12.0)
    parser.add_argument("--drag-area-error", type=float, default=0.05,
                        help="half width of the brake area error, as a fraction")
    parser.add_argument("--draws", type=int, default=2000, help="one dimensional Monte Carlo flights")
    parser.add_argument("--sim-flights", type=int, default=0, help="real simulator flights per case (calm and windy)")
    parser.add_argument("--wind", default=DEFAULT_WINDY, help="training YAML for the windy simulator case")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.drag_area_cm2 is None and args.terminal_speed is None:
        rocket = load_rocket_config(args.rocket)
        if rocket.drag_device is None:
            parser.error("give --terminal-speed or --drag-area-cm2 for a rocket without a drag_device section")
    return args


def print_descent(
    sizing: DescentSizing, rocket: RocketConfig, apogee: float, profile: tuple[np.ndarray, np.ndarray],
    command_height: float,
) -> None:
    print("1. Descent at terminal speed")
    print(f"   mass at landing ignition {sizing.mass:.3f} kg (dry + empty ascent case + landing motor), "
          f"weight {sizing.weight:.1f} N")
    print(f"   body Cd*A {sizing.body_drag_area * CM2_PER_M2:.1f} cm2, brake Cd*A "
          f"{sizing.device_drag_area * CM2_PER_M2:.0f} cm2 (flat plate at Cd 1.2: "
          f"{sizing.plate_area * CM2_PER_M2:.0f} cm2), total {sizing.total_drag_area * CM2_PER_M2:.0f} cm2; "
          f"from Cd*A = 2 m g / (rho v_t^2) with rho {sizing.density:.3f} kg/m3")
    print(f"   terminal speed {sizing.terminal_speed:.1f} m/s (drag = weight); "
          f"{sizing.terminal_speed_low_drag:.1f} m/s if the brake area is at the low end of its error band, "
          f"{sizing.terminal_speed_high_drag:.1f} m/s at the high end")
    heights, speeds = profile
    marks = [m for m in PROFILE_MARKS if command_height <= m < apogee]
    brake = rocket.computer.brake
    opens = f"brake opens above {brake.deploy_descent_speed:g} m/s" if brake is not None else "no brake"
    print(f"   apogee {apogee:.1f} m (1D vertical boost and coast); {opens}")
    at_marks = speed_at_heights(heights, speeds, marks)
    speeds_text = ", ".join(f"{m:.0f} m: {v:.1f}" for m, v in zip(marks, at_marks))
    print(f"   feet height to descent speed: {speeds_text} m/s")
    arrived = height_where_speed_reaches(heights, speeds, TERMINAL_FRACTION * sizing.terminal_speed)
    print(f"   {PERCENT * TERMINAL_FRACTION:.0f} % of terminal speed reached at {arrived:.0f} m "
          "(must be above the ignition height; 0 means never)")


def print_burn(design: BurnDesign, gravity: float, sizing: DescentSizing, sized_here: bool) -> None:
    motor = design.motor
    print("2. Landing motor" + ("" if sized_here else f" (as in {motor.name}, not sized here)"))
    origin = "terminal speed + margin" if sized_here else "terminal speed"
    print(f"   design speed {design.design_speed:.1f} m/s = {origin}; hard part {design.hard_thrust:.1f} N "
          f"({design.hard_thrust / sizing.weight:.2f} times the weight) to {design.hard_end:.2f} s brings it to "
          "the target speed at its end (1D burn with the brake drag, iterated on the end time)")
    tail_weight = design.mass_at_tail_start * gravity
    print(f"   tail {design.tail_thrust:.1f} N = {design.tail_thrust / tail_weight:.2f} of the {tail_weight:.1f} N "
          f"weight at tail start, to {design.tail_end:.1f} s")
    print(f"   impulse {motor.total_impulse:.0f} N s ({design.hard_impulse:.0f} hard, "
          f"{motor.total_impulse - design.hard_impulse:.0f} tail), propellant {kg_to_g(motor.propellant_mass):.0f} g "
          f"at {motor.exhaust_velocity:.0f} m/s, motor {kg_to_g(motor.total_mass):.0f} g")
    print("   " + curve_yaml(design).replace("\n", "\n   "))


def print_trigger(trigger: LandingTrigger, nominal: LandingOutcome, design: BurnDesign) -> None:
    print("3. Trigger")
    command_speed = float(nominal.command_speed[0])
    print(f"   the table can stop at most {trigger.can_reach_target_from:.1f} m/s; expected arrival "
          f"{trigger.expected_arrival_speed:.1f} m/s; stopping distance from {command_speed:.1f} m/s: "
          f"{trigger.stopping_distance(command_speed):.1f} m")
    print(f"   command at {nominal.command_height[0]:.1f} m and {command_speed:.1f} m/s; thrust on at "
          f"{nominal.thrust_on_height[0]:.1f} m and {nominal.thrust_on_speed[0]:.1f} m/s after the "
          f"{trigger.ignition_delay:.2f} s of igniter delay and loop latency")
    sink_time = nominal.touchdown_time[0] - nominal.thrust_on_time[0] - nominal.stop_time[0]
    print(f"   hand-over (target speed) at {nominal.stop_height[0]:.2f} m, {nominal.stop_time[0]:.2f} s after "
          f"thrust on; climb after it {nominal.climb_after_stop[0]:.2f} m; touchdown "
          f"{nominal.touchdown_speed[0]:.2f} m/s, {sink_time:.1f} s after the stop")
    tail_available = design.tail_end - design.hard_end - HARD_TO_TAIL_RAMP
    print(f"   tail available {tail_available:.1f} s; the tail must outlast the sink from the highest stop it can "
          "survive (1D nominal flight)")
    marks = [m for m in PROFILE_MARKS if m < nominal.thrust_on_height[0]]
    profile = burn_profile(
        trigger.spec, trigger.mass_at_ignition, trigger.drag_factor, trigger.gravity,
        float(nominal.thrust_on_height[0]), float(nominal.thrust_on_speed[0]), marks,
    )
    print("   during the burn, feet height to descent speed: "
          + ", ".join(f"{m:.0f} m: {v:.1f}" for m, v in profile) + " m/s")


def print_sensitivities(
    design: BurnDesign, sizing: DescentSizing, nominal: LandingOutcome, checked: LandingOutcome,
    trigger: LandingTrigger, args: argparse.Namespace, rocket: RocketConfig,
) -> None:
    speed = float(nominal.thrust_on_speed[0])
    model = StopModel.from_nominal(
        design, trigger.gravity, trigger.config.target_speed, float(nominal.stop_height[0]),
        float(nominal.climb_after_stop[0]), float(nominal.touchdown_speed[0]), float(nominal.stop_time[0]),
    )
    rows = stop_sensitivities(
        design, sizing, speed, trigger.stopping_distance(speed), model, args.igniter_spread_s,
        ErrorBudget().speed_noise,
    )
    low_margin, high_margin = model.margins(rocket.legs.max_vertical_speed)
    print("4. Stop point sensitivity (closed forms in rocketsim/landing_sensitivity.py, checked against the 1D model)")
    print(f"   nominal stop {model.stop_height:.2f} m up, then a {model.climb:.2f} m climb on the rest of the hard "
          f"ramp and a sink at {model.sink_accel:.2f} m/s2 on the tail, which has {model.tail_time_left:.1f} s left")
    print(f"   allowed stop error: {low_margin:.2f} m too low, {high_margin:.2f} m too high, before the touchdown "
          f"speed exceeds {rocket.legs.max_vertical_speed:g} m/s")
    print("   error                        stop moves   touchdown if the stop is low   if high   "
          "(closed form / 1D model)")
    for index, row in enumerate(rows):
        low, high = checked.touchdown_speed[2 * index], checked.touchdown_speed[2 * index + 1]
        print(f"   {row.name:28s} {row.shift:5.2f} m     {row.touchdown_if_low:4.1f} / {low:4.1f} m/s"
              f"                {row.touchdown_if_high:4.1f} / {high:4.1f} m/s")
    step = speed / rocket.control.control_rate_hz
    print(f"   where the two differ, trust the 1D model: its trigger decides once per control step ({step:.2f} m of "
          "fall at this speed) and its climb after the stop grows with the error; the closed form keeps both nominal")


def print_window(rocket: RocketConfig, trigger: LandingTrigger, apogee: float, control_dt: float, dt: float,
                 args: argparse.Namespace) -> None:
    scales = 1.0 + SWEEP_PERCENT / PERCENT
    igniter_mean = rocket.motor("landing").ignition_delay_mean
    outcome = simulate_landings(rocket, trigger, apogee, nominal_draws(igniter_mean, scales), control_dt, dt)
    landed = SWEEP_PERCENT[outcome.landed]
    print("5. Motor strength window (whole curve scaled, 1D, no other errors)")
    print("   " + "  ".join(f"{p:+d}%:{'ok' if ok else f'{v:.0f}m/s'}"
                            for p, ok, v in zip(SWEEP_PERCENT, outcome.landed, outcome.touchdown_speed)))
    if landed.size:
        print(f"   lands from {landed.min():+d} % to {landed.max():+d} %; set landing_trigger.thrust_margin to "
              f"1 + half your batch spread ({1.0 + args.thrust_error:.3f}) so every real motor is nominal or weaker "
              "than the table assumes")


def monte_carlo(rocket: RocketConfig, trigger: LandingTrigger, apogee: float, budget: ErrorBudget, control_dt: float,
                dt: float, args: argparse.Namespace) -> LandingOutcome:
    rng = np.random.default_rng(args.seed)
    draws = draw_errors(budget, rocket.motor("landing").ignition_delay_mean, args.draws, rng)
    return simulate_landings(rocket, trigger, apogee, draws, control_dt, dt)


def describe_outcome(outcome: LandingOutcome) -> str:
    speeds = outcome.touchdown_speed[np.isfinite(outcome.touchdown_speed)]
    stops = outcome.stop_height[np.isfinite(outcome.stop_height)]
    return (f"landed {int(outcome.landed.sum())} of {outcome.landed.size} "
            f"({PERCENT * outcome.landed.mean():.0f} %); touchdown median {np.median(speeds):.2f} max "
            f"{speeds.max():.1f} m/s; stop height {stops.min():.2f} to {stops.max():.2f} m")


def fly_simulator(rocket: RocketConfig, sim_config: SimConfig, count: int) -> str:
    sim = Simulation(rocket, sim_config, seed=SIM_SEED)
    results, stops, failures = [], [], {}
    for seed in range(SIM_SEED, SIM_SEED + count):
        sim.reset(seed=seed)
        sim.run()
        touchdown = sim.flight.touchdown
        results.append(touchdown)
        if sim.burn_summary.stop_height is not None:
            stops.append(sim.burn_summary.stop_height)
        for kind in (touchdown.failures if touchdown is not None else ("no touchdown",)):
            failures[kind] = failures.get(kind, 0) + 1
    landed = sum(1 for td in results if td is not None and td.success)
    speeds = [td.vertical_speed for td in results if td is not None]
    return (f"landed {landed} of {count}; touchdown median {statistics.median(speeds):.2f} max "
            f"{max(speeds):.2f} m/s; stop height {min(stops):.2f} to {max(stops):.2f} m; failures {failures or 'none'}")


def main() -> None:
    args = parse_args()
    rocket = load_rocket_config(args.rocket)
    sim_config = load_sim_config(args.sim)
    environment = sim_config.environment
    area = cm2_to_m2(args.drag_area_cm2) if args.drag_area_cm2 is not None else None
    if area is None and rocket.drag_device is not None and args.terminal_speed is None:
        area = rocket.drag_device.drag_area
    sizing = size_descent(rocket, environment, args.terminal_speed, area, args.drag_area_error)
    trigger_config = rocket.computer.landing_trigger
    if args.motor is not None:
        motors = tuple(replace(m, spec=load_motor(args.motor)) if m.role == "landing" else m for m in rocket.motors)
        rocket = replace(rocket, motors=motors)
    designed = design_rocket(rocket, sizing, None)
    brake = designed.computer.brake
    burn_fraction = brake.burn_fraction if brake is not None else NO_BRAKE_AREA
    if args.motor is None:
        design = size_burn(
            rocket, sizing, environment.gravity, args.hard_n, args.tail_ratio, args.tail_s,
            sizing.terminal_speed + args.design_margin_mps, trigger_config.target_speed, burn_fraction,
        )
        designed = design_rocket(rocket, sizing, design)
    else:
        design = design_from_motor(designed.motor("landing").spec, designed.descent_mass, sizing.terminal_speed)
    control_dt = 1.0 / designed.control.control_rate_hz
    dt = sim_config.simulation.dt
    latency = (designed.control.action_delay_steps + HALF_STEP) * control_dt
    trigger = LandingTrigger(designed, trigger_config, environment, decision_latency=latency)
    apogee = ascent_apogee(designed, environment)
    igniter_mean = designed.motor("landing").ignition_delay_mean
    nominal_draw = nominal_draws(igniter_mean, np.array([1.0]))
    nominal = simulate_landings(designed, trigger, apogee, nominal_draw, control_dt, dt)
    print(f"Landing design for {designed.name} ({Path(args.rocket).name})")
    deploy_speed = brake.deploy_descent_speed if brake is not None else None
    profile = descent_profile(sizing, environment.gravity, apogee, designed.drag_device, deploy_speed)
    print_descent(sizing, designed, apogee, profile, float(nominal.command_height[0]))
    print_burn(design, environment.gravity, sizing, args.motor is None)
    print_trigger(trigger, nominal, design)
    singles = single_error_draws(igniter_mean, designed.descent_mass, args.igniter_spread_s,
                                 ErrorBudget().speed_noise)
    checked = simulate_landings(designed, trigger, apogee, singles, control_dt, dt)
    print_sensitivities(design, sizing, nominal, checked, trigger, args, designed)
    print_window(designed, trigger, apogee, control_dt, dt, args)
    budget = ErrorBudget(args.igniter_spread_s, args.thrust_error, g_to_kg(args.mass_error_g), args.drag_area_error)
    print(f"6. Expected landing rate, {args.draws} one dimensional flights with igniter "
          f"+-{budget.igniter_spread * MS_PER_S:.0f} ms, thrust +-{PERCENT * budget.thrust_error:.0f} %, mass "
          f"+-{args.mass_error_g:.0f} g, brake area +-{PERCENT * budget.drag_area_error:.0f} %, estimate noise "
          f"{budget.height_noise:g} m and {budget.speed_noise:g} m/s")
    full = monte_carlo(designed, trigger, apogee, budget, control_dt, dt, args)
    print("   " + describe_outcome(full))
    if args.sim_flights > 0:
        print(f"   simulator, {args.sim_flights} flights each with the training YAML's hidden errors, "
              f"seeds from {SIM_SEED}:")
        print("   calm:  " + fly_simulator(designed, sim_config, args.sim_flights))
        print("   windy: " + fly_simulator(designed, load_sim_config(args.wind), args.sim_flights))
    print("7. Measure these first (landings won per 1000 flights when that one error is removed from the budget above)")
    saved = {}
    for name in ERROR_SOURCES:
        outcome = monte_carlo(designed, trigger, apogee, budget.without(name), control_dt, dt, args)
        saved[name] = PER_THOUSAND * (outcome.landed.mean() - full.landed.mean())
    for rank, name in enumerate(sorted(saved, key=saved.get, reverse=True)[:TOP_MEASUREMENTS], start=1):
        print(f"   {rank}. {ERROR_LABELS[name]}: about {saved[name]:.0f} landings in 1000")


if __name__ == "__main__":
    main()
