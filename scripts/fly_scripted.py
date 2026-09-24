"""Fly the rocket closed loop with the onboard PID and landing trigger, or open loop.

Closed loop (default): the flight computer calibrates on the pad, launches, steers the boost,
lights the landing motor from its stopping distance table and steers the landing. Open loop
(--open-loop): fixed gimbal, no PID, landing motor at --landing-ignite-at if given. Either way
the flight log CSV has the sensor readings, the estimate and the truth. With --episodes the
same flight is repeated with different seeds and summarised. --landing-thrust-scale and
--dry-mass-offset-g and --brake-area-scale pin a hidden error the flight computer does not know,
for sweeps.

Usage: python scripts/fly_scripted.py --sim configs/training/windy.yaml --out runs/windy.csv --plot
"""

import argparse
from pathlib import Path

from plot_flight import DPI, plot_flight_log

from rocketsim.commands import PlaneAction
from rocketsim.config import load_rocket_config
from rocketsim.flightlog import read_flight_log
from rocketsim.simconfig import SimConfig, WindConfig, load_sim_config
from rocketsim.simulation import Simulation, with_fixed_errors, with_wind
from rocketsim.touchdown import TouchdownResult
from rocketsim.units import CM2_PER_M2, deg_to_rad, g_to_kg, kg_to_g, rad_to_deg

DEFAULT_ROCKET = "configs/rockets/example_tvc.yaml"
DEFAULT_SIM = "configs/training/default.yaml"
DEFAULT_OUT = "runs/scripted.csv"
DEFAULT_SEED = 0
PERCENT = 100.0


def fly(sim: Simulation, args: argparse.Namespace) -> None:
    """One flight. Open loop hands the flight computer a fixed action every control step."""
    if not args.open_loop:
        sim.run()
        return
    delta_x, delta_y = deg_to_rad(args.gimbal_pitch_deg), deg_to_rad(args.gimbal_yaw_deg)
    while not sim.done:
        ignite = args.landing_ignite_at is not None and sim.t >= args.landing_ignite_at
        sim.control_step(PlaneAction(delta_x, delta_y, ignite))


def describe(touchdown: TouchdownResult | None) -> str:
    if touchdown is None:
        return "no touchdown before the time limit"
    outcome = "landed" if touchdown.success else f"crashed: {', '.join(touchdown.failures)}"
    return (
        f"{outcome}; vertical {touchdown.vertical_speed:.2f} m/s, lateral {touchdown.lateral_speed:.2f} m/s, "
        f"tilt {rad_to_deg(touchdown.tilt):.1f} deg (limit {rad_to_deg(touchdown.tilt_limit):.1f}), "
        f"miss {touchdown.miss_distance:.2f} m"
    )


def describe_errors(sim: Simulation) -> str:
    draw = sim.draw
    return (
        f"hidden errors: landing thrust x{draw.landing_thrust_scale:.3f}, ascent thrust x{draw.ascent_thrust_scale:.3f}, "
        f"dry mass {kg_to_g(draw.dry_mass_offset):+.0f} g, brake area x{draw.device_drag_area_scale:.3f}"
    )


def report(sim: Simulation) -> None:
    flight, computer = sim.flight, sim.flight_computer
    print(f"flight time {flight.t:.2f} s, apogee {flight.apogee:.1f} m")
    print("phases: " + ", ".join(f"{phase.value} at {t:.2f} s" for t, phase in computer.phases.history))
    if computer.refusals:
        print("safety: " + "; ".join(computer.refusals))
    print(describe_errors(sim))
    trigger, rocket = computer.trigger, sim.rocket
    if trigger.calibrated:
        body_area = rocket.aero.drag_coefficient * rocket.reference_area
        fitted = trigger.drag_factor_device / trigger.drag_factor_body * body_area
        print(f"drag fitted in flight: brake Cd*A {fitted * CM2_PER_M2:.0f} cm2 "
              f"(file {rocket.drag_device.drag_area * CM2_PER_M2:.0f} cm2)")
    print(sim.burn_summary.describe())
    print(f"touchdown: {describe(flight.touchdown)}")


def load_world(args: argparse.Namespace) -> tuple[SimConfig, WindConfig]:
    """The training YAML with the command line's wind and pinned hidden errors applied."""
    sim_config = load_sim_config(args.sim)
    mass_offset = g_to_kg(args.dry_mass_offset_g) if args.dry_mass_offset_g is not None else None
    sim_config = with_fixed_errors(sim_config, args.landing_thrust_scale, mass_offset, args.brake_area_scale)
    direction = deg_to_rad(args.wind_direction_deg) if args.wind_direction_deg is not None else None
    return sim_config, with_wind(sim_config, args.wind_mps, direction, args.gust_mps)


def run_episodes(rocket_path: str, args: argparse.Namespace) -> None:
    """Repeat the flight over consecutive seeds and print one line per flight plus the totals."""
    rocket = load_rocket_config(rocket_path)
    sim_config, wind = load_world(args)
    sim = Simulation(rocket, sim_config, seed=args.seed, wind=wind)
    landed = 0
    for episode in range(args.episodes):
        seed = args.seed + episode
        sim.reset(seed=seed)
        fly(sim, args)
        touchdown = sim.flight.touchdown
        landed += bool(touchdown is not None and touchdown.success)
        stop = sim.burn_summary.stop_height
        stopped = f"stop {stop:.2f} m" if stop is not None else "no stop"
        print(f"seed {seed:4d}: {describe(touchdown)}; {stopped}")
    print(f"landed {landed} of {args.episodes} ({PERCENT * landed / args.episodes:.0f} %)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rocket", default=DEFAULT_ROCKET)
    parser.add_argument("--sim", default=DEFAULT_SIM, help="training YAML with the simulation, environment and wind")
    parser.add_argument("--out", default=DEFAULT_OUT, help="flight log CSV to write (single flight only)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for sensor errors, igniter delays and gusts")
    parser.add_argument("--episodes", type=int, default=1, help="repeat with seeds seed, seed+1, ... and summarise")
    parser.add_argument("--wind-mps", type=float, help="override the steady wind speed")
    parser.add_argument("--wind-direction-deg", type=float, help="override the direction the wind blows toward")
    parser.add_argument("--gust-mps", type=float, help="override the gust standard deviation")
    parser.add_argument("--landing-thrust-scale", type=float, help="pin the hidden landing motor strength, e.g. 1.05")
    parser.add_argument("--dry-mass-offset-g", type=float, help="pin the hidden dry mass error in grams, e.g. -36")
    parser.add_argument("--brake-area-scale", type=float, help="pin the hidden brake drag area error, e.g. 1.15")
    parser.add_argument("--open-loop", action="store_true", help="no PID and no trigger: fixed gimbal, optional timed landing burn")
    parser.add_argument("--gimbal-pitch-deg", type=float, default=0.0, help="open loop: fixed world-plane gimbal toward +x")
    parser.add_argument("--gimbal-yaw-deg", type=float, default=0.0, help="open loop: fixed world-plane gimbal toward +y")
    parser.add_argument("--landing-ignite-at", type=float, help="open loop: send the landing igniter command at this time")
    parser.add_argument("--plot", action="store_true", help="also write a PNG next to the CSV")
    args = parser.parse_args()
    if args.episodes > 1:
        run_episodes(args.rocket, args)
        return
    rocket = load_rocket_config(args.rocket)
    sim_config, wind = load_world(args)
    sim = Simulation(rocket, sim_config, seed=args.seed, log_path=args.out, wind=wind)
    fly(sim, args)
    sim.close()
    report(sim)
    print(f"wrote {args.out}")
    if args.plot:
        png = Path(args.out).with_suffix(".png")
        plot_flight_log(read_flight_log(args.out), Path(args.out).stem).savefig(png, dpi=DPI)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
