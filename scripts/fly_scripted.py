"""Fly the rocket open loop: ascent motor at t = 0, gimbal fixed, no controller.

The landing motor can be lit at a fixed time with --landing-ignite-at. Stage 2 adds
the PID controller. Writes a flight log CSV and, with --plot, a PNG next to it.

Usage: python scripts/fly_scripted.py --rocket configs/rockets/example_tvc.yaml \
           --sim configs/training/default.yaml --out runs/vertical.csv --plot
"""

import argparse
from pathlib import Path

from plot_flight import DPI, plot_flight_log

from rocketsim.aero import Wind
from rocketsim.config import load_rocket_config
from rocketsim.flightlog import FlightLogWriter, read_flight_log
from rocketsim.physics import IPOS, IQ, IVEL, IW, Flight, RocketDynamics
from rocketsim.simconfig import load_sim_config
from rocketsim.units import deg_to_rad, rad_to_deg

PHASE_NAME = "OPEN_LOOP"
ASCENT, LANDING = 0, 1
STATE_NAMES = ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")
QUATERNION_NAMES = ("qw", "qx", "qy", "qz")
RATE_NAMES = ("wx_radps", "wy_radps", "wz_radps")


def fly_open_loop(args: argparse.Namespace) -> Flight:
    rocket = load_rocket_config(args.rocket)
    sim = load_sim_config(args.sim)
    dynamics = RocketDynamics(rocket, sim.environment)
    flight = Flight(dynamics, sim.simulation.dt)
    wind = Wind(sim.wind)
    flight.inputs.gimbal_pitch = deg_to_rad(args.gimbal_pitch_deg)
    flight.inputs.gimbal_yaw = deg_to_rad(args.gimbal_yaw_deg)
    flight.inputs.ignite(ASCENT, rocket.motors[ASCENT].ignition_delay_mean)
    if args.landing_ignite_at is not None:
        flight.inputs.ignite(LANDING, args.landing_ignite_at + rocket.motors[LANDING].ignition_delay_mean)
    with FlightLogWriter(args.out) as log:
        while not flight.done and flight.t < sim.simulation.max_flight_time:
            flight.inputs.wind = wind.step(sim.simulation.dt)
            log.write(log_row(flight, dynamics))
            flight.step()
        log.write(log_row(flight, dynamics))
    return flight


def log_row(flight: Flight, dynamics: RocketDynamics) -> dict[str, float | str]:
    y, t, inputs = flight.y, flight.t, flight.inputs
    row: dict[str, float | str] = {"time_s": t, "phase": PHASE_NAME}
    for name, value in zip(STATE_NAMES, list(y[IPOS]) + list(y[IVEL])):
        row[f"true_{name}"] = value
    for name, value in zip(QUATERNION_NAMES + RATE_NAMES, list(y[IQ]) + list(y[IW])):
        row[f"true_{name}"] = value
    row.update(
        {
            "true_mass_kg": dynamics.mass_properties(y).mass,
            "gimbal_pitch_cmd_rad": inputs.gimbal_pitch,
            "gimbal_yaw_cmd_rad": inputs.gimbal_yaw,
            "gimbal_pitch_act_rad": inputs.gimbal_pitch,
            "gimbal_yaw_act_rad": inputs.gimbal_yaw,
            "ascent_ignite_cmd": float(inputs.motors[ASCENT].ignition_time is not None),
            "ascent_thrust_n": dynamics.motor_thrust(ASCENT, t, y, inputs),
            "landing_ignite_cmd": float(inputs.motors[LANDING].ignition_time is not None),
            "landing_thrust_n": dynamics.motor_thrust(LANDING, t, y, inputs),
            "wind_x_mps": inputs.wind[0],
            "wind_y_mps": inputs.wind[1],
        }
    )
    return row


def report(flight: Flight) -> None:
    print(f"flight time {flight.t:.2f} s, apogee {flight.apogee:.1f} m")
    if flight.touchdown is None:
        print("no touchdown before the time limit")
        return
    td = flight.touchdown
    print(
        f"touchdown: vertical {td.vertical_speed:.2f} m/s, lateral {td.lateral_speed:.2f} m/s, "
        f"tilt {rad_to_deg(td.tilt):.1f} deg (limit {rad_to_deg(td.tilt_limit):.1f} deg), "
        f"miss {td.miss_distance:.2f} m"
    )
    print("landed" if td.success else f"crashed: {', '.join(td.failures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rocket", default="configs/rockets/example_tvc.yaml")
    parser.add_argument("--sim", default="configs/training/default.yaml")
    parser.add_argument("--out", default="runs/open_loop.csv", help="flight log CSV to write")
    parser.add_argument("--gimbal-pitch-deg", type=float, default=0.0, help="fixed pitch gimbal angle")
    parser.add_argument("--gimbal-yaw-deg", type=float, default=0.0, help="fixed yaw gimbal angle")
    parser.add_argument("--landing-ignite-at", type=float, help="send the landing igniter command at this time")
    parser.add_argument("--plot", action="store_true", help="also write a PNG next to the CSV")
    args = parser.parse_args()
    flight = fly_open_loop(args)
    report(flight)
    print(f"wrote {args.out}")
    if args.plot:
        png = Path(args.out).with_suffix(".png")
        plot_flight_log(read_flight_log(args.out), Path(args.out).stem).savefig(png, dpi=DPI)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
