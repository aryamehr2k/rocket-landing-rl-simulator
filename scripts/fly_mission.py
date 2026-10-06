"""Fly the electric vehicle's mission (launch, climb, hover, descent, landing) with the PID or a trained model.

Usage: python scripts/fly_mission.py --model models/hop_latest --flights 50 --compare --wind-mps 4 --animate
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from rocketsim.flightlog import read_flight_log
from rocketsim.hop.evaluate import (
    HopModel, draw_errors, fly, load_hop_model, load_world, summarize, summary_table, with_mission_wind,
)
from rocketsim.hop.plots import batch_figure, mission_figure
from rocketsim.hop.training import HiddenErrorRanges, load_hop_training_config
from rocketsim.motors import MotorFileError
from rocketsim.simconfig import load_sim_config
from rocketsim.units import deg_to_rad
from rocketsim.yaml_section import ConfigError

DEFAULT_VEHICLE = "configs/vehicles/electric_hopper.yaml"
DEFAULT_MISSION = "configs/missions/hop_50m.yaml"
DEFAULT_WORLD = "configs/training/hop.yaml"
DEFAULT_OUT = "runs/flights"
DPI = 120
ANIMATION_SPEED = "3"


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vehicle", default=DEFAULT_VEHICLE)
    parser.add_argument("--mission", default=DEFAULT_MISSION)
    parser.add_argument("--world", default=DEFAULT_WORLD, help="training YAML that gives the simulation settings and the hidden error ranges")
    parser.add_argument("--model", help="model folder (models/<name>) for the AI; without it the PID flies")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--flights", type=int, default=1)
    parser.add_argument("--compare", action="store_true", help="also fly the PID on the same seeds (with --model)")
    parser.add_argument("--wind-mps", type=float, help="steady wind speed")
    parser.add_argument("--wind-direction-deg", type=float, help="direction the wind blows toward")
    parser.add_argument("--gust-mps", type=float, help="gust standard deviation")
    parser.add_argument("--hidden-errors", action="store_true", help="draw thrust and mass errors from the world file's ranges")
    parser.add_argument("--out", default=DEFAULT_OUT, help="folder for logs, plots and animations")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="write plots (on by default)")
    parser.add_argument("--animate", action="store_true", help="write a 3D GIF of the (first) flight")
    args = parser.parse_args()
    if args.flights < 1:
        parser.error("--flights must be at least 1")
    return args


def main() -> None:
    args = parse()
    try:
        vehicle, mission = load_world(Path(args.vehicle), Path(args.mission))
        sim = load_sim_config(args.world)
        direction = deg_to_rad(args.wind_direction_deg) if args.wind_direction_deg is not None else None
        sim = with_mission_wind(sim, args.wind_mps, direction, args.gust_mps)
        ranges = load_hop_training_config(args.world).hidden_errors if args.hidden_errors else HiddenErrorRanges()
        model = load_hop_model(args.model) if args.model else None
    except (ConfigError, MotorFileError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    controllers: list[tuple[str, HopModel | None]] = [(model.name if model else "pid", model)]
    if model is not None and args.compare:
        controllers.append(("pid", None))
    rows = {}
    for name, flyer in controllers:
        infos, logs = [], []
        for seed in range(args.seed, args.seed + args.flights):
            log_path = out / f"{name}_seed{seed}.csv"
            _, info = fly(vehicle, mission, sim, seed, flyer, draw_errors(ranges, seed), log_path)
            infos.append(info)
            logs.append(log_path)
            if args.flights == 1 or not info["success"]:
                print(f"{name} seed {seed}: {describe(info)}")
        rows[name] = summarize(infos)
        if args.plot:
            if args.flights == 1:
                figure = mission_figure(read_flight_log(logs[0]), mission, f"{name}, seed {args.seed}: {describe(infos[0])}")
                figure.savefig(logs[0].with_suffix(".png"), dpi=DPI)
                print(f"wrote {logs[0].with_suffix('.png')}")
            else:
                figure = batch_figure([read_flight_log(p) for p in logs], [i["success"] for i in infos], mission, name)
                figure.savefig(out / f"{name}_flights.png", dpi=DPI)
                print(f"wrote {out / f'{name}_flights.png'}")
        if args.animate:
            gif = logs[0].with_suffix(".gif")
            subprocess.run([sys.executable, str(Path(__file__).parent / "animate_flight.py"), str(logs[0]), "--rocket",
                            args.vehicle, "--out", str(gif), "--follow", "--speed", ANIMATION_SPEED], check=True)
    if args.flights > 1 or len(rows) > 1:
        print(summary_table(rows))
    print(f"logs in {out}")


def describe(info: dict[str, Any]) -> str:
    verdict = "MISSION OK" if info["success"] else "mission failed"
    if info["aborted"]:
        verdict += " (aborted)"
    return (
        f"{verdict}; max height {info['max_height']:.1f} m, hover held {info['hover_held']:.1f} s, "
        f"{'landed' if info['landed'] else 'hard landing'} at {info['vertical_speed']:.2f} m/s, {info['miss']:.1f} m from the pad"
    )


if __name__ == "__main__":
    main()
