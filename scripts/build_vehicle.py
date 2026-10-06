"""Turn a filled-in data sheet of the electric vehicle into the simulator's config files and check the vehicle.

Usage: python scripts/build_vehicle.py configs/datasheets/<name>.yaml [--name <name>] [--force]
"""

import argparse
from pathlib import Path

from rocketsim.hop.vehicle_check import check_lines, check_vehicle
from rocketsim.hop.vehicle_files import build_vehicle
from rocketsim.motors import MotorFileError
from rocketsim.yaml_section import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS = REPO_ROOT / "configs"
TRAINING_BASE = DEFAULT_CONFIGS / "training" / "hop.yaml"


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheet", help="the filled-in data sheet, configs/datasheets/<name>.yaml")
    parser.add_argument("--name", help="name of the files (default: vehicle.name in the sheet, else its file name)")
    parser.add_argument("--force", action="store_true", help="replace files that exist already")
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS, help="configs folder to write into")
    parser.add_argument("--training-base", type=Path, default=TRAINING_BASE,
                        help="training file the new one starts from")
    return parser.parse_args()


def shown(path: Path) -> str:
    """The path relative to the working folder when it is inside it."""
    path = path.resolve()
    return str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else str(path)


def main() -> None:
    args = parse()
    try:
        built = build_vehicle(args.sheet, args.configs, args.training_base, args.name, args.force)
    except (ConfigError, MotorFileError) as error:
        raise SystemExit(str(error)) from error
    paths = built.paths
    print(f"{built.name}: built from {shown(Path(args.sheet))}")
    for path in paths.all():
        print(f"  wrote {shown(path)}")
    print("check:")
    print("\n".join(check_lines(built, check_vehicle(built))))
    print("next:")
    print(f"  python scripts/fly_mission.py --vehicle {shown(paths.vehicle)} --mission {shown(paths.mission)} "
          f"--world {shown(paths.training)}")
    print(f"  python scripts/train.py --training {shown(paths.training)}")


if __name__ == "__main__":
    main()
