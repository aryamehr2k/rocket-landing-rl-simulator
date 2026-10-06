"""Fly a trained policy and the PID on the same seeds, with the training YAML's errors and wind, and compare.

Usage: python scripts/evaluate.py runs/20260926_120000_ppo --episodes 50 --sim configs/training/windy.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from rocketsim.config import RocketConfig, load_rocket_config
from rocketsim.env import episode_info, rocket_for_training
from rocketsim.phases import Phase
from rocketsim.units import rad_to_deg
from rocketsim.observation import ObservationBuilder
from rocketsim.policy import POLICY_FILE, PolicyController, load_policy
from rocketsim.simconfig import SimConfig, load_sim_config
from rocketsim.simulation import Simulation
from rocketsim.training_config import TrainingConfig, load_training_config
from rocketsim.yaml_section import ConfigError

CONFIG_DIR = "configs"
TRAINING_DIR = "training"
DEFAULT_EPISODES = 20
DEFAULT_SEED = 100


def run_configs(run: Path) -> tuple[Path, Path]:
    """The rocket and training YAML copies stored with a run."""
    configs = run / CONFIG_DIR
    if not configs.is_dir():
        raise FileNotFoundError(f"{run}: not a run folder (no {CONFIG_DIR}/ inside)")
    training = next((p for p in configs.glob(f"{TRAINING_DIR}/*.yaml")), None)
    rocket = next((p for p in configs.glob("*/*.yaml") if p.parent.name != TRAINING_DIR and "airframe" in p.read_text()), None)
    if training is None or rocket is None:
        raise FileNotFoundError(f"{configs}: the rocket or training YAML copy is missing")
    return rocket, training


def load_controller(run: Path, rocket: RocketConfig, training: TrainingConfig) -> PolicyController:
    policy = load_policy(run / POLICY_FILE)
    observer = ObservationBuilder(training.observation, rocket.motor("landing").spec.burn_time)
    return PolicyController(policy, observer, rocket.gimbal.max_angle, training.action.ignite_threshold)


def fly(
    rocket: RocketConfig, sim: SimConfig, seed: int, controller: PolicyController | None, log_path: Path | None
) -> dict[str, Any]:
    """One flight with the policy (or the PID when controller is None); returns the episode summary."""
    simulation = Simulation(rocket, sim, seed=seed, log_path=log_path)
    if controller is not None:
        controller.reset()
    moved, burn_steps = 0.0, 0
    last = (0.0, 0.0)
    while not simulation.done:
        action = controller.action(simulation.flight_computer, simulation.t) if controller is not None else None
        simulation.control_step(action)
        applied = simulation.applied
        if simulation.flight_computer.phase == Phase.LANDING_BURN:
            moved += abs(applied.gimbal_pitch - last[0]) + abs(applied.gimbal_yaw - last[1])
            burn_steps += 1
        last = (applied.gimbal_pitch, applied.gimbal_yaw)
    simulation.close()
    info = episode_info(simulation)
    # How much the two gimbal commands moved per control step during the burn, in degrees.
    info["gimbal_activity"] = rad_to_deg(moved / burn_steps) if burn_steps else 0.0
    return info


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    landed = [r for r in results if r["landed"]]
    touched = [r for r in results if "vertical_speed" in r]
    return {
        "landed_rate": len(landed) / len(results),
        "vertical_speed": float(np.mean([r["vertical_speed"] for r in touched])) if touched else float("nan"),
        "lateral_speed": float(np.mean([r["lateral_speed"] for r in touched])) if touched else float("nan"),
        "tilt_deg": float(np.mean([r["tilt_deg"] for r in touched])) if touched else float("nan"),
        "miss": float(np.mean([r["miss"] for r in touched])) if touched else float("nan"),
        "gimbal_activity": float(np.mean([r["gimbal_activity"] for r in results])),
        "timeouts": sum(1 for r in results if r["timeout"]),
    }


def print_table(rows: dict[str, dict[str, float]], episodes: int) -> None:
    print(
        f"{'controller':<12}{'landed':>10}{'v down':>9}{'v side':>9}{'tilt':>8}{'miss':>8}"
        f"{'gimbal move':>13}{'timeouts':>10}"
    )
    for name, s in rows.items():
        landed = f"{round(s['landed_rate'] * episodes)}/{episodes}"
        print(
            f"{name:<12}{landed:>10}{s['vertical_speed']:>9.2f}{s['lateral_speed']:>9.2f}"
            f"{s['tilt_deg']:>8.1f}{s['miss']:>8.1f}{s['gimbal_activity']:>13.2f}{s['timeouts']:>10}"
        )
    print("gimbal move: degrees the two gimbal commands change per control step during the burn, averaged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run folder written by scripts/train.py")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="first seed; both controllers fly seeds seed..seed+episodes-1")
    parser.add_argument("--sim", help="training YAML for the weather and hidden errors; default: the run's own copy")
    parser.add_argument("--write-logs", help="folder to write one flight log per policy flight into")
    parser.add_argument("--no-pid", action="store_true", help="skip the PID baseline")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    try:
        run = Path(args.run)
        rocket_path, training_path = run_configs(run)
        training = load_training_config(training_path)
        rocket = rocket_for_training(load_rocket_config(rocket_path), training)
        sim = load_sim_config(args.sim if args.sim else training_path)
        controller = load_controller(run, rocket, training)
    except (ConfigError, FileNotFoundError) as error:
        raise SystemExit(f"cannot load the run: {error}") from error
    logs = Path(args.write_logs) if args.write_logs else None
    seeds = range(args.seed, args.seed + args.episodes)
    policy_results = [fly(rocket, sim, seed, controller, logs / f"policy_seed{seed}.csv" if logs else None) for seed in seeds]
    rows = {"policy": summarize(policy_results)}
    if not args.no_pid:
        rows["pid"] = summarize([fly(rocket, sim, seed, None, None) for seed in seeds])
    print_table(rows, args.episodes)
    failed = [(seed, r) for seed, r in zip(seeds, policy_results) if not r["landed"]]
    for seed, r in failed:
        reason = "time limit" if r["timeout"] else ", ".join(r["failures"])
        print(f"policy seed {seed}: {reason}; v down {r.get('vertical_speed', float('nan')):.2f} m/s, miss {r.get('miss', float('nan')):.1f} m")


if __name__ == "__main__":
    main()
