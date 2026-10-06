"""Train a control policy with PPO on many simulated flights at once; the run goes to runs/<date>_<time>_<name>/.

Usage: python scripts/train.py --training configs/training/hop.yaml
       python scripts/train.py --rocket configs/rockets/example_tvc.yaml --training configs/training/default.yaml
"""

import argparse
import json
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rocketsim.config import load_rocket_config
from rocketsim.curriculum import CurriculumLevel, level_for
from rocketsim.env import PlaneEpisode
from rocketsim.hop.env import HopEpisode
from rocketsim.hop.mission import load_mission_config
from rocketsim.hop.training import HOP_TASK, HopTrainingConfig, load_hop_training_config, training_task
from rocketsim.hop.vehicle import load_vehicle_config
from rocketsim.modelpack import MODELS_DIR
from rocketsim.policy import POLICY_FILE, policy_from_training
from rocketsim.ppo import Ppo
from rocketsim.simconfig import load_sim_config
from rocketsim.training_config import TrainingConfig, load_training_config
from rocketsim.training_log import FLIGHT_WINDOW, NO_WIND_LIMIT, PROGRESS_FILE, FlightStats, ProgressLog, console_line
from rocketsim.vecenv import PLANES_PER_FLIGHT, PlanePairVecEnv
from rocketsim.yaml_section import ConfigError

from export_policy import publish  # a sibling script; scripts/ is on sys.path when this runs

MODEL_FILE = "model.pt"
SUMMARY_FILE = "summary.json"
CHECKPOINT_FILE = "checkpoint.json"
CONFIG_DIR = "configs"
TRAINING_DIR = "training"
SEED_STRIDE = 1000  # flights of one run get seeds seed * SEED_STRIDE + index
TORCH_THREADS = 1  # the networks are tiny; more threads only fight the simulation workers for the CPU


def run_folder(root: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = root / f"{stamp}_{name}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def copy_run_configs(folder: Path, sources: list[Path], training_path: Path) -> None:
    """Exact copies of the YAML files a run used, under configs/.

    Each source keeps its folder name, so relative paths between them still resolve; the training
    file always goes to configs/training/, where the model packaging looks for it.
    """
    targets = [folder / CONFIG_DIR / source.parent.name / source.name for source in sources]
    targets.append(folder / CONFIG_DIR / TRAINING_DIR / training_path.name)
    for source, target in zip(sources + [training_path], targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)


def copy_configs(folder: Path, rocket_path: Path, training_path: Path) -> None:
    """The rocket, sensor, motor and training files of a solid rocket run."""
    rocket = load_rocket_config(rocket_path)
    sources = [rocket_path, Path(rocket.computer.sensors.source)] + [Path(m.spec.source) for m in rocket.motors]
    copy_run_configs(folder, sources, training_path)


def copy_hop_configs(folder: Path, training_path: Path, training: HopTrainingConfig) -> None:
    """The vehicle, mission, sensor, motor and training files of an electric vehicle run."""
    vehicle = load_vehicle_config(training.vehicle_path)
    sources = [training.vehicle_path, training.mission_path, Path(vehicle.sensors.source), Path(vehicle.motor.spec.source)]
    copy_run_configs(folder, sources, training_path)


def save_checkpoint(agent: Ppo, observation_names: tuple[str, ...], folder: Path, flights: FlightStats) -> None:
    """The network so far as policy.npz, so a model can be exported or flown while training continues."""
    policy_from_training(agent.model, agent.normalizer, observation_names).save(folder / POLICY_FILE)
    state: dict[str, Any] = {"timesteps": agent.timesteps, "recent_flights": len(flights.recent)}
    success = flights.rate("success")
    if success is not None:
        state["recent_mission_success_rate"] = success
    (folder / CHECKPOINT_FILE).write_text(json.dumps(state, indent=2))


def run_ppo(flights_env: PlanePairVecEnv, training: TrainingConfig | HopTrainingConfig, total: int,
            folder: Path) -> tuple[Ppo, FlightStats]:
    """PPO updates until total samples, with the curriculum by progress, checkpoints and one log row per update."""
    flights = FlightStats()
    curriculum: dict[str, float] = {}

    def apply_level(level: CurriculumLevel) -> None:
        flights_env.set_level(level)
        curriculum["curriculum/wind_max"] = level.wind_max if level.wind_max is not None else NO_WIND_LIMIT
        curriculum["curriculum/hidden_errors"] = level.hidden_errors

    level = level_for(training.curriculum, 0.0)
    apply_level(level)  # before the agent resets the flights, so the first ones already fly the first stage

    def on_step(timesteps: int, infos: list[dict[str, Any]]) -> None:
        nonlocal level
        new_level = level_for(training.curriculum, min(timesteps / total, 1.0))  # the last rollout may overshoot
        if new_level != level:
            level = new_level
            apply_level(level)
        flights.add(infos)

    ppo = replace(training.ppo, total_timesteps=total)  # model.pt then records what this run was asked for
    agent = Ppo(flights_env, ppo, episode_window=FLIGHT_WINDOW * PLANES_PER_FLIGHT)
    log = ProgressLog(folder / PROGRESS_FILE)
    started = time.time()
    next_checkpoint = training.ppo.checkpoint_every
    iteration = 0
    while agent.timesteps < total:
        rollout = agent.collect(on_step)
        losses = agent.update(rollout)
        iteration += 1
        if agent.timesteps >= next_checkpoint:
            next_checkpoint += training.ppo.checkpoint_every
            save_checkpoint(agent, flights_env.observation_names, folder, flights)
        elapsed = time.time() - started
        row: dict[str, Any] = {
            "time/iterations": iteration, "time/total_timesteps": agent.timesteps,
            "time/fps": int(agent.timesteps / elapsed), "time/time_elapsed": int(elapsed),
            "rollout/ep_rew_mean": float(np.mean(agent.finished_returns)) if agent.finished_returns else None,
            "rollout/ep_len_mean": float(np.mean(agent.finished_lengths)) if agent.finished_lengths else None,
            **flights.summary(), **curriculum, **{f"train/{key}": value for key, value in losses.items()},
        }
        log.write(row)
        print(console_line(row), flush=True)
    log.close()
    return agent, flights


def train(
    rocket_path: Path, training_path: Path, folder: Path, timesteps: int | None, models: Path = MODELS_DIR
) -> Path:
    sim = load_sim_config(training_path)
    training: TrainingConfig | HopTrainingConfig
    if training_task(training_path) == HOP_TASK:
        training = load_hop_training_config(training_path)
        vehicle = load_vehicle_config(training.vehicle_path)
        mission = load_mission_config(training.mission_path)
        copy_hop_configs(folder, training_path, training)
        hop: HopTrainingConfig = training

        def make_episode(index: int) -> PlaneEpisode | HopEpisode:
            return HopEpisode(vehicle, mission, sim, hop, seed=hop.ppo.seed * SEED_STRIDE + index)
    else:
        rocket = load_rocket_config(rocket_path)
        training = load_training_config(training_path)
        copy_configs(folder, rocket_path, training_path)
        solid: TrainingConfig = training

        def make_episode(index: int) -> PlaneEpisode | HopEpisode:
            return PlaneEpisode(rocket, sim, solid, seed=solid.ppo.seed * SEED_STRIDE + index)
    ppo = training.ppo
    total = timesteps if timesteps is not None else ppo.total_timesteps

    torch.set_num_threads(TORCH_THREADS)
    torch.backends.mkldnn.enabled = False  # oneDNN takes milliseconds on layers this small, plain BLAS microseconds
    flights_env = PlanePairVecEnv(make_episode, ppo.n_sims, ppo.workers)
    started = time.time()
    try:
        agent, flights = run_ppo(flights_env, training, total, folder)
    finally:
        flights_env.close()
    elapsed = time.time() - started
    agent.save(folder / MODEL_FILE)
    policy_from_training(agent.model, agent.normalizer, flights_env.observation_names).save(folder / POLICY_FILE)
    summary = {
        "timesteps": agent.timesteps, "requested_timesteps": total, "flights": flights.count,
        "seconds": round(elapsed, 1), "recent_landed_rate": flights.rate("landed"), "training": training_path.name,
    }
    (folder / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    model_folder = publish(folder, folder.name, evaluate=True, install=False, models=models)
    print(f"model written to {model_folder}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rocket", default="configs/rockets/example_tvc.yaml")
    parser.add_argument("--training", default="configs/training/default.yaml")
    parser.add_argument("--runs", default="runs", help="folder the run folder is created in")
    parser.add_argument("--name", default="ppo", help="suffix of the run folder name")
    parser.add_argument("--timesteps", type=int, help="override ppo.total_timesteps from the YAML")
    args = parser.parse_args()
    try:
        folder = run_folder(Path(args.runs), args.name)
        train(Path(args.rocket), Path(args.training), folder, args.timesteps)
    except (ConfigError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
    print(f"saved run to {folder}")


if __name__ == "__main__":
    main()
