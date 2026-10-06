"""Train a control policy with PPO on many simulated flights at once.

The training YAML's `task` picks the problem: `hop` flies the electric test vehicle through its
whole mission (vehicle and mission named in the YAML); without a task it is the solid rocket's
landing burn (rocket from --rocket). Every run gets a timestamped folder under runs/ with the
model, the policy as plain numpy weights, the observation normalisation, the training log and
exact copies of the YAML files. At the end the trained network is written to models/<run>/
(see scripts/export_policy.py) and, for the electric vehicle, flown against the PID.

Usage: python scripts/train.py --training configs/training/hop.yaml
       python scripts/train.py --rocket configs/rockets/example_tvc.yaml --training configs/training/default.yaml
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rocketsim.config import load_rocket_config
from rocketsim.curriculum import level_for
from rocketsim.env import PlaneEpisode
from rocketsim.hop.env import HopEpisode
from rocketsim.hop.mission import load_mission_config
from rocketsim.hop.training import HOP_TASK, HopTrainingConfig, load_hop_training_config, training_task
from rocketsim.hop.vehicle import load_vehicle_config
from rocketsim.policy import POLICY_FILE, policy_from_sb3
from rocketsim.simconfig import load_sim_config
from rocketsim.training_config import PpoConfig, TrainingConfig, load_training_config
from rocketsim.vecenv import PLANES_PER_FLIGHT, PlanePairVecEnv
from rocketsim.yaml_section import ConfigError
from export_policy import publish  # noqa: E402
from rocketsim.modelpack import MODELS_DIR

MODEL_FILE = "model.zip"
NORMALIZER_FILE = "vecnormalize.pkl"
SUMMARY_FILE = "summary.json"
CHECKPOINT_FILE = "checkpoint.json"
CONFIG_DIR = "configs"
TRAINING_DIR = "training"
OBS_CLIP = 10.0
EPISODE_WINDOW = 100  # recent flights the logged landing rate is averaged over
LOG_INTERVAL = 1  # PPO updates between log lines
SEED_STRIDE = 1000  # flights of one run get seeds seed * SEED_STRIDE + index
TORCH_THREADS = 1  # the networks are tiny; more threads only fight the simulation workers for the CPU


def run_folder(root: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = root / f"{stamp}_{name}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def copy_configs(folder: Path, rocket_path: Path, training_path: Path) -> None:
    """Exact copies of every YAML the run used. The rocket, motor and sensor files keep their folder
    names so the relative paths inside the rocket file still resolve; the training file goes to training/."""
    rocket = load_rocket_config(rocket_path)
    sources = [rocket_path, Path(rocket.computer.sensors.source)] + [Path(m.spec.source) for m in rocket.motors]
    targets = [folder / CONFIG_DIR / source.parent.name / source.name for source in sources]
    targets.append(folder / CONFIG_DIR / TRAINING_DIR / training_path.name)
    for source, target in zip(sources + [training_path], targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)


def copy_hop_configs(folder: Path, training_path: Path, training: HopTrainingConfig) -> None:
    """Copies of the training, vehicle, mission, motor and sensor files, keeping their folder names."""
    vehicle = load_vehicle_config(training.vehicle_path)
    sources = [training_path, training.vehicle_path, training.mission_path, Path(vehicle.sensors.source),
               Path(vehicle.motor.spec.source)]
    for source in sources:
        target = folder / CONFIG_DIR / source.parent.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)


def make_callback(training: TrainingConfig | HopTrainingConfig, total: int, folder: Path) -> Any:
    """Curriculum by progress, plus a landing rate over the last flights in the PPO log."""
    from stable_baselines3.common.callbacks import BaseCallback

    class TrainingCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.recent: list[dict[str, Any]] = []
            self.flights = 0
            self.level = None
            self.next_checkpoint = training.ppo.checkpoint_every

        def _on_step(self) -> bool:
            progress = self.num_timesteps / total
            level = level_for(training.curriculum, progress)
            if level != self.level:
                self.level = level
                flights = getattr(self.training_env, "venv", self.training_env)  # VecNormalize wraps the flights
                flights.set_level(level)
                self.logger.record("curriculum/wind_max", level.wind_max if level.wind_max is not None else -1.0)
                self.logger.record("curriculum/hidden_errors", level.hidden_errors)
            # Both planes of a flight carry the same summary; count the flight once.
            for info in self.locals.get("infos", ())[::PLANES_PER_FLIGHT]:
                if "landed" in info and "terminal_observation" in info:
                    self.recent.append(info)
                    self.flights += 1
            self.recent = self.recent[-EPISODE_WINDOW:]
            return True

        def _on_rollout_end(self) -> None:
            if self.num_timesteps >= self.next_checkpoint:
                self.next_checkpoint += training.ppo.checkpoint_every
                save_checkpoint(self.model, self.training_env, folder, self.num_timesteps, self.recent)
            if self.recent:
                landed = [r for r in self.recent if r["landed"]]
                self.logger.record("flights/landed_rate", len(landed) / len(self.recent))
                if "success" in self.recent[0]:
                    self.logger.record("flights/mission_success_rate", float(np.mean([r["success"] for r in self.recent])))
                self.logger.record("flights/count", self.flights)
                speeds = [r["vertical_speed"] for r in self.recent if "vertical_speed" in r]
                misses = [r["miss"] for r in self.recent if "miss" in r]
                if speeds:
                    self.logger.record("flights/mean_touchdown_speed", float(np.mean(speeds)))
                if misses:
                    self.logger.record("flights/mean_miss", float(np.mean(misses)))

    return TrainingCallback()


def save_checkpoint(model: Any, normalized: Any, folder: Path, timesteps: int, recent: list[dict[str, Any]]) -> None:
    """The network so far as policy.npz, so a model can be exported or flown while training continues."""
    flights = getattr(normalized, "venv", normalized)
    policy_from_sb3(model, normalized, flights.observation_names, model.policy.activation_fn.__name__.lower()).save(
        folder / POLICY_FILE
    )
    state = {"timesteps": timesteps, "recent_flights": len(recent)}
    if recent and "success" in recent[0]:
        state["recent_mission_success_rate"] = float(np.mean([r["success"] for r in recent]))
    (folder / CHECKPOINT_FILE).write_text(json.dumps(state, indent=2))


def build_model(venv: Any, ppo: PpoConfig, folder: Path) -> Any:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from torch import nn

    torch.set_num_threads(TORCH_THREADS)

    activation = nn.Tanh if ppo.activation == "tanh" else nn.ReLU
    model = PPO(
        "MlpPolicy", venv, n_steps=ppo.n_steps, batch_size=ppo.batch_size, n_epochs=ppo.n_epochs,
        learning_rate=ppo.learning_rate, gamma=ppo.gamma, gae_lambda=ppo.gae_lambda, clip_range=ppo.clip_range,
        ent_coef=ppo.ent_coef, seed=ppo.seed, verbose=1,
        policy_kwargs={
            "net_arch": {"pi": list(ppo.hidden_layers), "vf": list(ppo.hidden_layers)}, "activation_fn": activation,
            "log_std_init": ppo.log_std_init,
        },
    )
    model.set_logger(configure(str(folder), ["stdout", "csv"]))
    return model


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

    venv = PlanePairVecEnv(make_episode, ppo.n_sims, ppo.workers)
    from stable_baselines3.common.vec_env import VecNormalize

    normalized = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=OBS_CLIP, gamma=ppo.gamma)
    model = build_model(normalized, ppo, folder)
    callback = make_callback(training, total, folder)
    started = time.time()
    model.learn(total_timesteps=total, callback=callback, log_interval=LOG_INTERVAL)
    elapsed = time.time() - started
    model.save(folder / MODEL_FILE)
    normalized.save(str(folder / NORMALIZER_FILE))
    policy_from_sb3(model, normalized, venv.observation_names, ppo.activation).save(folder / POLICY_FILE)
    summary = {
        "timesteps": total, "flights": callback.flights, "seconds": round(elapsed, 1),
        "recent_landed_rate": float(np.mean([r["landed"] for r in callback.recent])) if callback.recent else None,
        "training": training_path.name,
    }
    (folder / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    normalized.close()
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
