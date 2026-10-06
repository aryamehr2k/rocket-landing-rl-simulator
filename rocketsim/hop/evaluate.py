"""Flying the electric vehicle's mission with the PID or a trained model, one flight or many.

The same seed gives the same sensor noise, gusts and hidden errors, so the PID and a model compare flight by flight.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rocketsim.curriculum import scale_range
from rocketsim.guidance_config import PID
from rocketsim.hop.env import flight_info, vehicle_for_training
from rocketsim.hop.mission import MissionConfig, load_mission_config
from rocketsim.hop.policy import HopPolicyController
from rocketsim.hop.simulation import HopErrors, HopSimulation
from rocketsim.hop.training import HiddenErrorRanges, HopTrainingConfig, load_hop_training_config
from rocketsim.hop.vehicle import HopVehicleConfig, load_vehicle_config
from rocketsim.observation import ObservationBuilder
from rocketsim.policy import POLICY_FILE, load_policy
from rocketsim.simconfig import SimConfig, WindConfig

CONFIG_DIR = "configs"
TRAINING_DIR = "training"
ERROR_SEED_OFFSET = 7919  # the hidden errors of seed s come from their own stream, independent of the sensors


@dataclass
class HopModel:
    name: str
    folder: Path
    training: HopTrainingConfig
    controller: HopPolicyController


def training_file(folder: Path) -> Path:
    """The training YAML copy stored with a model or run folder."""
    files = sorted((folder / CONFIG_DIR / TRAINING_DIR).glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"{folder}: no {CONFIG_DIR}/{TRAINING_DIR}/*.yaml copy; not a model or run folder")
    return files[0]


def load_hop_model(folder: str | Path) -> HopModel:
    """A trained model with the observation fields and action settings it was trained with."""
    folder = Path(folder)
    training = load_hop_training_config(training_file(folder))
    mission = load_mission_config(training.mission_path)
    policy = load_policy(folder / POLICY_FILE)
    observer = ObservationBuilder(training.observation, mission.planned_duration)
    return HopModel(folder.name, folder, training, HopPolicyController(policy, observer, training.action))


def draw_errors(ranges: HiddenErrorRanges, seed: int, factor: float = 1.0) -> HopErrors:
    """The hidden vehicle errors of one flight, the same for every controller flying that seed."""
    rng = np.random.default_rng(seed + ERROR_SEED_OFFSET)
    return HopErrors(
        thrust_scale=float(rng.uniform(*scale_range(ranges.thrust_scale, factor))),
        dry_mass_offset=float(rng.uniform(*scale_range(ranges.dry_mass_offset, factor))),
    )


def with_mission_wind(sim: SimConfig, speed: float | None, direction: float | None, gust: float | None) -> SimConfig:
    wind: WindConfig = sim.wind
    if speed is not None:
        wind = replace(wind, steady=speed)
    if direction is not None:
        wind = replace(wind, steady_direction=direction)
    if gust is not None:
        wind = replace(wind, gust_std=gust)
    return replace(sim, wind=wind)


def fly(
    vehicle: HopVehicleConfig, mission: MissionConfig, sim: SimConfig, seed: int, model: HopModel | None = None,
    errors: HopErrors | None = None, log_path: str | Path | None = None,
) -> tuple[HopSimulation, dict[str, Any]]:
    """One mission flight with the PID (model None) or a trained model. Returns the finished simulation and its summary."""
    if model is None:
        vehicle = replace(vehicle, controllers=replace(vehicle.controllers, steering=PID, throttle=PID))
    else:
        vehicle = vehicle_for_training(vehicle, model.training)
        model.controller.reset()
    simulation = HopSimulation(vehicle, mission, sim, seed=seed, log_path=log_path)
    simulation.reset(seed, errors=errors)
    while not simulation.done:
        action = model.controller.action(simulation.computer, simulation.t) if model is not None else None
        simulation.control_step(action)
    simulation.close()
    return simulation, flight_info(simulation)


def summarize(infos: list[dict[str, Any]]) -> dict[str, float]:
    """Rates and means over many flights."""
    touched = [i for i in infos if np.isfinite(i["vertical_speed"])]
    return {
        "flights": len(infos),
        "success": sum(i["success"] for i in infos),
        "landed": sum(i["landed"] for i in infos),
        "aborted": sum(i["aborted"] for i in infos),
        "max_height": float(np.mean([i["max_height"] for i in infos])),
        "hover_held": float(np.mean([i["hover_held"] for i in infos])),
        "miss": float(np.mean([i["miss"] for i in infos])),
        "touchdown_speed": float(np.mean([i["vertical_speed"] for i in touched])) if touched else float("nan"),
    }


def summary_table(rows: dict[str, dict[str, float]]) -> str:
    lines = [f"{'controller':<14}{'mission ok':>12}{'landed':>9}{'aborted':>9}{'max h':>8}{'hover':>8}{'miss':>8}{'v down':>8}"]
    for name, s in rows.items():
        n = int(s["flights"])
        lines.append(
            f"{name:<14}{int(s['success']):>8}/{n:<3}{int(s['landed']):>6}/{n:<2}{int(s['aborted']):>7}  "
            f"{s['max_height']:>6.1f}{s['hover_held']:>8.1f}{s['miss']:>8.1f}{s['touchdown_speed']:>8.2f}"
        )
    lines.append("max h in m, hover = longest hold inside the altitude band in s, miss = distance from the pad in m")
    return "\n".join(lines)


def load_world(vehicle_path: Path, mission_path: Path) -> tuple[HopVehicleConfig, MissionConfig]:
    return load_vehicle_config(vehicle_path), load_mission_config(mission_path)
