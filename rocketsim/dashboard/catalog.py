"""What the forms can offer: vehicle, mission, world and training files, and the trained models."""

from pathlib import Path
from typing import Any

from rocketsim.dashboard.flightsetup import (
    DEFAULT_MISSION, DEFAULT_SEED, DEFAULT_VEHICLE, DEFAULT_WORLD, NUMBERS, OVERRIDES, vehicle_geometry,
)
from rocketsim.dashboard.live import AS_FAST_AS_POSSIBLE, SPEEDS
from rocketsim.dashboard.paths import ProjectPaths
from rocketsim.dashboard.training import SUMMARY_FILE, TRAINING_COPIES, read_json, run_task
from rocketsim.hop.mission import load_mission_config
from rocketsim.hop.training import HOP_TASK, training_task
from rocketsim.hop.vehicle import is_vehicle_file, load_vehicle_config
from rocketsim.policy import POLICY_FILE
from rocketsim.simconfig import load_sim_config
from rocketsim.units import rad_to_deg

MODEL_INFO_FILE = "model.json"
VEHICLES, MISSIONS = "vehicles", "missions"


def _yaml_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.yaml")) if folder.is_dir() else []


def _task(path: Path) -> str | None:
    try:
        return training_task(path)
    except (ValueError, OSError):
        return None


def vehicles(paths: ProjectPaths) -> list[dict[str, Any]]:
    """Electric vehicle files in configs/vehicles, with the sizes to draw them."""
    files = []
    for path in _yaml_files(paths.configs / VEHICLES):
        try:
            if is_vehicle_file(path):
                geometry = vehicle_geometry(load_vehicle_config(path))
                files.append({"path": paths.client_path(path), "name": path.stem, "geometry": geometry})
        except (ValueError, OSError):
            continue
    return files


def missions(paths: ProjectPaths) -> list[dict[str, Any]]:
    """Mission files with the values the form's overrides start from."""
    files = []
    for path in _yaml_files(paths.configs / MISSIONS):
        try:
            mission = load_mission_config(path)
        except (ValueError, OSError) as error:
            files.append({"path": paths.client_path(path), "name": path.stem, "error": str(error)})
            continue
        files.append({
            "path": paths.client_path(path), "name": mission.name,
            "values": {key: getattr(mission, key) for key in OVERRIDES},
            "altitude_tolerance": mission.altitude_tolerance,
        })
    return files


def trainings(paths: ProjectPaths) -> list[dict[str, Any]]:
    """Every training file with its task; the hop ones double as world files for flights."""
    files = []
    for path in _yaml_files(paths.training_files):
        entry: dict[str, Any] = {"path": paths.client_path(path), "name": path.stem, "task": _task(path)}
        if entry["task"] == HOP_TASK:
            try:
                wind = load_sim_config(path).wind
                entry["wind"] = {
                    "speed": wind.steady, "direction_deg": rad_to_deg(wind.steady_direction), "gust_std": wind.gust_std,
                }
            except (ValueError, OSError) as error:
                entry["error"] = str(error)
        files.append(entry)
    return files


def models(paths: ProjectPaths) -> list[dict[str, Any]]:
    """Exported models (models/<name>/) and hop runs (runs/<run>/) that have a policy.npz, newest first."""
    found = []
    candidates = [(folder, "model") for folder in _folders(paths.models)]
    candidates += [(folder, "run") for folder in _folders(paths.runs)]
    for folder, kind in candidates:
        if not (folder / POLICY_FILE).exists() or run_task(folder) != HOP_TASK:
            continue
        copies = sorted((folder / TRAINING_COPIES).glob("*.yaml"))
        found.append({
            "path": paths.client_path(folder), "name": folder.name, "kind": kind,
            "training": copies[0].name if copies else None,
            "created": (folder / POLICY_FILE).stat().st_mtime,
            "info": read_json(folder / MODEL_INFO_FILE), "summary": read_json(folder / SUMMARY_FILE),
        })
    return sorted(found, key=lambda model: model["created"], reverse=True)


def _folders(root: Path) -> list[Path]:
    return [folder for folder in root.iterdir() if folder.is_dir()] if root.is_dir() else []


def options(paths: ProjectPaths, batch_processes: int, max_flights: int) -> dict[str, Any]:
    """Everything the Fly, Many flights and Training forms list, with their defaults and limits."""
    training_files = trainings(paths)
    return {
        "vehicles": vehicles(paths),
        "missions": missions(paths),
        "worlds": [entry for entry in training_files if entry["task"] == HOP_TASK],
        "trainings": training_files,
        "models": [{"path": m["path"], "name": m["name"], "kind": m["kind"]} for m in models(paths)],
        "defaults": {"vehicle": DEFAULT_VEHICLE, "mission": DEFAULT_MISSION, "world": DEFAULT_WORLD, "seed": DEFAULT_SEED},
        "numbers": {key: {"default": d, "min": low, "max": high} for key, (d, low, high) in NUMBERS.items()},
        "overrides": {key: {"min": low, "max": high} for key, (low, high) in OVERRIDES.items()},
        "speeds": list(SPEEDS) + [AS_FAST_AS_POSSIBLE],
        "max_flights": max_flights,
        "batch_processes": batch_processes,
    }
