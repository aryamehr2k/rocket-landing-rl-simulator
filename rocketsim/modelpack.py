"""Model folders in models/<name>/: everything needed to fly or deploy one trained network.

policy.npz for the simulator, policy.onnx, policy_weights.h for the firmware, model.json and copies of the configs.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from rocketsim.export import ONNX_FILE, TOLERANCE, WEIGHTS_HEADER, check_c_forward, check_onnx, weights_header, write_onnx
from rocketsim.hop.mission import load_mission_config
from rocketsim.hop.training import HOP_TASK, load_hop_training_config, training_task
from rocketsim.hop.vehicle import load_vehicle_config
from rocketsim.observation import DESCRIPTIONS
from rocketsim.policy import POLICY_FILE, MlpPolicy, load_policy
from rocketsim.units import STANDARD_GRAVITY, rad_to_deg

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = "model.json"
CONFIG_DIR = "configs"
TRAINING_DIR = "training"
SUMMARY_FILE = "summary.json"
CHECK_SAMPLES = 1000
REPLAY_SEED = 3
INDENT = 2


def package_model(run: Path, out: Path, evaluation: dict[str, Any] | None = None, seed: int = 0) -> dict[str, Any]:
    """Write a model folder from a training run folder and return the model.json content.

    model.json says what every input and output means, with the scales, rates, export checks and results.
    """
    policy = load_policy(run / POLICY_FILE)
    out.mkdir(parents=True, exist_ok=True)
    if run.resolve() != out.resolve():
        shutil.copy(run / POLICY_FILE, out / POLICY_FILE)
        shutil.copytree(run / CONFIG_DIR, out / CONFIG_DIR, dirs_exist_ok=True)
    (out / WEIGHTS_HEADER).write_text(weights_header(policy))
    write_onnx(policy, out / ONNX_FILE)
    worst_c = check_c_forward(policy, out, CHECK_SAMPLES, seed)
    worst_onnx = check_onnx(policy, out / ONNX_FILE, CHECK_SAMPLES, seed)
    description = describe(policy, out, run.name)
    description["checks"] = {
        "samples": CHECK_SAMPLES, "tolerance": TOLERANCE,
        "c_max_difference": worst_c, "onnx_max_difference": worst_onnx,
        "passed": bool(worst_c <= TOLERANCE and worst_onnx <= TOLERANCE),
    }
    summary = run / SUMMARY_FILE
    if summary.exists():
        description["training"] = json.loads(summary.read_text())
    if evaluation is not None:
        description["evaluation"] = evaluation
    (out / MODEL_FILE).write_text(json.dumps(description, indent=INDENT))
    return description


def describe(policy: MlpPolicy, folder: Path, source_run: str) -> dict[str, Any]:
    """The input and output contract of the network, for whoever wires it into flight software."""
    training_path = sorted((folder / CONFIG_DIR / TRAINING_DIR).glob("*.yaml"))[0]
    sizes = [policy.weights[0].shape[1]] + [w.shape[0] for w in policy.weights]
    contract: dict[str, Any] = {
        "name": folder.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_run": source_run,
        "task": training_task(training_path),
        "network": {"layer_sizes": sizes, "activation": policy.activation, "parameters": int(
            sum(w.size + b.size for w, b in zip(policy.weights, policy.biases)))},
        "input_rule": "x[i] = clip((raw[i] / scale[i] - mean[i]) / std[i], -clip, +clip); the network runs once with the "
                      "pitch plane values (x, vx, tilt toward +x) and once with the yaw plane values (y, vy, tilt toward +y)",
        "input_clip": float(policy.obs_clip),
        "inputs": [],
        "outputs": [],
    }
    scales = _scales(training_path)
    for i, name in enumerate(policy.field_names):
        contract["inputs"].append({
            "index": i, "name": name, "scale": scales.get(name, 1.0), "mean": float(policy.obs_mean[i]),
            "std": float(policy.obs_std[i]), "meaning": DESCRIPTIONS.get(name, ""),
        })
    if contract["task"] == HOP_TASK:
        contract.update(_hop_contract(training_path))
        contract["firmware"] = _hop_firmware(folder)
    else:
        contract["outputs"] = [
            {"index": 0, "name": "gimbal", "range": [-1, 1], "meaning": "gimbal command for this plane = output x max gimbal angle"},
            {"index": 1, "name": "ignite_vote", "range": [-1, 1], "meaning": "landing igniter vote; mean of both planes above the threshold ignites"},
        ]
    return contract


def _scales(training_path: Path) -> dict[str, float]:
    import yaml

    data = yaml.safe_load(training_path.read_text())
    return {f["name"]: float(f.get("scale", 1.0)) for f in data.get("observation", {}).get("fields", [])}


def _hop_contract(training_path: Path) -> dict[str, Any]:
    training = load_hop_training_config(training_path)
    vehicle = load_vehicle_config(training.vehicle_path)
    mission = load_mission_config(training.mission_path)
    max_gimbal = rad_to_deg(vehicle.body.gimbal.max_angle)
    residual = training.controllers is not None and training.controllers.mode == "residual"
    controllers = training.controllers or vehicle.controllers
    return {
        "vehicle": vehicle.name,
        "mission": mission.name,
        "control_rate_hz": vehicle.body.control.control_rate_hz,
        "action_delay_steps": vehicle.body.control.action_delay_steps,
        "controllers": {"steering": controllers.steering, "throttle": controllers.throttle, "mode": controllers.mode},
        "outputs": [
            {"index": 0, "name": "gimbal", "range": [-1, 1], "meaning": (
                f"this plane's gimbal correction added to the PID's = output x {rad_to_deg(training.action.residual_gimbal):.1f} deg"
                if residual else f"this plane's gimbal command = output x {max_gimbal:.1f} deg; positive leans the nose toward +x (pitch run) or +y (yaw run)")},
            {"index": 1, "name": "throttle_vote", "range": [-1, 1], "meaning": (
                f"throttle correction added to the PID's = mean of the two planes' votes x {training.action.throttle_range}"
                if residual else f"throttle = hover throttle / cos(tilt) + mean of the two planes' votes x {training.action.throttle_range}, "
                                 "clipped to 0..1; hover throttle = mass x g / max thrust")},
        ],
        "hover_throttle_nominal": vehicle.mass * STANDARD_GRAVITY / vehicle.max_thrust,
        "max_gimbal_deg": max_gimbal,
    }


def _hop_firmware(folder: Path) -> dict[str, Any]:
    """Write the C headers for firmware/hop and check the C controller against Python on one flight."""
    from rocketsim.hop.env import vehicle_for_training
    from rocketsim.hop.evaluate import load_hop_model
    from rocketsim.hop.firmware import PARAMS_HEADER, POLICY_CONFIG_HEADER, replay_check, write_headers
    from rocketsim.simconfig import load_sim_config

    model = load_hop_model(folder)
    training_path = sorted((folder / CONFIG_DIR / TRAINING_DIR).glob("*.yaml"))[0]
    vehicle = vehicle_for_training(load_vehicle_config(model.training.vehicle_path), model.training)
    mission = load_mission_config(model.training.mission_path)
    sim = load_sim_config(training_path)
    write_headers(folder, model, vehicle, mission, sim)
    return {"headers": [WEIGHTS_HEADER, POLICY_CONFIG_HEADER, PARAMS_HEADER],
            "replay_check": replay_check(folder, model, vehicle, mission, sim, seed=REPLAY_SEED)}
