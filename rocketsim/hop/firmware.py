"""C headers that carry one trained model and its vehicle and mission to firmware/hop/hop_control.c,
and a replay check that flies a mission in Python and runs the same estimates through the C code.

    policy_config.h  which inputs the network expects, in order, with their scales; output scaling
    hop_params.h     the mission plan, PID gains, safety limits and launch time as C initialisers
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from rocketsim.export import WEIGHTS_HEADER, c_float
from rocketsim.guidance_config import POLICY
from rocketsim.hop.evaluate import HopModel, draw_errors
from rocketsim.hop.mission import MissionConfig
from rocketsim.hop.simulation import HopSimulation
from rocketsim.hop.training import HopTrainingConfig
from rocketsim.hop.vehicle import RESIDUAL, HopVehicleConfig
from rocketsim.simconfig import SimConfig

POLICY_CONFIG_HEADER = "policy_config.h"
PARAMS_HEADER = "hop_params.h"
FIRMWARE = Path(__file__).resolve().parent.parent.parent / "firmware"
SUPPORTED_FIELDS = (
    "height_error", "vertical_speed_error", "reference_speed", "height", "vertical_speed", "lateral_position",
    "lateral_speed", "tilt", "tilt_rate", "roll", "gimbal", "throttle", "phase_ascent", "phase_hover", "phase_descent",
    "phase_landing",
)
REPLAY_TOLERANCE = 1e-3  # rad, throttle fraction and N m; C runs in float32, Python in float64


def policy_config_header(training: HopTrainingConfig) -> str:
    unsupported = [f.name for f in training.observation if f.name not in SUPPORTED_FIELDS]
    if unsupported:
        raise ValueError(f"the C controller has no input named {unsupported}; add it to firmware/hop/hop_fields.h first")
    residual = training.controllers is not None and training.controllers.mode == RESIDUAL
    ids = ", ".join(f"HOP_FIELD_{f.name.upper()}" for f in training.observation)
    scales = ", ".join(c_float(f.scale) for f in training.observation)
    return "\n".join([
        "/* Generated with the model by rocketsim/hop/firmware.py. Do not edit; export the model again instead. */",
        "#ifndef POLICY_CONFIG_H", "#define POLICY_CONFIG_H", "", '#include "hop_fields.h"', f'#include "{WEIGHTS_HEADER}"', "",
        f"#define POLICY_RESIDUAL {1 if residual else 0}",
        f"#define POLICY_THROTTLE_RANGE {c_float(training.action.throttle_range)}",
        f"#define POLICY_RESIDUAL_GIMBAL {c_float(training.action.residual_gimbal)}",
        f"static const int POLICY_FIELD_IDS[POLICY_INPUT_SIZE] = {{{ids}}};",
        f"static const float POLICY_FIELD_SCALES[POLICY_INPUT_SIZE] = {{{scales}}};", "", "#endif", "",
    ])


def params_header(vehicle: HopVehicleConfig, mission: MissionConfig, launch_time: float, gravity: float) -> str:
    m, a, alt, roll, safety = mission, vehicle.attitude, vehicle.altitude, vehicle.roll, vehicle.safety
    control_dt = 1.0 / vehicle.body.control.control_rate_hz
    mission_fields = {
        "target_altitude": m.target_altitude, "hover_time": m.hover_time, "hover_margin": m.hover_margin,
        "climb_speed": m.climb_speed, "climb_acceleration": m.climb_acceleration, "descent_speed": m.descent_speed,
        "final_height": m.final_height, "final_speed": m.final_speed,
    }
    vehicle_fields = {
        "control_dt": control_dt, "hover_throttle": vehicle.mass * gravity / vehicle.max_thrust,
        "max_gimbal": vehicle.body.gimbal.max_angle, "max_gimbal_rate": safety.max_gimbal_rate,
        "max_throttle_rate": safety.max_throttle_rate, "abort_tilt": safety.abort_tilt,
        "geofence_radius": safety.geofence_radius, "gravity": gravity,
        "att_kp": a.attitude.kp, "att_ki": a.attitude.ki, "att_kd": a.attitude.kd, "att_max_integral": a.attitude.max_integral,
        "pos_kp": a.position_kp, "pos_kd": a.position_kd, "max_tilt_command": a.max_tilt_command,
        "alt_height_gain": alt.height_gain, "alt_max_speed_correction": alt.max_speed_correction,
        "alt_speed_gain": alt.speed_gain, "alt_integral_gain": alt.speed_integral_gain,
        "alt_max_integral_accel": alt.max_integral_accel, "roll_kp": roll.kp, "roll_kd": roll.kd,
    }

    def initialiser(values: dict[str, float]) -> str:
        return "{" + ", ".join(f".{k} = {c_float(v)}" for k, v in values.items())

    flags = f", .steering_policy = {int(vehicle.controllers.steering == POLICY)}, .throttle_policy = {int(vehicle.controllers.throttle == POLICY)}}}"
    return "\n".join([
        f"/* Generated from {Path(vehicle.source).name} and {Path(mission.source).name}. All angles in radians. */",
        "#ifndef HOP_PARAMS_H", "#define HOP_PARAMS_H", "",
        f"#define HOP_LAUNCH_TIME {c_float(launch_time)}",
        f"#define HOP_MISSION_INIT {initialiser(mission_fields)}}}",
        f"#define HOP_VEHICLE_INIT {initialiser(vehicle_fields)}{flags}", "", "#endif", "",
    ])


def write_headers(folder: Path, model: HopModel, vehicle: HopVehicleConfig, mission: MissionConfig, sim: SimConfig) -> None:
    (folder / POLICY_CONFIG_HEADER).write_text(policy_config_header(model.training))
    (folder / PARAMS_HEADER).write_text(params_header(vehicle, mission, sim.simulation.pad_hold_time, sim.environment.gravity))


def record_flight(model: HopModel, vehicle: HopVehicleConfig, mission: MissionConfig, sim: SimConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fly one mission in Python; return the estimate rows the C harness reads and the commands Python produced."""
    simulation = HopSimulation(vehicle, mission, sim, seed=seed)
    simulation.reset(seed, errors=draw_errors(model.training.hidden_errors, seed))
    model.controller.reset()
    inputs, commands = [], []
    computer = simulation.computer
    while not simulation.done:
        est = computer.estimate
        tilt, rate = est.tilt, est.tilt_rate
        inputs.append([simulation.t, computer.height, float(est.velocity[2]), float(est.position[0]), float(est.position[1]),
                       float(est.velocity[0]), float(est.velocity[1]), tilt[0], tilt[1], rate[0], rate[1],
                       est.total_tilt, est.roll, float(est.angular_rate[2])])
        command = simulation.control_step(model.controller.action(computer, simulation.t))
        commands.append([command.gimbal_pitch, command.gimbal_yaw, command.throttle, command.roll_torque, float(command.arm)])
    return np.array(inputs), np.array(commands)


def replay_in_c(model_folder: Path, header_folder: Path, inputs: np.ndarray) -> np.ndarray:
    """Compile the C controller with the model's headers and run the recorded estimates through it."""
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        raise RuntimeError("no C compiler (gcc or cc) on PATH")
    with tempfile.TemporaryDirectory() as folder:
        build = Path(folder)
        for part in ("policy", "hop", "tests"):
            shutil.copytree(FIRMWARE / part, build / part)
        for header in (model_folder / WEIGHTS_HEADER, header_folder / POLICY_CONFIG_HEADER, header_folder / PARAMS_HEADER):
            target = build / ("policy" if header.name == WEIGHTS_HEADER else "hop") / header.name
            shutil.copy(header, target)
        binary = build / "hop_replay"
        sources = [build / "tests" / "hop_replay.c", build / "hop" / "hop_control.c", build / "hop" / "hop_mission.c", build / "policy" / "policy.c"]
        subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-I", str(build / "hop"), "-I", str(build / "policy"),
                        "-o", str(binary), *map(str, sources), "-lm"], check=True)
        text = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in inputs) + "\n"
        result = subprocess.run([str(binary)], input=text, capture_output=True, text=True, check=True)
    return np.array([[float(v) for v in line.split()[:5]] for line in result.stdout.strip().splitlines()])


def replay_check(model_folder: Path, model: HopModel, vehicle: HopVehicleConfig, mission: MissionConfig, sim: SimConfig,
                 seed: int) -> dict[str, float | bool | int]:
    """Fly one mission in Python and replay its estimates through the C controller; compare every command."""
    inputs, python = record_flight(model, vehicle, mission, sim, seed)
    c = replay_in_c(model_folder, model_folder, inputs)
    difference = float(np.abs(python - c).max()) if len(c) == len(python) else float("inf")
    return {"steps": len(python), "max_difference": difference, "tolerance": REPLAY_TOLERANCE,
            "passed": bool(difference <= REPLAY_TOLERANCE)}
