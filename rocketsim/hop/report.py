"""Evaluation of a trained model against the PID after training, written into the model folder.

The flight logs stay in the run folder (they are large); the model folder gets the summary, plots and animation.
"""

import os
import subprocess
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from rocketsim.flightlog import read_flight_log
from rocketsim.hop.evaluate import draw_errors, fly, load_hop_model, summarize, summary_table, training_file, with_mission_wind
from rocketsim.hop.mission import load_mission_config
from rocketsim.hop.plots import batch_figure, mission_figure
from rocketsim.hop.training import load_hop_training_config
from rocketsim.hop.vehicle import load_vehicle_config
from rocketsim.simconfig import load_sim_config

FIRST_SEED = 100
SPARE_CORES = 4
DPI = 110
ANIMATE = Path(__file__).resolve().parent.parent.parent / "scripts" / "animate_flight.py"
ANIMATION_SPEED = "3"


def _fly_one(job: tuple[str, str | None, int, str]) -> dict[str, Any]:
    model_dir, controller, seed, log_path = job
    model = load_hop_model(model_dir)
    training = model.training
    vehicle, mission = load_vehicle_config(training.vehicle_path), load_mission_config(training.mission_path)
    evaluation = training.evaluation
    sim = with_mission_wind(load_sim_config(training_file(Path(model_dir))), evaluation.wind, None, evaluation.gust_std)
    _, info = fly(vehicle, mission, sim, seed, model if controller else None, draw_errors(training.hidden_errors, seed), log_path)
    return info


def evaluate_model(model_dir: Path, log_dir: Path) -> dict[str, Any]:
    """Fly the model and the PID on the same seeds; write plots and a table into model_dir/evaluation."""
    training = load_hop_training_config(training_file(model_dir))
    evaluation = training.evaluation
    mission = load_mission_config(training.mission_path)
    out = model_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(FIRST_SEED, FIRST_SEED + evaluation.flights)
    workers = max(1, min(evaluation.flights, (os.cpu_count() or 1) - SPARE_CORES))
    rows, logs = {}, {}
    with get_context("fork").Pool(workers) as pool:
        for name, controller in (("model", model_dir.name), ("pid", None)):
            paths = [str(log_dir / f"{name}_seed{seed}.csv") for seed in seeds]
            infos = pool.map(_fly_one, [(str(model_dir), controller, seed, path) for seed, path in zip(seeds, paths)])
            rows[name] = summarize(infos)
            logs[name] = (paths, infos)
    table = summary_table(rows)
    conditions = f"{evaluation.flights} flights, seeds {seeds[0]}-{seeds[-1]}, wind up to {evaluation.wind} m/s with {evaluation.gust_std} m/s gusts, hidden errors on"
    (out / "summary.txt").write_text(conditions + "\n" + table + "\n")
    if evaluation.plots:
        for name, (paths, infos) in logs.items():
            batch_figure([read_flight_log(p) for p in paths], [i["success"] for i in infos], mission, name).savefig(out / f"{name}_flights.png", dpi=DPI)
            mission_figure(read_flight_log(paths[0]), mission, f"{name}, seed {seeds[0]}").savefig(out / f"{name}_seed{seeds[0]}.png", dpi=DPI)
    if evaluation.animation:
        subprocess.run([sys.executable, str(ANIMATE), logs["model"][0][0], "--rocket", str(training.vehicle_path),
                        "--out", str(out / f"model_seed{seeds[0]}.gif"), "--follow", "--speed", ANIMATION_SPEED], check=True)
    return {"conditions": conditions, "model": rows["model"], "pid": rows["pid"], "table": table}
