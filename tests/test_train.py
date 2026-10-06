import json
import sys
from pathlib import Path

import pytest
import yaml

from rocketsim.config import load_rocket_config
from rocketsim.simconfig import load_sim_config
from rocketsim.training_config import load_training_config
from rocketsim.training_log import COLUMNS, FlightStats
from tests.conftest import EXAMPLE_ROCKET, EXAMPLE_SIM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate  # noqa: E402
import train  # noqa: E402

TINY_PPO = {"total_timesteps": 256, "n_sims": 2, "workers": 1, "n_steps": 64, "batch_size": 64, "n_epochs": 1, "checkpoint_every": 128}


def test_train_and_evaluate_round_trip(tmp_path: Path) -> None:
    data = yaml.safe_load(EXAMPLE_SIM.read_text())
    data["ppo"].update(TINY_PPO)
    training_path = tmp_path / "tiny.yaml"
    training_path.write_text(yaml.safe_dump(data))
    folder = tmp_path / "run"
    folder.mkdir()
    train.train(EXAMPLE_ROCKET, training_path, folder, timesteps=None, models=tmp_path / "models")
    assert (tmp_path / "models" / "run" / "model.json").exists()
    for name in (train.MODEL_FILE, train.SUMMARY_FILE, train.CHECKPOINT_FILE, "policy.npz", "progress.csv"):
        assert (folder / name).exists(), name
    rocket_copy, training_copy = evaluate.run_configs(folder)
    rocket = load_rocket_config(rocket_copy)
    training = load_training_config(training_copy)
    assert training.ppo.n_sims == 2
    controller = evaluate.load_controller(folder, rocket, training)
    result = evaluate.fly(rocket, load_sim_config(training_copy), seed=7, controller=controller, log_path=None)
    assert "landed" in result and not result["timeout"]
    summary = evaluate.summarize([result])
    assert 0.0 <= summary["landed_rate"] <= 1.0
    summary_file = json.loads((folder / train.SUMMARY_FILE).read_text())
    assert summary_file["timesteps"] >= summary_file["requested_timesteps"] == TINY_PPO["total_timesteps"]
    progress = (folder / "progress.csv").read_text()
    assert progress.splitlines()[0].split(",") == list(COLUMNS) and len(progress.splitlines()) > 1


def test_flight_stats_count_each_flight_once_and_skip_missing_touchdowns() -> None:
    landed = {"landed": True, "success": True, "vertical_speed": 1.0, "miss": 1.0}
    hovering = {"landed": False, "success": False, "vertical_speed": float("nan"), "miss": 3.0}
    flights = FlightStats()
    flights.add([landed, landed, {}, {}])  # both planes of a finished flight, then one still flying
    flights.add([{}, {}, hovering, hovering])
    summary = flights.summary()
    assert summary["flights/count"] == 2
    assert summary["flights/landed_rate"] == pytest.approx(0.5)
    assert summary["flights/mean_miss"] == pytest.approx(2.0)
    assert summary["flights/mean_touchdown_speed"] == pytest.approx(1.0)
