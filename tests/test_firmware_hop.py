import shutil
from pathlib import Path

import numpy as np
import pytest

from rocketsim.hop.training import load_hop_training_config
from rocketsim.modelpack import package_model
from rocketsim.policy import MlpPolicy
from tests.conftest import REPO_ROOT

TRAINING = REPO_ROOT / "configs" / "training" / "hop.yaml"
HIDDEN = 8


def make_run(folder: Path) -> None:
    """A run folder with a small random network and copies of the configs it would be trained with."""
    training = load_hop_training_config(TRAINING)
    for source in (TRAINING, training.vehicle_path, training.mission_path,
                   REPO_ROOT / "configs" / "motors" / "example_edf_90mm.yaml",
                   REPO_ROOT / "configs" / "sensors" / "example_imu_gps.yaml"):
        target = folder / "configs" / source.parent.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    size = len(training.observation)
    rng = np.random.default_rng(0)
    MlpPolicy(
        weights=(rng.normal(0.0, 0.3, (HIDDEN, size)).astype(np.float32), rng.normal(0.0, 0.3, (2, HIDDEN)).astype(np.float32)),
        biases=(np.zeros(HIDDEN, dtype=np.float32), np.zeros(2, dtype=np.float32)),
        activation="tanh", obs_mean=np.zeros(size, dtype=np.float32), obs_std=np.ones(size, dtype=np.float32),
        obs_clip=10.0, field_names=tuple(f.name for f in training.observation),
    ).save(folder / "policy.npz")


@pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="needs a C compiler")
def test_model_folder_and_c_controller_match_python(tmp_path: Path) -> None:
    run = tmp_path / "run"
    make_run(run)
    description = package_model(run, tmp_path / "models" / "test_model")
    assert description["checks"]["passed"]
    replay = description["firmware"]["replay_check"]
    assert replay["passed"], replay
    assert replay["steps"] > 50
    folder = tmp_path / "models" / "test_model"
    for name in ("policy.npz", "policy.onnx", "policy_weights.h", "policy_config.h", "hop_params.h", "model.json"):
        assert (folder / name).exists(), name
    assert [i["name"] for i in description["inputs"]][0] == "height_error"
    assert description["outputs"][1]["name"] == "throttle_vote"
