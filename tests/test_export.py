import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from rocketsim.policy import MlpPolicy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import export_policy  # noqa: E402


def make_policy(size: int = 5) -> MlpPolicy:
    rng = np.random.default_rng(3)
    return MlpPolicy(
        weights=(rng.normal(size=(6, size)).astype(np.float32), rng.normal(size=(2, 6)).astype(np.float32)),
        biases=(np.zeros(6, dtype=np.float32), np.array([0.5, -0.5], dtype=np.float32)),
        activation="tanh",
        obs_mean=np.arange(size, dtype=np.float32),
        obs_std=np.full(size, 2.0, dtype=np.float32),
        obs_clip=10.0,
        field_names=tuple(f"f{i}" for i in range(size)),
    )


def test_c_literals_are_valid() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        export_policy.c_float(float("inf"))
    assert export_policy.c_float(10.0) == "10.0f"
    assert export_policy.c_float(0.0) == "0.0f"
    assert export_policy.c_float(1e-7) == "1e-07f"
    assert export_policy.c_float(-0.25) == "-0.25f"


@pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="needs a C compiler")
def test_c_forward_matches_python(tmp_path: Path) -> None:
    policy = make_policy()
    policy.save(tmp_path / "policy.npz")
    worst_c, worst_onnx = export_policy.export(tmp_path, tmp_path / "out", samples=200, seed=1)
    assert worst_c <= export_policy.TOLERANCE and worst_onnx <= export_policy.TOLERANCE
    header = (tmp_path / "out" / export_policy.WEIGHTS_HEADER).read_text()
    assert "POLICY_INPUT_SIZE 5" in header and "/*   0: f0 */" in header
    assert (tmp_path / "out" / export_policy.ONNX_FILE).exists()
