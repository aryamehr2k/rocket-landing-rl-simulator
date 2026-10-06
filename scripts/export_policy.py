"""Turn a training run into a model folder in models/, ready to fly in the simulator or put on the board.

Usage: python scripts/export_policy.py runs/<run> [--name hop_v1] [--evaluate] [--install]
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from rocketsim.export import (  # noqa: F401  (also used by the tests)
    FIRMWARE_POLICY, ONNX_FILE, TOLERANCE, WEIGHTS_HEADER, c_float, check_c_forward, check_onnx, weights_header, write_onnx,
)
from rocketsim.hop.firmware import FIRMWARE, PARAMS_HEADER, POLICY_CONFIG_HEADER
from rocketsim.hop.training import HOP_TASK
from rocketsim.modelpack import MODEL_FILE, MODELS_DIR, package_model
from rocketsim.policy import POLICY_FILE, load_policy
from rocketsim.yaml_section import ConfigError

EVALUATION_LOGS = "evaluation_logs"
FIRMWARE_HOP = FIRMWARE / "hop"
INDENT = 2


def export(run: Path, out: Path, samples: int, seed: int) -> tuple[float, float]:
    """Write only the header and the ONNX file; return the largest C and ONNX differences from Python."""
    policy = load_policy(run / POLICY_FILE)
    out.mkdir(parents=True, exist_ok=True)
    (out / WEIGHTS_HEADER).write_text(weights_header(policy))
    write_onnx(policy, out / ONNX_FILE)
    return check_c_forward(policy, out, samples, seed), check_onnx(policy, out / ONNX_FILE, samples, seed)


def publish(run: Path, name: str, evaluate: bool, install: bool, models: Path = MODELS_DIR) -> Path:
    """Package a run as models/<name>, optionally evaluate it and install it into the firmware.

    The C and ONNX versions are checked against Python first. The evaluation (electric vehicle runs only) flies the
    model and the PID on the same seeds into models/<name>/evaluation/.
    """
    out = models / name
    description = package_model(run, out)
    if evaluate and description["task"] == HOP_TASK:
        from rocketsim.hop.report import evaluate_model

        description["evaluation"] = evaluate_model(out, run / EVALUATION_LOGS)
        (out / MODEL_FILE).write_text(json.dumps(description, indent=INDENT))
    if install:
        shutil.copy(out / WEIGHTS_HEADER, FIRMWARE_POLICY / WEIGHTS_HEADER)
        for header in (POLICY_CONFIG_HEADER, PARAMS_HEADER):
            if (out / header).exists():
                shutil.copy(out / header, FIRMWARE_HOP / header)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run folder written by scripts/train.py")
    parser.add_argument("--name", help="model folder name in models/ (default: the run folder name)")
    parser.add_argument("--evaluate", action="store_true", help="fly the model and the PID and store the comparison")
    parser.add_argument("--install", action="store_true", help="copy the C header into firmware/policy/")
    args = parser.parse_args()
    run = Path(args.run)
    try:
        out = publish(run, args.name or run.name, args.evaluate, args.install)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, ConfigError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    description = json.loads((out / MODEL_FILE).read_text())
    checks = description["checks"]
    print(f"wrote {out}")
    print(f"C and ONNX against Python on {checks['samples']} inputs: largest differences "
          f"{checks['c_max_difference']:.1e} and {checks['onnx_max_difference']:.1e} ({'match' if checks['passed'] else 'MISMATCH'})")
    if "evaluation" in description:
        print(description["evaluation"]["conditions"])
        print(description["evaluation"]["table"])
    if "firmware" in description:
        replay = description["firmware"]["replay_check"]
        print(f"C controller replay of a {replay['steps']} step flight: largest difference {replay['max_difference']:.1e} "
              f"({'match' if replay['passed'] else 'MISMATCH'})")
    if args.install:
        print(f"installed the model headers into {FIRMWARE_POLICY} and {FIRMWARE_HOP}")
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
