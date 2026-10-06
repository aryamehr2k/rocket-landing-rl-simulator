"""Training runs: the run folders under runs/, their progress logs, and trainings started from the page.

A run folder `<date>_<time>_<name>` holds progress.csv (rocketsim/training_log.py), and summary.json once finished.
"""

import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rocketsim.dashboard.paths import ProjectPaths
from rocketsim.hop.training import training_task
from rocketsim.policy import POLICY_FILE

PROGRESS_FILE = "progress.csv"
SUMMARY_FILE = "summary.json"
TRAINING_COPIES = Path("configs") / "training"
RUNNING_WINDOW_S = 120.0  # a run whose progress.csv changed this recently counts as running
STAMP = re.compile(r"^(?P<stamp>\d{8}_\d{6})_(?P<name>.+)$")
STAMP_FORMAT = "%Y%m%d_%H%M%S"
START_SLACK_S = 5.0  # a run folder may carry a stamp this much earlier than the process start we noted
NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_TIMESTEPS = 1_000_000_000
LOG_TAIL_BYTES = 24_000
LOG_TAIL_LINES = 80
TIMESTEPS = "time/total_timesteps"
# Browser name of each progress.csv column the charts use.
SERIES = {
    "timesteps": TIMESTEPS,
    "success": "flights/mission_success_rate",
    "landed": "flights/landed_rate",
    "reward": "rollout/ep_rew_mean",
    "miss": "flights/mean_miss",
    "touchdown_speed": "flights/mean_touchdown_speed",
    "wind": "curriculum/wind_max",
    "fps": "time/fps",
}
FILLED = ("wind",)  # older runs logged the wind only on the rows where it changed; carry it forward


class TrainingError(ValueError):
    """A training that cannot be started or stopped."""


@dataclass
class TrainingProcess:
    name: str
    training: str
    log: Path
    process: subprocess.Popen
    started: float

    @property
    def alive(self) -> bool:
        return self.process.poll() is None


def read_progress(path: Path) -> dict[str, list[float | None]]:
    """The charted columns of a progress.csv as lists, None where a row has no value."""
    series: dict[str, list[float | None]] = {key: [] for key in SERIES}
    if not path.exists():
        return series
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if _number(row.get(TIMESTEPS)) is None:
                continue
            for key, column in SERIES.items():
                value = _number(row.get(column))
                if value is None and key in FILLED and series[key]:
                    value = series[key][-1]
                series[key].append(value)
    return series


def _number(text: str | None) -> float | None:
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def run_name(folder: Path) -> str:
    """The name part of a run folder, without its date and time."""
    match = STAMP.match(folder.name)
    return match.group("name") if match else folder.name


def run_started(folder: Path) -> float | None:
    """When the run began, from the date and time in its folder name (seconds since the epoch)."""
    match = STAMP.match(folder.name)
    return datetime.strptime(match.group("stamp"), STAMP_FORMAT).timestamp() if match else None


def is_run(folder: Path) -> bool:
    return folder.is_dir() and ((folder / PROGRESS_FILE).exists() or (folder / TRAINING_COPIES).is_dir())


def run_task(folder: Path) -> str | None:
    """The `task` of the run's training file copy: "hop", "solid_landing", or None without a copy."""
    copies = sorted((folder / TRAINING_COPIES).glob("*.yaml"))
    try:
        return training_task(copies[0]) if copies else None
    except (ValueError, OSError):
        return None


def last_value(values: list[float | None]) -> float | None:
    return next((value for value in reversed(values) if value is not None), None)


class TrainingManager:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.processes: dict[str, TrainingProcess] = {}
        self.lock = threading.Lock()  # two quick clicks on Start must not launch the same name twice

    def runs(self) -> list[dict[str, Any]]:
        """Every run folder, newest first, with its state and latest numbers."""
        if not self.paths.runs.is_dir():
            return []
        folders = [folder for folder in self.paths.runs.iterdir() if is_run(folder)]
        return [self.run_info(folder) for folder in sorted(folders, key=lambda f: f.name, reverse=True)]

    def run_info(self, folder: Path) -> dict[str, Any]:
        progress = folder / PROGRESS_FILE
        series = read_progress(progress)
        summary = read_json(folder / SUMMARY_FILE)
        managed = self._managed(folder)
        if summary is not None:
            state = "finished"
        elif managed is not None:
            state = "running" if managed.alive else "stopped"
        elif progress.exists() and time.time() - progress.stat().st_mtime < RUNNING_WINDOW_S:
            state = "running"
        else:
            state = "stopped"
        return {
            "path": self.paths.client_path(folder), "folder": folder.name, "name": run_name(folder),
            "task": run_task(folder), "state": state, "summary": summary, "has_policy": (folder / POLICY_FILE).exists(),
            "timesteps": last_value(series["timesteps"]), "fps": last_value(series["fps"]),
            "success": last_value(series["success"]), "landed": last_value(series["landed"]),
            "started": run_started(folder), "updated": progress.stat().st_mtime if progress.exists() else None,
            "stoppable": managed is not None and managed.alive,
        }

    def run_detail(self, folder: Path) -> dict[str, Any]:
        """A run with its full charted series and the end of its console log."""
        log = self._log_path(folder)
        return {
            "run": self.run_info(folder), "series": read_progress(folder / PROGRESS_FILE),
            "log": _tail(log) if log is not None else "", "log_path": str(log) if log is not None else None,
        }

    def start(self, training: Path, name: str, timesteps: int | None) -> dict[str, Any]:
        """Run scripts/train.py in the background, its console output into runs/train_logs/<name>.log.

        It gets its own process group, so the Stop button ends it with its simulation workers, and only those.
        """
        if not NAME.match(name):
            raise TrainingError("the run name may use letters, digits, - and _ (at most 40)")
        if timesteps is not None and not 1 <= timesteps <= MAX_TIMESTEPS:
            raise TrainingError(f"timesteps must be from 1 to {MAX_TIMESTEPS}")
        if not training.is_relative_to(self.paths.training_files) or training.suffix != ".yaml":
            raise TrainingError("the training file must be a YAML in configs/training/")
        command = [
            sys.executable, str(self.paths.scripts / "train.py"), "--training", str(training), "--name", name,
            "--runs", str(self.paths.runs),
        ]
        if timesteps is not None:
            command += ["--timesteps", str(timesteps)]
        log = self.paths.train_logs / f"{name}.log"
        with self.lock:
            current = self.processes.get(name)
            if current is not None and current.alive:
                raise TrainingError(f"a training named {name!r} is already running")
            if log.exists():
                raise TrainingError(f"{log.name} already exists in {self.paths.client_path(log.parent)}/; pick another name")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as output:
                process = subprocess.Popen(
                    command, cwd=self.paths.root, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            self.processes[name] = TrainingProcess(name, training.name, log, process, time.time())
        return {"name": name, "pid": process.pid, "log": self.paths.client_path(log), "command": command[1:]}

    def stop(self, name: str) -> bool:
        """End a training this dashboard started. Returns False if there is none by that name."""
        managed = self.processes.get(name)
        if managed is None or not managed.alive:
            return False
        try:
            os.killpg(managed.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    def started_here(self) -> list[dict[str, Any]]:
        return [
            {"name": p.name, "training": p.training, "alive": p.alive, "returncode": p.process.poll(),
             "log": self.paths.client_path(p.log), "started": p.started}
            for p in self.processes.values()
        ]

    def _managed(self, folder: Path) -> TrainingProcess | None:
        managed = self.processes.get(run_name(folder))
        started = run_started(folder)
        if managed is None or started is None or started < managed.started - START_SLACK_S:
            return None
        return managed

    def _log_path(self, folder: Path) -> Path | None:
        managed = self._managed(folder)
        if managed is not None:
            return managed.log
        log = self.paths.train_logs / f"{run_name(folder)}.log"
        return log if log.exists() else None


def read_json(path: Path) -> dict[str, Any] | None:
    """The JSON object in a file, or None when it is missing or unreadable."""
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (ValueError, OSError):
        return None


def _tail(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - LOG_TAIL_BYTES))
        text = handle.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-LOG_TAIL_LINES:])
