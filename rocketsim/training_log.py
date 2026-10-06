"""What a training run reports: statistics of the recent flights, progress.csv and the console line."""

import csv
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from rocketsim.vecenv import PLANES_PER_FLIGHT

PROGRESS_FILE = "progress.csv"
FLIGHT_WINDOW = 100  # recent flights the rates and the mean return per plane are averaged over
COLUMNS = (
    "time/iterations", "time/total_timesteps", "time/fps", "time/time_elapsed",
    "flights/count", "flights/mission_success_rate", "flights/landed_rate", "flights/mean_miss",
    "flights/mean_touchdown_speed", "rollout/ep_rew_mean", "rollout/ep_len_mean",
    "curriculum/wind_max", "curriculum/hidden_errors",
    "train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl", "train/clip_fraction",
    "train/explained_variance", "train/std",
)
NO_WIND_LIMIT = -1.0  # logged as curriculum/wind_max when the level keeps the training file's wind


class FlightStats:
    """The outcome of the last FLIGHT_WINDOW finished flights and how many have finished in total."""

    def __init__(self, window: int = FLIGHT_WINDOW) -> None:
        self.recent: deque[dict[str, Any]] = deque(maxlen=window)
        self.count = 0

    def add(self, infos: list[dict[str, Any]]) -> None:
        """Record the flights that finished this step; their infos carry the flight summary."""
        # Both planes of a flight carry the same summary; count the flight once.
        for info in infos[::PLANES_PER_FLIGHT]:
            if "landed" in info:
                self.recent.append(info)
                self.count += 1

    def rate(self, key: str) -> float | None:
        """Mean of key over the recent flights that have a finite value (no touchdown speed without a touchdown)."""
        values = [float(r[key]) for r in self.recent if key in r and math.isfinite(r[key])]
        return float(np.mean(values)) if values else None

    def summary(self) -> dict[str, float | None]:
        return {
            "flights/count": self.count,
            "flights/landed_rate": self.rate("landed"),
            "flights/mission_success_rate": self.rate("success"),
            "flights/mean_touchdown_speed": self.rate("vertical_speed"),
            "flights/mean_miss": self.rate("miss"),
        }


class ProgressLog:
    """progress.csv with a fixed set of columns, one row per PPO update; empty where there is no value yet."""

    def __init__(self, path: Path) -> None:
        self.handle = path.open("w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=COLUMNS)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow({key: "" if row.get(key) is None else row[key] for key in COLUMNS})
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def console_line(row: dict[str, Any]) -> str:
    """For example: update 12  196,608 samples  820/s  reward -35.2  mission ok 0.45  landed 0.80  std 0.21"""
    parts = [f"update {row['time/iterations']}", f"{row['time/total_timesteps']:,} samples", f"{row['time/fps']}/s"]
    for label, key, form in (("reward", "rollout/ep_rew_mean", ".1f"), ("mission ok", "flights/mission_success_rate", ".2f"),
                             ("landed", "flights/landed_rate", ".2f"), ("std", "train/std", ".2f")):
        if row.get(key) is not None:
            parts.append(f"{label} {row[key]:{form}}")
    return "  ".join(parts)
