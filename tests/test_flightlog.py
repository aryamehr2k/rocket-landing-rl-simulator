import math
from pathlib import Path

import numpy as np
import pytest

from rocketsim.flightlog import COLUMNS, FlightLogWriter, read_flight_log


def test_round_trip_with_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "flight.csv"
    with FlightLogWriter(path) as log:
        log.write({"time_s": 0.0, "phase": "PAD", "true_z_m": 0.5})
        log.write({"time_s": 0.005, "phase": "BOOST", "true_z_m": 0.6, "ascent_thrust_n": 30.0})
    assert len(COLUMNS) == 56
    data = read_flight_log(path)
    assert list(data) == list(COLUMNS)
    assert data["time_s"] == pytest.approx([0.0, 0.005])
    assert list(data["phase"]) == ["PAD", "BOOST"]
    assert math.isnan(data["ascent_thrust_n"][0]) and data["ascent_thrust_n"][1] == 30.0
    assert np.all(np.isnan(data["est_x_m"]))


def test_unknown_column_is_rejected(tmp_path: Path) -> None:
    with FlightLogWriter(tmp_path / "flight.csv") as log:
        with pytest.raises(KeyError):
            log.write({"time_s": 0.0, "bogus": 1.0})
