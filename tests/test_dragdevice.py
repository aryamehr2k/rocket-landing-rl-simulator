"""The drag device: its force, the sign of its moment by station, damping, and sizing helpers."""

import math
from dataclasses import replace

import numpy as np
import pytest

from rocketsim import quaternion as quat
from rocketsim.aero import air_density
from rocketsim.config import load_rocket_config
from rocketsim.dragdevice import (
    FULLY_OPEN, DragDeviceConfig, device_loads, drag_area_for_terminal_speed, drag_factor, plate_area, terminal_speed,
)
from rocketsim.physics import IVZ, IZ, RocketDynamics
from rocketsim.simconfig import load_sim_config
from tests.conftest import EXAMPLE_ROCKET, EXAMPLE_SIM, ROTATE_Y

NOSE_STATION, TAIL_STATION, CG = 0.15, 0.96, 0.494
NO_SPIN = np.zeros(3)


def example_device() -> DragDeviceConfig:
    device = load_rocket_config(EXAMPLE_ROCKET).drag_device
    assert device is not None
    return device


def test_example_device_loads_in_si() -> None:
    device = example_device()
    assert device.drag_area == pytest.approx(0.0566)
    assert device.station == pytest.approx(0.15)
    assert (device.deploy_time, device.retract_time) == (0.5, 0.3)


def test_open_device_drag_at_terminal_speed() -> None:
    env = load_sim_config(EXAMPLE_SIM).environment
    density = air_density(0.0, env)
    force, moment = device_loads(example_device(), FULLY_OPEN, density, np.array([0.0, 0.0, -20.0]), NO_SPIN, CG)
    # 0.5 * 1.1825 * 0.0566 * 400 = 13.4 N, pointing up the body axis against the fall.
    assert force == pytest.approx([0.0, 0.0, 0.5 * density * 0.0566 * 400.0], abs=1e-9)
    assert force[2] == pytest.approx(13.39, abs=0.05)
    assert moment == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert device_loads(example_device(), 0.5, density, np.array([0.0, 0.0, -20.0]), NO_SPIN, CG)[0][2] == pytest.approx(force[2] / 2)


def test_nose_device_restores_and_tail_device_diverges_tail_first() -> None:
    tilt = 0.2  # nose leaning toward +x while falling straight down
    airspeed_body = np.array([20.0 * math.sin(tilt), 0.0, -20.0 * math.cos(tilt)])
    nose = DragDeviceConfig(0.05, NOSE_STATION, 0.5, 0.3)
    tail = replace(nose, station=TAIL_STATION)
    _, nose_moment = device_loads(nose, FULLY_OPEN, 1.0, airspeed_body, NO_SPIN, CG)
    _, tail_moment = device_loads(tail, FULLY_OPEN, 1.0, airspeed_body, NO_SPIN, CG)
    assert nose_moment[1] < 0.0  # a negative moment about +b_y leans the nose back toward -x
    assert tail_moment[1] > 0.0
    assert nose_moment[0] == 0.0 and nose_moment[2] == 0.0


def test_pure_pitch_rate_is_damped() -> None:
    device = DragDeviceConfig(0.05, NOSE_STATION, 0.5, 0.3)
    _, moment = device_loads(device, FULLY_OPEN, 1.0, np.zeros(3), np.array([0.0, 1.0, 0.0]), CG)
    assert moment[1] < 0.0
    _, moment = device_loads(device, FULLY_OPEN, 1.0, np.zeros(3), np.array([0.0, -1.0, 0.0]), CG)
    assert moment[1] > 0.0


def test_shut_device_changes_nothing() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    env = load_sim_config(EXAMPLE_SIM).environment
    with_device, without = RocketDynamics(rocket, env), RocketDynamics(replace(rocket, drag_device=None), env)
    y = with_device.initial_state()
    y[IZ], y[IVZ] = 50.0, -20.0
    y[6:10] = quat.from_axis_angle(ROTATE_Y, 0.1)
    inputs = with_device.new_inputs()
    assert np.array_equal(with_device.derivatives(0.0, y, inputs), without.derivatives(0.0, y, inputs))
    assert with_device.device_drag(y, inputs) == 0.0
    inputs.brake_fraction = 1.0
    assert with_device.derivatives(0.0, y, inputs)[IVZ] > without.derivatives(0.0, y, inputs)[IVZ]
    assert with_device.device_drag(y, inputs) > 10.0


def test_terminal_speed_sizing_round_trip() -> None:
    mass, gravity, density = 1.432, 9.80665, 1.1825
    area = drag_area_for_terminal_speed(mass, gravity, density, 20.0)
    assert area == pytest.approx(0.0592, abs=0.0005)
    assert terminal_speed(mass, gravity, 0.5 * density * area) == pytest.approx(20.0)
    assert terminal_speed(mass, gravity, 0.0) == math.inf
    assert plate_area(0.0566) == pytest.approx(0.0472, abs=0.0001)
    assert drag_factor(None, 1.0, density) == 0.0
    assert drag_factor(DragDeviceConfig(area, 0.15, 0.5, 0.3), 0.5, density) == pytest.approx(0.25 * density * area)
