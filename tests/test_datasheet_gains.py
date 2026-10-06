"""The gains and inertias the data sheet builder works out."""

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from rocketsim.actuators import RollActuator
from rocketsim.hop.datasheet import (
    ATTITUDE_DAMPING, ATTITUDE_FREQUENCY, ROLL_DAMPING, ROLL_LINEAR_ANGLE_DEG, Datasheet, Derived, derive,
    load_datasheet, roll_frequency_limit, solid_cylinder_pitch_inertia, solid_cylinder_roll_inertia,
)
from rocketsim.physics import RocketDynamics
from rocketsim.simconfig import load_sim_config
from rocketsim.units import deg_to_rad
from tests.test_datasheet import EXAMPLE, TRAINING_BASE, build, derived, edited_sheet

PHYSICS_DT = 0.005
ROLL_START_DEG = 10.0
ROLL_TEST_TIME_S = 4.0
ROLL_SETTLED_SHARE = 0.05  # of the starting roll error, left in the last second of the roll loop test


def test_gains_put_the_loops_where_the_constants_say(tmp_path: Path) -> None:
    d = derived(EXAMPLE)
    assert d.authority * d.attitude_kp == pytest.approx(ATTITUDE_FREQUENCY ** 2)
    assert d.authority * d.attitude_kd == pytest.approx(2.0 * ATTITUDE_DAMPING * ATTITUDE_FREQUENCY)
    heavier = derived(edited_sheet(tmp_path, {"vehicle": {"pitch_inertia_kgm2": 0.15}}))
    assert heavier.attitude_kp / d.attitude_kp == pytest.approx(heavier.pitch_inertia / d.pitch_inertia)
    # The roll loop I T s^3 + I s^2 + kd s + kp has a pole pair at the reported frequency and ROLL_DAMPING.
    roots = np.roots([d.roll_inertia * d.roll_lag, d.roll_inertia, d.roll_kd, d.roll_kp])
    pair = roots[np.argmax(roots.imag)]
    assert abs(pair) == pytest.approx(d.roll_frequency)
    assert -pair.real / abs(pair) == pytest.approx(ROLL_DAMPING)
    assert d.roll_lag == pytest.approx(0.05 + 1.5 / 50.0)
    weak_roll = derived(edited_sheet(tmp_path, {"roll_control": {"max_torque_nm": 0.01}}))
    assert weak_roll.roll_kp == pytest.approx(0.01 / deg_to_rad(ROLL_LINEAR_ANGLE_DEG))
    assert weak_roll.roll_frequency < roll_frequency_limit(weak_roll.roll_lag)


def test_roll_loop_stays_damped_with_a_slow_roll_actuator(tmp_path: Path) -> None:
    for response_ms in (0.0, 50.0, 100.0, 150.0):
        sheet = load_datasheet(edited_sheet(tmp_path, {"roll_control": {"response_ms": response_ms}}))
        roll = fly_roll_loop(sheet, derive(sheet, load_sim_config(TRAINING_BASE)))
        last_second = roll[-int(1.0 / PHYSICS_DT):]
        assert np.max(np.abs(last_second)) < ROLL_SETTLED_SHARE * deg_to_rad(ROLL_START_DEG), response_ms


def fly_roll_loop(sheet: Datasheet, d: Derived) -> np.ndarray:
    """Roll angle after a roll error, with the sheet's control rate, command delay and roll actuator."""
    sensors = sheet["sensors"]
    steps = round(1.0 / (sensors["control_rate_hz"] * PHYSICS_DT))
    actuator = RollActuator(sheet["roll_control"]["max_torque_nm"], sheet.si("roll_control", "response_ms"), PHYSICS_DT)
    pending = deque([0.0] * sensors["command_delay_steps"])
    angle, rate, history = deg_to_rad(ROLL_START_DEG), 0.0, []
    for _ in range(round(ROLL_TEST_TIME_S / (steps * PHYSICS_DT))):
        pending.append(-(d.roll_kp * angle + d.roll_kd * rate))
        actuator.command(pending.popleft())
        for _ in range(steps):
            rate += actuator.step() / d.roll_inertia * PHYSICS_DT
            angle += rate * PHYSICS_DT
            history.append(angle)
    return np.array(history)


def test_inertias_from_a_solid_cylinder_match_the_physics(tmp_path: Path) -> None:
    sheet = edited_sheet(tmp_path, {"vehicle": {"pitch_inertia_kgm2": None, "roll_inertia_kgm2": None}})
    built = build(tmp_path, sheet)
    d = built.derived
    radius, length = 0.055, 0.7
    assert d.dry_mass == pytest.approx(1.8)
    assert d.dry_pitch_inertia == pytest.approx(solid_cylinder_pitch_inertia(1.8, radius, length))
    assert d.dry_roll_inertia == pytest.approx(solid_cylinder_roll_inertia(1.8, radius))
    dynamics = RocketDynamics(built.vehicle.body, built.sim.environment)
    props = dynamics.mass_properties(dynamics.initial_state())
    assert props.pitch_inertia == pytest.approx(d.pitch_inertia, rel=1e-3)
    assert props.roll_inertia == pytest.approx(d.roll_inertia, rel=1e-3)
    assert props.cg == pytest.approx(d.cg, abs=1e-4)  # the files keep a tenth of a millimetre
