import math
from pathlib import Path

import numpy as np
import pytest

from rocketsim.motors import MotorFileError, load_motor, parse_eng, parse_motor_yaml

ENG_TEXT = """
; a comment line
; another one
TEST_F20 29 100 0-5-10 0.030 0.080 Example
0.1 30.0
0.5 20.0
1.0 20.0
1.1 0.0
;
"""


def test_parse_eng_header_and_curve() -> None:
    motor = parse_eng(ENG_TEXT)
    assert motor.name == "TEST_F20"
    assert motor.diameter == pytest.approx(0.029)
    assert motor.length == pytest.approx(0.100)
    assert motor.propellant_mass == pytest.approx(0.030)
    assert motor.total_mass == pytest.approx(0.080)
    assert motor.case_mass == pytest.approx(0.050)
    assert motor.burn_time == pytest.approx(1.1)
    # The curve starts at (0, 0) because the first sample is after ignition.
    assert motor.times[0] == 0.0 and motor.thrusts[0] == 0.0
    # Trapezoids: 0.1*15 + 0.4*25 + 0.5*20 + 0.1*10 = 1.5 + 10 + 10 + 1 = 22.5 N s
    assert motor.total_impulse == pytest.approx(22.5)


def test_thrust_and_impulse_interpolation() -> None:
    motor = parse_eng(ENG_TEXT)
    assert motor.thrust_at(-0.1) == 0.0
    assert motor.thrust_at(0.05) == pytest.approx(15.0)
    assert motor.thrust_at(0.75) == pytest.approx(20.0)
    assert motor.thrust_at(1.05) == pytest.approx(10.0)
    assert motor.thrust_at(5.0) == 0.0
    assert motor.impulse_at(0.1) == pytest.approx(1.5)
    assert motor.impulse_at(0.5) == pytest.approx(11.5)
    assert motor.impulse_at(10.0) == pytest.approx(22.5)


def test_propellant_follows_thrust_curve() -> None:
    motor = parse_eng(ENG_TEXT)
    assert motor.propellant_burned_at(0.0) == 0.0
    assert motor.propellant_burned_at(0.5) == pytest.approx(0.030 * 11.5 / 22.5)
    assert motor.propellant_burned_at(2.0) == pytest.approx(0.030)
    assert motor.exhaust_velocity == pytest.approx(22.5 / 0.030)


def test_parse_eng_several_motors_and_selection() -> None:
    text = ENG_TEXT + "\nOTHER 18 70 0 0.010 0.020 Example\n0.2 10.0\n0.4 0.0\n"
    assert parse_eng(text).name == "TEST_F20"
    other = parse_eng(text, name="OTHER")
    assert other.total_impulse == pytest.approx(0.2 * 5.0 + 0.2 * 5.0)
    with pytest.raises(MotorFileError, match="no motor named"):
        parse_eng(text, name="MISSING")


def test_parse_eng_errors() -> None:
    with pytest.raises(MotorFileError, match="header"):
        parse_eng("BAD 29 100\n0.1 1.0\n0.2 0.0\n")
    with pytest.raises(MotorFileError, match="end with zero"):
        parse_eng("M 29 100 0 0.01 0.02 X\n0.1 10.0\n0.2 5.0\n")
    with pytest.raises(MotorFileError, match="strictly increase"):
        parse_eng("M 29 100 0 0.01 0.02 X\n0.2 10.0\n0.1 0.0\n")
    with pytest.raises(MotorFileError, match="total mass"):
        parse_eng("M 29 100 0 0.05 0.02 X\n0.1 10.0\n0.2 0.0\n")


def test_parse_motor_yaml() -> None:
    text = """
name: Y
propellant_mass_g: 40
total_mass_g: 100
throttleable: true
throttle_lag_s: 0.1
ignition_delay_s: {mean: 0.2, spread: 0.05}
thrust_curve:
  - [0.0, 0.0]
  - [0.5, 10.0]
  - [1.5, 10.0]
  - [2.0, 0.0]
"""
    motor = parse_motor_yaml(text)
    assert motor.propellant_mass == pytest.approx(0.040)
    assert motor.total_mass == pytest.approx(0.100)
    assert motor.throttleable is True
    assert motor.throttle_lag == 0.1
    assert motor.ignition_delay_mean == 0.2
    assert motor.ignition_delay_spread == 0.05
    assert motor.total_impulse == pytest.approx(2.5 + 10.0 + 2.5)
    assert np.all(np.diff(motor.times) > 0)


def test_parse_motor_yaml_errors() -> None:
    with pytest.raises(MotorFileError, match="missing required key"):
        parse_motor_yaml("name: X\npropellant_mass_g: 1\n")
    with pytest.raises(MotorFileError, match="pairs"):
        parse_motor_yaml("name: X\npropellant_mass_g: 1\ntotal_mass_g: 2\nthrust_curve: [[0, 0, 1]]\n")
    with pytest.raises(MotorFileError, match="zero total impulse"):
        parse_motor_yaml("name: X\npropellant_mass_g: 1\ntotal_mass_g: 2\nthrust_curve: [[0, 0], [1, 0]]\n")


def test_load_motor_by_extension(tmp_path: Path) -> None:
    eng = tmp_path / "m.eng"
    eng.write_text(ENG_TEXT)
    assert load_motor(eng).name == "TEST_F20"
    with pytest.raises(MotorFileError, match="not found"):
        load_motor(tmp_path / "missing.eng")
    bad = tmp_path / "m.txt"
    bad.write_text("x")
    with pytest.raises(MotorFileError, match="extension"):
        load_motor(bad)


def test_example_motor_files_load() -> None:
    root = Path(__file__).resolve().parent.parent / "configs" / "motors"
    ascent = load_motor(root / "example_g40.eng")
    landing = load_motor(root / "example_g120_landing.yaml")
    assert 80.0 < ascent.total_impulse < 160.0
    assert 100.0 < landing.total_impulse < 140.0
    assert math.isclose(ascent.propellant_burned_at(ascent.burn_time), ascent.propellant_mass)


def test_brake_landing_motor_matches_its_design() -> None:
    from rocketsim.config import load_rocket_config
    from tests.conftest import EXAMPLE_ROCKET

    rocket = load_rocket_config(EXAMPLE_ROCKET)
    motor = rocket.motor("landing").spec
    assert motor.name == "EXAMPLE_G127_LANDING_BRAKE"
    assert 126.0 < motor.total_impulse < 128.0
    assert motor.burn_time == pytest.approx(8.2)
    tail_start = 1.56
    mass_at_tail = rocket.descent_mass - motor.propellant_burned_at(tail_start)
    tail_ratio = motor.thrust_at(tail_start) / (mass_at_tail * 9.80665)
    assert 0.90 < tail_ratio < 0.94
    assert motor.thrust_at(0.12) / (rocket.descent_mass * 9.80665) == pytest.approx(2.2, abs=0.05)  # hard part over weight
