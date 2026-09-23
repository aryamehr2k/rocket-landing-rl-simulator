import math
from pathlib import Path
from typing import Callable

import pytest
import yaml

from rocketsim.config import ConfigError, ControlConfig, load_rocket_config
from rocketsim.simconfig import load_sim_config, physics_steps_per_control_step
from tests.conftest import EXAMPLE_ROCKET, EXAMPLE_SIM

Change = Callable[[dict], None]


def test_example_rocket_loads_in_si() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    assert rocket.name == "example_tvc"
    assert rocket.airframe.dry_mass == pytest.approx(1.2)
    assert rocket.airframe.dry_roll_inertia == pytest.approx(0.0015)
    assert rocket.airframe.length == pytest.approx(0.9)
    assert rocket.gimbal.max_angle == pytest.approx(math.radians(7.0))
    assert rocket.gimbal.servo_rate_limit == pytest.approx(math.radians(300.0))
    assert rocket.legs.max_tilt == pytest.approx(math.radians(10.0))
    assert rocket.legs.count == 4
    assert rocket.motor("ascent").spec.name == "EXAMPLE_G40"
    assert rocket.motor("landing").position == pytest.approx(0.69)
    assert rocket.motor("landing").ignition_delay_mean == pytest.approx(0.15)
    assert rocket.feet_station == pytest.approx(1.02)
    assert rocket.reference_area == pytest.approx(math.pi * 0.075 ** 2 / 4)


def test_servo_calibration_chain() -> None:
    gimbal = load_rocket_config(EXAMPLE_ROCKET).gimbal
    for cal in (gimbal.pitch_servo, gimbal.yaw_servo):
        # 7 deg of gimbal is 14 deg of servo at 10 us per degree: 1500 + 140 us.
        assert cal.pulse_for_gimbal(math.radians(7.0)) == pytest.approx(1640.0)
        assert cal.pulse_for_gimbal(math.radians(-7.0)) == pytest.approx(1360.0)
        assert cal.gimbal_for_pulse(1640.0) == pytest.approx(math.radians(7.0))
        assert cal.quantize(1500.4) == 1500.0
        assert cal.quantize(1500.5) == 1501.0
        assert cal.quantize(2500.0) == 2000.0


def test_example_sim_config_loads() -> None:
    sim = load_sim_config(EXAMPLE_SIM)
    assert sim.simulation.dt == pytest.approx(0.005)
    assert sim.environment.site_elevation == 300.0
    assert sim.wind.gust_std == 0.0
    assert sim.wind.steady_direction == 0.0


def test_control_rate_must_divide_physics_rate() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    sim = load_sim_config(EXAMPLE_SIM)
    assert physics_steps_per_control_step(sim.simulation, rocket.control) == 4
    with pytest.raises(ConfigError, match="integer multiple"):
        physics_steps_per_control_step(sim.simulation, ControlConfig(60.0, 1))


def write_modified(tmp_path: Path, change: Change) -> Path:
    """Copy the example rocket YAML with one change applied by `change(data)`."""
    data = yaml.safe_load(EXAMPLE_ROCKET.read_text())
    change(data)
    out = tmp_path / "rocket.yaml"
    out.write_text(yaml.safe_dump(data))
    return out


def with_motor_paths(data: dict) -> None:
    for role in ("ascent", "landing"):
        data["motors"][role]["file"] = str(EXAMPLE_ROCKET.parent / data["motors"][role]["file"])


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["airframe"].pop("dry_mass_g"), "airframe.dry_mass_g is required"),
        (lambda d: d["airframe"].__setitem__("dry_mass_g", -5), "dry_mass_g must be > 0"),
        (lambda d: d["airframe"].__setitem__("dry_mass", 1200), "not a known setting"),
        (lambda d: d["airframe"].pop("dry_roll_inertia_kgm2"), "dry_roll_inertia_kgm2 is required"),
        (lambda d: d["aero"].__setitem__("cp_from_nose_mm", 5000), "cp_from_nose_mm must be <= 900"),
        (lambda d: d["gimbal"]["servos"]["pitch"].__setitem__("sign", 2), "sign must be 1 or -1"),
        (lambda d: d["gimbal"].__setitem__("max_angle_deg", 60), "outside min_us..max_us"),
        (lambda d: d["gimbal"].__setitem__("servo_deadband_deg", 8), "smaller than max_angle_deg"),
        (lambda d: d["gimbal"]["servos"]["yaw"].__setitem__("center_us", 1500.5), "multiple of pulse_resolution_us"),
        (lambda d: d["gimbal"]["servos"].pop("yaw"), "servos.yaw must be a mapping"),
        (lambda d: d["motors"]["ascent"].__setitem__("file", "nope.eng"), "motor file not found"),
        (lambda d: d["motors"]["ascent"].pop("ignition_delay_s"), "ignition_delay_s is required"),
        (lambda d: d["legs"].__setitem__("span_mm", "wide"), "span_mm must be a number"),
        (lambda d: d["legs"].__setitem__("count", 2), "count must be >= 3"),
        (lambda d: d["control"].__setitem__("action_delay_steps", 1.5), "must be an integer"),
        (lambda d: d["motors"].pop("landing"), "motors.landing must be a mapping"),
    ],
)
def test_validation_errors(tmp_path: Path, change: Change, message: str) -> None:
    def apply(data: dict) -> None:
        with_motor_paths(data)
        change(data)

    path = write_modified(tmp_path, apply)
    with pytest.raises(ConfigError, match=message):
        load_rocket_config(path)


def test_motor_yaml_default_ignition_delay_is_used(tmp_path: Path) -> None:
    def apply(data: dict) -> None:
        with_motor_paths(data)
        data["motors"]["landing"].pop("ignition_delay_s")

    rocket = load_rocket_config(write_modified(tmp_path, apply))
    assert rocket.motor("landing").ignition_delay_mean == pytest.approx(0.15)
    assert rocket.motor("landing").ignition_delay_spread == pytest.approx(0.08)


def test_missing_file_and_bad_top_level(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        load_rocket_config(tmp_path / "missing.yaml")
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_sim_config(bad)
