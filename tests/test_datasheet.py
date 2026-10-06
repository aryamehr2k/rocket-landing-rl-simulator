"""The vehicle data sheet: building config files from it, and the entries it refuses."""

import math
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from rocketsim.hop.datasheet import DatasheetError, Derived, derive, load_datasheet
from rocketsim.hop.datasheet_docs import request_markdown, template_text
from rocketsim.hop.datasheet_fields import GROUPS
from rocketsim.hop.mission import load_mission_config, with_targets
from rocketsim.hop.simulation import HopSimulation
from rocketsim.hop.training import load_hop_training_config
from rocketsim.hop.training_text import mission_time_budget
from rocketsim.hop.vehicle import load_vehicle_config
from rocketsim.hop.vehicle_check import check_vehicle
from rocketsim.hop.vehicle_files import BuiltVehicle, build_vehicle, config_paths
from rocketsim.simconfig import load_sim_config
from rocketsim.yaml_section import ConfigError
from tests.conftest import REPO_ROOT

DATASHEETS = REPO_ROOT / "configs" / "datasheets"
EXAMPLE = DATASHEETS / "example.yaml"
TEMPLATE = DATASHEETS / "template.yaml"
TRAINING_BASE = REPO_ROOT / "configs" / "training" / "hop.yaml"
HAND_WRITTEN = REPO_ROOT / "configs" / "vehicles" / "electric_hopper.yaml"
HAND_WRITTEN_MISSION = REPO_ROOT / "configs" / "missions" / "hop_50m.yaml"
GAIN_TOLERANCE = 0.1  # the worked-out attitude gains may differ this much from the hand-tuned ones


def build(tmp_path: Path, sheet: Path = EXAMPLE, **options: Any) -> BuiltVehicle:
    return build_vehicle(sheet, tmp_path / "configs", TRAINING_BASE, **options)


def derived(sheet: Path) -> Derived:
    return derive(load_datasheet(sheet), load_sim_config(TRAINING_BASE))


def edited_sheet(tmp_path: Path, changes: dict[str, dict[str, Any]]) -> Path:
    data = yaml.safe_load(EXAMPLE.read_text())
    for group, values in changes.items():
        data.setdefault(group, {}).update(values)
    path = tmp_path / "edited.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def refusal(sheet: Path) -> str:
    with pytest.raises(DatasheetError) as error:
        derived(sheet)
    return str(error.value)


def without_values(text: str) -> list[str]:
    """The comment and key lines of a sheet, values and the file header left out."""
    body = text[text.index("\nvehicle:"):]
    return [re.sub(r"^(\s+\w+:).*?(#.*)$", r"\1 \2", line) for line in body.splitlines()]


def test_template_and_request_page_match_the_entry_list() -> None:
    assert TEMPLATE.read_text() == template_text(), "run python scripts/write_data_request.py"
    assert (REPO_ROOT / "docs" / "vehicle_data_request.md").read_text() == request_markdown()
    assert without_values(EXAMPLE.read_text()) == without_values(TEMPLATE.read_text())
    template = yaml.safe_load(TEMPLATE.read_text())
    for group in GROUPS:
        assert list(template[group.key]) == [entry.key for entry in group.entries]
        for entry in group.entries:
            assert template[group.key][entry.key] == (None if entry.required else entry.default)


def test_example_sheet_builds_and_its_files_load(tmp_path: Path) -> None:
    built = build(tmp_path)
    assert built.name == "example_hopper"
    assert all(path.is_file() for path in built.paths.all())
    assert built.training.vehicle_path.resolve() == built.paths.vehicle.resolve()
    assert built.training.mission_path.resolve() == built.paths.mission.resolve()
    assert built.training.mission_variation == load_hop_training_config(TRAINING_BASE).mission_variation
    assert built.vehicle.sensors.gps is not None
    assert load_mission_config(built.paths.mission) == built.mission
    assert check_vehicle(built).warnings == ()


def test_example_sheet_reproduces_the_hand_written_vehicle(tmp_path: Path) -> None:
    # Everything but the roll gains: the hand-tuned ones leave the sampled roll loop barely damped.
    built = build(tmp_path)
    vehicle, hand = built.vehicle, load_vehicle_config(HAND_WRITTEN)
    for field in ("dry_mass", "dry_cg", "dry_pitch_inertia", "dry_roll_inertia", "length", "reference_diameter"):
        assert getattr(vehicle.body.airframe, field) == pytest.approx(getattr(hand.body.airframe, field), rel=1e-3)
    assert vehicle.body.gimbal == hand.body.gimbal
    legs = vehicle.body.legs
    assert replace(legs, max_vertical_speed=hand.body.legs.max_vertical_speed) == hand.body.legs
    assert legs.max_vertical_speed == pytest.approx(hand.body.legs.max_vertical_speed, rel=1e-3)
    assert vehicle.body.aero == hand.body.aero
    assert vehicle.motor.position == hand.motor.position
    assert vehicle.motor.spec.max_thrust == hand.motor.spec.max_thrust
    assert vehicle.sensors.gps == hand.sensors.gps and vehicle.sensors.barometer == hand.sensors.barometer
    assert vehicle.estimator == hand.estimator and vehicle.altitude == hand.altitude and vehicle.safety == hand.safety
    assert vehicle.attitude.attitude.kp == pytest.approx(hand.attitude.attitude.kp, rel=GAIN_TOLERANCE)
    assert vehicle.attitude.attitude.kd == pytest.approx(hand.attitude.attitude.kd, rel=GAIN_TOLERANCE)
    assert replace(vehicle.roll, kp=hand.roll.kp, kd=hand.roll.kd) == hand.roll
    assert replace(built.mission, name="", source="") == replace(load_mission_config(HAND_WRITTEN_MISSION), name="",
                                                                  source="")


def test_built_vehicle_flies_the_mission_with_the_pid(tmp_path: Path) -> None:
    built = build(tmp_path)
    vehicle = replace(built.vehicle, controllers=replace(built.vehicle.controllers, steering="pid", throttle="pid"))
    assert built.sim.wind.steady == 0.0 and built.sim.wind.gust_std == 0.0
    result = HopSimulation(vehicle, built.mission, built.sim, seed=0).run()
    assert result.success, result.summary()
    assert result.miss_distance < 5.0


def test_empty_required_entries_are_all_listed_at_once() -> None:
    with pytest.raises(DatasheetError) as error:
        load_datasheet(TEMPLATE)
    required = [f"{group.key}.{entry.key}" for group in GROUPS for entry in group.entries if entry.required]
    message = str(error.value)
    assert f"{len(required)} required entries are empty" in message
    for name in required:
        assert f"{name}:" in message


def test_bad_entries_are_reported_together(tmp_path: Path) -> None:
    sheet = edited_sheet(tmp_path, {
        "vehicle": {"total_mass_g": "heavy"},
        "fan": {"max_thrust": 30.0},
        "thrust_vectoring": {"pitch_servo_sign": 2},
        "legs": {"count": 2, "max_tilt_deg": 95.0},
    })
    with pytest.raises(DatasheetError) as error:
        load_datasheet(sheet)
    message = str(error.value)
    assert "5 entries need fixing" in message
    assert "vehicle.total_mass_g must be a number" in message
    assert "fan.max_thrust is not a known entry (did you mean max_thrust_n?)" in message
    assert "pitch_servo_sign must be one of 1, -1" in message
    assert "legs.count must be at least 3" in message
    assert "legs.max_tilt_deg must be at most 90" in message


def test_yaml_typos_name_the_line(tmp_path: Path) -> None:
    sheet = tmp_path / "typo.yaml"
    sheet.write_text(EXAMPLE.read_text().replace("  model:     ", "  model: EDF 90mm: 12S", 1))
    with pytest.raises(DatasheetError, match=r"typo\.yaml line \d+: .*space after its colon"):
        load_datasheet(sheet)


def test_impossible_layouts_are_refused(tmp_path: Path) -> None:
    assert "pivot_mm must be below the balance point" in refusal(
        edited_sheet(tmp_path, {"thrust_vectoring": {"pivot_mm": 300.0}}))
    assert "cannot lift" in refusal(edited_sheet(tmp_path, {"fan": {"max_thrust_n": 15.0}}))
    message = refusal(edited_sheet(tmp_path, {
        "vehicle": {"centre_of_pressure_mm": 950.0},
        "thrust_vectoring": {"max_angle_deg": 15.0, "linkage_ratio": 0.25, "servo_deadband_deg": 15.0},
    }))
    assert "vehicle.centre_of_pressure_mm (950 mm) is beyond the body length" in message
    assert "servo_deadband_deg must be smaller than max_angle_deg" in message
    assert "max_angle_deg (15 deg) needs pulses 600 us either side" in message


def test_rates_the_simulator_cannot_step_are_refused(tmp_path: Path) -> None:
    for rate in (400, 75):
        message = refusal(edited_sheet(tmp_path, {"sensors": {"control_rate_hz": rate}}))
        assert f"sensors.control_rate_hz ({rate} Hz) must divide" in message and "200, 100, 50, 40, 25, 20" in message
    message = refusal(edited_sheet(tmp_path, {"sensors": {"imu_rate_hz": 1000, "barometer_rate_hz": 400}}))
    assert "sensors.imu_rate_hz (1000 Hz) is faster than the simulator's 200 Hz" in message
    assert "sensors.barometer_rate_hz (400 Hz)" in message


def test_mission_plans_that_cannot_land_are_refused(tmp_path: Path) -> None:
    message = refusal(edited_sheet(tmp_path, {"mission": {
        "climb_acceleration_mps2": 1.0, "descent_speed_mps": 2.5, "final_speed_mps": 0.5, "final_height_m": 3.0,
    }}))
    assert "mission.final_height_m (3 m) is too low" in message and "at least 4.0 m" in message
    message = refusal(edited_sheet(tmp_path, {"mission": {"final_speed_mps": 1.6}, "legs": {"max_drop_height_mm": 80}}))
    assert "mission.final_speed_mps (1.6 m/s)" in message and "1.25 m/s the legs survive" in message
    message = refusal(edited_sheet(tmp_path, {"mission": {"max_flight_time_s": 40.0}}))
    assert "mission.max_flight_time_s (40 s) is shorter than the 2 s pad calibration plus the 48 s plan" in message


def test_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    build(tmp_path)
    with pytest.raises(ConfigError, match="exist already"):
        build(tmp_path)
    assert build(tmp_path, force=True).name == "example_hopper"
    assert build(tmp_path, name="second").paths.vehicle.name == "second.yaml"


def test_files_the_builder_did_not_write_are_never_replaced(tmp_path: Path) -> None:
    shared = tmp_path / "configs" / "training" / "hop.yaml"
    shared.parent.mkdir(parents=True)
    shared.write_text(TRAINING_BASE.read_text())
    for base in (TRAINING_BASE, shared):
        with pytest.raises(ConfigError, match="never replaces them") as error:
            build_vehicle(EXAMPLE, tmp_path / "configs", base, name="hop", force=True)
        assert "configs/training/hop.yaml" in str(error.value)
    assert shared.read_text() == TRAINING_BASE.read_text()


def test_a_file_the_loaders_reject_is_reported_against_the_sheet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The sheet checks catch this one first; without them the trial load still stops it and writes nothing.
    sheet = edited_sheet(tmp_path, {"thrust_vectoring": {"max_angle_deg": 40.0}})
    monkeypatch.setattr("rocketsim.hop.datasheet.sheet_problems", lambda *args: [])
    with pytest.raises(DatasheetError, match="builds a file the simulator refuses") as error:
        build(tmp_path, sheet)
    assert "configs/vehicles/example_hopper.yaml: gimbal.servos.pitch.min_us" in str(error.value)
    assert tempfile.gettempdir() not in str(error.value).replace(str(tmp_path), "")
    assert not any(path.exists() for path in config_paths(tmp_path / "configs", "example_hopper").all())


def test_training_missions_fit_the_battery_and_the_time_limit(tmp_path: Path) -> None:
    sheet = edited_sheet(tmp_path, {
        "fan": {"battery_time_s": 60.0},
        "mission": {"target_height_m": 15.0, "hover_time_s": 5.0, "climb_speed_mps": 2.0, "descent_speed_mps": 1.5},
    })
    built = build(tmp_path, sheet)
    variation = built.training.mission_variation
    assert variation.target_altitude[0] == 10.0 and variation.hover_time[0] == 3.0
    assert 15.0 <= variation.target_altitude[1] < 60.0 and 5.0 <= variation.hover_time[1] < 12.0
    longest = with_targets(built.mission, variation.target_altitude[1], variation.hover_time[1])
    budget = mission_time_budget(60.0, built.mission, built.sim.simulation.pad_hold_time)
    assert longest.planned_duration <= budget


def test_check_warns_about_thrust_gps_battery_and_roll(tmp_path: Path) -> None:
    sheet = edited_sheet(tmp_path, {
        "fan": {"max_thrust_n": 28.0, "battery_time_s": 50.0, "reaction_torque_nm_per_n": 0.006},
        "sensors": {"gps_rate_hz": 0},
        "roll_control": {"response_ms": 150.0},
    })
    built = build(tmp_path, sheet)
    assert built.vehicle.sensors.gps is None
    check = check_vehicle(built)
    warnings = " | ".join(check.warnings)
    for text in ("thrust to weight 1.31 is below 1.4", "hover throttle 0.76", "no GPS",
                 "the battery lasts 50 s", "the fan twists the vehicle", "fan's twist turns the vehicle"):
        assert text in warnings
    assert math.isclose(check.hover_throttle * check.thrust_to_weight, 1.0)
