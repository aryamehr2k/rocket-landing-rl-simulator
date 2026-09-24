"""The design tool reproduces the example rocket's brake and motor and its landing rate."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rocketsim.config import load_rocket_config
from rocketsim.flightcomputer import HALF_STEP
from rocketsim.landing_design import (
    ascent_apogee, descent_profile, design_from_motor, size_burn, size_descent, speed_at_heights,
)
from rocketsim.landing_montecarlo import ErrorBudget, draw_errors, nominal_draws, simulate_landings
from rocketsim.landing_sensitivity import ROW_NAMES, StopModel, single_error_draws, stop_sensitivities
from rocketsim.landing_trigger import LandingTrigger
from rocketsim.simconfig import load_sim_config
from tests.conftest import EXAMPLE_ROCKET, EXAMPLE_SIM, REPO_ROOT

CONTROL_DT, DT = 0.02, 0.005
FREEFALL_ROCKET = REPO_ROOT / "configs" / "rockets" / "example_tvc_freefall.yaml"
FREEFALL_MOTOR = REPO_ROOT / "configs" / "motors" / "example_g120_landing.yaml"
TOOL = REPO_ROOT / "scripts" / "design_landing_burn.py"


@pytest.fixture(scope="module")
def world() -> tuple:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    env = load_sim_config(EXAMPLE_SIM).environment
    latency = (1 + HALF_STEP) * CONTROL_DT
    trigger = LandingTrigger(rocket, rocket.computer.landing_trigger, env, decision_latency=latency)
    return rocket, env, trigger


def test_brake_sizing_and_terminal_speed(world) -> None:
    rocket, env, _ = world
    sizing = size_descent(rocket, env, 20.0, None, 0.05)
    assert sizing.device_drag_area == pytest.approx(0.0566, abs=0.001)
    assert sizing.total_drag_area == pytest.approx(0.0592, abs=0.0005)
    given = size_descent(rocket, env, None, rocket.drag_device.drag_area, 0.05)
    assert given.terminal_speed == pytest.approx(20.0, abs=0.3)
    assert given.terminal_speed_low_drag > given.terminal_speed > given.terminal_speed_high_drag
    apogee = ascent_apogee(rocket, env)
    assert 95.0 < apogee < 115.0
    heights, speeds = descent_profile(given, env.gravity, apogee, rocket.drag_device, 8.0)
    at_marks = speed_at_heights(heights, speeds, [80.0, 40.0, 20.0])
    assert at_marks == pytest.approx([17.0, 19.6, 19.9], abs=0.4)


def test_hard_part_and_tail_sizing_reproduce_the_example_motor(world) -> None:
    rocket, env, _ = world
    sizing = size_descent(rocket, env, None, rocket.drag_device.drag_area, 0.05)
    design = size_burn(rocket, sizing, env.gravity, 31.0, 0.92, 6.5, sizing.terminal_speed + 0.3, 0.5, 1.0)
    assert design.hard_end == pytest.approx(1.41, abs=0.03)
    assert design.tail_thrust == pytest.approx(12.6, abs=0.3)
    assert 125.0 < design.motor.total_impulse < 130.0
    assert design.motor.total_mass == pytest.approx(0.170, abs=0.003)


def test_motor_strength_window_and_landing_rates(world) -> None:
    rocket, env, trigger = world
    apogee = 105.5
    scales = 1.0 + np.arange(-8, 9) / 100.0
    sweep = simulate_landings(rocket, trigger, apogee, nominal_draws(0.15, scales), CONTROL_DT, DT)
    landed = dict(zip(np.round(scales, 2), sweep.landed))
    assert all(landed[s] for s in (0.97, 0.98, 0.99, 1.0, 1.01, 1.02))
    assert not any(landed[s] for s in (1.03, 1.05, 1.08))
    nominal = simulate_landings(rocket, trigger, apogee, nominal_draws(0.15, np.array([1.0])), CONTROL_DT, DT)
    assert nominal.command_speed[0] == pytest.approx(19.9, abs=0.2)
    assert 16.0 < nominal.command_height[0] < 19.0
    assert 1.0 < nominal.stop_height[0] < 2.0 and nominal.touchdown_speed[0] < 2.0
    rng = np.random.default_rng(1)
    tight = simulate_landings(rocket, trigger, apogee, draw_errors(ErrorBudget(), 0.15, 600, rng), CONTROL_DT, DT)
    assert tight.landed.mean() > 0.9
    loose = ErrorBudget(thrust_error=0.05, mass_error=0.036)
    loose_outcome = simulate_landings(rocket, trigger, apogee, draw_errors(loose, 0.15, 600, rng), CONTROL_DT, DT)
    assert 0.6 < loose_outcome.landed.mean() < tight.landed.mean()


def test_closed_form_sensitivities_match_the_1d_model_one_error_at_a_time(world) -> None:
    rocket, env, trigger = world
    apogee = 105.5
    sizing = size_descent(rocket, env, None, rocket.drag_device.drag_area, 0.05)
    design = design_from_motor(rocket.motor("landing").spec, rocket.descent_mass, sizing.terminal_speed)
    nominal = simulate_landings(rocket, trigger, apogee, nominal_draws(0.15, np.array([1.0])), CONTROL_DT, DT)
    model = StopModel.from_nominal(
        design, env.gravity, 0.5, float(nominal.stop_height[0]), float(nominal.climb_after_stop[0]),
        float(nominal.touchdown_speed[0]), float(nominal.stop_time[0]),
    )
    speed = float(nominal.thrust_on_speed[0])
    rows = stop_sensitivities(design, sizing, speed, trigger.stopping_distance(speed), model, 0.03, 0.3)
    singles = single_error_draws(0.15, rocket.descent_mass, 0.03, 0.3)
    checked = simulate_landings(rocket, trigger, apogee, singles, CONTROL_DT, DT)
    assert [row.name.split(" ")[0] for row in rows] == [name.split(" ")[0] for name in ROW_NAMES]
    for index, row in enumerate(rows):
        low, high = checked.touchdown_speed[2 * index], checked.touchdown_speed[2 * index + 1]
        assert row.touchdown_if_low == pytest.approx(low, abs=0.3), row.name
        assert row.touchdown_if_high == pytest.approx(high, abs=0.3), row.name
        assert row.touchdown_if_low < row.touchdown_if_high  # this design is forgiving on the low side
    assert model.touchdown_if_low(0.0) == pytest.approx(float(nominal.touchdown_speed[0]), abs=1e-9)
    assert 0.5 < model.stop_height < 2.5 and model.touchdown_if_low(model.stop_height) < 1.0
    low_margin, high_margin = model.margins(rocket.legs.max_vertical_speed)
    assert low_margin > model.stop_height and 0.8 < high_margin < 1.5  # a 60 ms early igniter (1.2 m) is the edge
    assert model.touchdown_if_high(high_margin + 0.1) > rocket.legs.max_vertical_speed
    assert model.touchdown_if_high(30.0) > 10.0  # the tail runs out and the rocket falls freely


def test_script_runs_on_the_example(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--rocket", str(EXAMPLE_ROCKET), "--terminal-speed", "20", "--draws", "200"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    )
    assert "2. Landing motor" in result.stdout and "hard part 31.0 N" in result.stdout
    assert "terminal speed 20.0 m/s" in result.stdout and "= terminal speed + margin" in result.stdout
    assert "4. Stop point sensitivity" in result.stdout and "7. Measure these first" in result.stdout


def test_script_needs_a_brake_size_for_a_rocket_without_a_device() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--rocket", str(FREEFALL_ROCKET), "--draws", "50"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 2 and "Traceback" not in result.stderr
    assert "--terminal-speed or --drag-area-cm2" in result.stderr


def test_script_checks_the_free_fall_with_a_zero_brake_area() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--rocket", str(FREEFALL_ROCKET), "--drag-area-cm2", "0", "--motor",
         str(FREEFALL_MOTOR), "--draws", "100"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    )
    assert "no brake" in result.stdout and "= terminal speed;" in result.stdout
    assert "hard part 46.0 N" in result.stdout and "lands from -3 % to +0 %" in result.stdout
