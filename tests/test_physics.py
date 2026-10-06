import dataclasses
import math

import numpy as np
import pytest

from rocketsim import quaternion as quat
from rocketsim.aero import Wind, air_density
from rocketsim.config import RocketConfig, load_rocket_config
from rocketsim.physics import IBURNED, IPOS, IQ, IVEL, IVX, IVY, IVZ, IW, IX, IY, IZ, Flight, RocketDynamics
from rocketsim.simconfig import EnvironmentConfig, WindConfig, load_sim_config
from tests.conftest import EXAMPLE_ROCKET, EXAMPLE_SIM, ROTATE_X, ROTATE_Y, make_environment, make_rocket


def test_mass_properties_by_hand(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    dynamics = RocketDynamics(rocket, environment)
    y = dynamics.initial_state()
    props = dynamics.mass_properties(y)
    assert props.mass == pytest.approx(1.3)
    cg = 0.76 / 1.3  # (1.0*0.5 + 0.2*0.9 + 0.1*0.8) / (1.0 + 0.2 + 0.1)
    assert props.cg == pytest.approx(cg)
    expected = 0.1 + 1.0 * (cg - 0.5) ** 2 + 0.2 * (cg - 0.9) ** 2 + 0.1 * (cg - 0.8) ** 2
    assert props.pitch_inertia == pytest.approx(expected)
    assert props.roll_inertia == pytest.approx(0.002)  # the test motors have no diameter: the dry value
    assert y[IZ] == pytest.approx(1.1 - cg)  # feet plane 1.0 + 0.1 from the nose
    assert list(y[IQ]) == [1.0, 0.0, 0.0, 0.0]
    y[IBURNED] = 0.1
    assert dynamics.mass_properties(y).mass == pytest.approx(1.2)
    assert dynamics.mass_properties(y).cg < cg


def test_derivatives_by_hand_pitch_plane(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    """Numbers worked out step by step from docs/conventions.md, not from the code."""
    dynamics = RocketDynamics(rocket, environment)
    y = dynamics.initial_state()
    y[IZ], y[IVX], y[IVZ] = 100.0, 2.0, 30.0
    y[IQ] = quat.from_axis_angle(ROTATE_Y, 0.1)
    inputs = dynamics.new_inputs()
    inputs.gimbal_pitch = 0.05
    inputs.wind = np.array([1.0, 0.0])
    inputs.ignite(0, 0.0)
    dy = dynamics.derivatives(1.0, y, inputs)
    # Airspeed in body axes: b_z = 1*sin(0.1) + 30*cos(0.1) = 29.949958, b_x = 1*cos(0.1) - 30*sin(0.1) = -1.999998
    # speed = sqrt(901) = 30.016662, factor = 0.5 * 1.0 * (pi*0.01/4) * speed = 0.117875
    # drag = -factor*0.5*(b_x, b_z) = (+0.117875, -1.765178); normal_x = -factor*2*b_x = +0.471500
    # aero moment about y = (cg - cp) * normal_x = (0.584615 - 0.7) * 0.471500 = -0.054404
    # thrust 20 N at gimbal 0.05: body (-0.999583, 0, 19.975003), moment_y = 20*(0.95-0.584615)*sin(0.05) = 0.365233
    # totals: F_bx = -0.410208, F_bz = 18.209827, M_y = 0.310829
    # world: Fx = F_bz*sin + F_bx*cos = 1.409790, Fz = F_bz*cos - F_bx*sin = 18.159806
    assert dy[IPOS] == pytest.approx([2.0, 0.0, 30.0])
    assert dy[IVX] == pytest.approx(1.084454, abs=1e-5)
    assert dy[IVY] == 0.0
    assert dy[IVZ] == pytest.approx(3.969082, abs=1e-5)
    assert dy[IW] == pytest.approx([0.0, 2.360263, 0.0], abs=1e-5)
    assert dy[IBURNED] == pytest.approx(20.0 / 400.0)
    assert dy[IBURNED + 1] == 0.0


def test_yaw_plane_mirrors_pitch_plane(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    dynamics = RocketDynamics(rocket, environment)
    y = dynamics.initial_state()
    y[IZ], y[IVY], y[IVZ] = 100.0, 2.0, 30.0
    y[IQ] = quat.from_axis_angle(ROTATE_X, -0.1)
    assert quat.tilt_angles(y[IQ]) == pytest.approx((0.0, 0.1))
    inputs = dynamics.new_inputs()
    inputs.gimbal_yaw = 0.05
    inputs.wind = np.array([0.0, 1.0])
    inputs.ignite(0, 0.0)
    dy = dynamics.derivatives(1.0, y, inputs)
    assert dy[IVY] == pytest.approx(1.084454, abs=1e-5)
    assert dy[IVX] == 0.0
    assert dy[IVZ] == pytest.approx(3.969082, abs=1e-5)
    assert dy[IW] == pytest.approx([-2.360263, 0.0, 0.0], abs=1e-5)  # nose toward +y: negative about x
    assert quat.tilt_rates(y[IQ], dy[IW])[1] > 0.0


def test_positive_gimbal_gives_positive_tilt_acceleration(
    rocket: RocketConfig, environment: EnvironmentConfig
) -> None:
    dynamics = RocketDynamics(rocket, environment)
    y = dynamics.initial_state()
    inputs = dynamics.new_inputs()
    inputs.ignite(0, 0.0)
    for pitch, yaw in ((0.1, 0.0), (0.0, 0.1)):
        inputs.gimbal_pitch, inputs.gimbal_yaw = pitch, yaw
        dy = dynamics.derivatives(0.5, y, inputs)
        rates = quat.tilt_rates(y[IQ], dy[IW])
        assert rates[0] > 0.0 if pitch else rates[1] > 0.0
        assert dy[IVX] < 0.0 if pitch else dy[IVY] < 0.0
        assert dy[IW][2] == 0.0


def test_fins_restore_when_cp_is_behind_cg(environment: EnvironmentConfig) -> None:
    dynamics = RocketDynamics(make_rocket(cp=0.9), environment)
    y = dynamics.initial_state()
    y[IZ], y[IVZ] = 50.0, 30.0
    for angle in (0.2, -0.2):
        y[IQ] = quat.from_axis_angle(ROTATE_Y, angle)
        dy = dynamics.derivatives(0.0, y, dynamics.new_inputs())
        assert math.copysign(1.0, dy[IW][1]) == -math.copysign(1.0, angle)


def test_one_rk4_step_by_hand_in_vacuum() -> None:
    """Ballistic flight with a constant body spin: every exact value is known."""
    dynamics = RocketDynamics(make_rocket(drag_coefficient=0.0, normal_force_slope=0.0), make_environment())
    y = dynamics.initial_state()
    y[IPOS], y[IVEL] = [1.0, 2.0, 50.0], [3.0, -1.0, 4.0]
    y[IQ] = quat.from_axis_angle(ROTATE_Y, 0.2)
    y[IW] = [0.5, 0.0, 0.0]
    dt = 0.005
    y_next = dynamics.step(0.0, y, dynamics.new_inputs(), dt)
    assert y_next[IPOS] == pytest.approx([1.0 + 3.0 * dt, 2.0 - 1.0 * dt, 50.0 + 4.0 * dt - 0.5 * 10.0 * dt ** 2])
    assert y_next[IVEL] == pytest.approx([3.0, -1.0, 4.0 - 10.0 * dt])
    expected_q = quat.multiply(y[IQ], quat.from_axis_angle(ROTATE_X, 0.5 * dt))
    assert y_next[IQ] == pytest.approx(expected_q, abs=1e-12)
    assert y_next[IW] == pytest.approx([0.5, 0.0, 0.0])


def test_free_fall_matches_analytic_solution() -> None:
    dynamics = RocketDynamics(make_rocket(drag_coefficient=0.0, normal_force_slope=0.0), make_environment())
    y = dynamics.initial_state()
    y[IZ] = 100.0
    inputs = dynamics.new_inputs()
    dt, t = 0.005, 0.0
    for _ in range(200):
        y = dynamics.step(t, y, inputs, dt)
        t += dt
    assert y[IZ] == pytest.approx(100.0 - 0.5 * 10.0 * t ** 2, abs=1e-9)
    assert y[IVZ] == pytest.approx(-10.0 * t, abs=1e-9)


def test_energy_and_momentum_conserved_in_vacuum() -> None:
    dynamics = RocketDynamics(make_rocket(drag_coefficient=0.0, normal_force_slope=0.0), make_environment())
    y = dynamics.initial_state()
    y[IZ], y[IVEL], y[IW] = 200.0, [3.0, -2.0, 20.0], [2.0, 0.5, 3.0]
    props = dynamics.mass_properties(y)
    inertia = np.array([props.pitch_inertia, props.pitch_inertia, props.roll_inertia])

    def energy(state: np.ndarray) -> float:
        kinetic = 0.5 * props.mass * float(state[IVEL] @ state[IVEL])
        rotational = 0.5 * float(state[IW] @ (inertia * state[IW]))
        return kinetic + rotational + props.mass * 10.0 * state[IZ]

    def angular_momentum(state: np.ndarray) -> np.ndarray:
        return quat.rotate(state[IQ], inertia * state[IW])

    start_energy, start_momentum = energy(y), angular_momentum(y)
    inputs = dynamics.new_inputs()
    for step in range(600):
        y = dynamics.step(step * 0.005, y, inputs, 0.005)
    assert energy(y) == pytest.approx(start_energy, rel=1e-9)
    assert angular_momentum(y) == pytest.approx(start_momentum, rel=1e-7)
    assert y[IVX] == pytest.approx(3.0)
    assert y[IVY] == pytest.approx(-2.0)
    assert np.linalg.norm(y[IQ]) == pytest.approx(1.0)


def test_drag_only_removes_energy(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    dynamics = RocketDynamics(rocket, environment)
    y = dynamics.initial_state()
    y[IZ], y[IVZ] = 200.0, 40.0
    inputs = dynamics.new_inputs()
    mass = dynamics.mass_properties(y).mass
    previous = 0.5 * mass * y[IVZ] ** 2 + mass * 10.0 * y[IZ]
    for step in range(400):
        y = dynamics.step(step * 0.005, y, inputs, 0.005)
        current = 0.5 * mass * float(y[IVEL] @ y[IVEL]) + mass * 10.0 * y[IZ]
        assert current < previous
        previous = current


def test_vertical_burn_with_zero_gimbal_stays_vertical(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    flight = Flight(RocketDynamics(rocket, environment), 0.005)
    flight.inputs.ignite(0, 0.0)
    while not flight.done and flight.t < 20.0:
        flight.step()
    assert flight.lifted_off and flight.done
    assert flight.apogee > 1.1
    assert flight.y[IX] == 0.0 and flight.y[IY] == 0.0
    assert list(flight.y[IQ]) == [1.0, 0.0, 0.0, 0.0]
    assert list(flight.y[IW]) == [0.0, 0.0, 0.0]
    assert flight.touchdown is not None
    assert flight.touchdown.miss_distance == 0.0 and flight.touchdown.tilt == 0.0
    assert flight.y[IBURNED] == pytest.approx(0.1, abs=1e-3)  # RK4 overshoots the burnout step a little
    assert flight.dynamics.mass_properties(flight.y).mass == pytest.approx(1.2)  # clipped


def test_roll_rate_is_untouched_by_thrust(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    flight = Flight(RocketDynamics(rocket, environment), 0.005)
    flight.inputs.ignite(0, 0.0)
    flight.step()
    flight.y[IW] = [0.0, 0.0, 1.0]
    for _ in range(200):
        flight.step()
    assert flight.y[IW] == pytest.approx([0.0, 0.0, 1.0])
    assert quat.tilt_angles(flight.y[IQ]) == pytest.approx((0.0, 0.0), abs=1e-12)


def test_solid_motor_cannot_be_reignited(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    inputs = RocketDynamics(rocket, environment).new_inputs()
    assert inputs.ignite(0, 1.0)
    assert not inputs.ignite(0, 5.0)
    assert inputs.motors[0].ignition_time == 1.0


def test_liftoff_is_never_graded_as_a_landing(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    """The CG moving noseward as propellant burns must not sink the feet below the pad, at any rate or gimbal."""
    for dt, gimbal in ((0.001, 0.0), (0.005, math.radians(10.0))):
        flight = Flight(RocketDynamics(rocket, environment), dt)
        flight.inputs.gimbal_pitch = gimbal
        flight.inputs.ignite(0, 0.0)
        for _ in range(200):
            flight.step()
        assert flight.lifted_off and not flight.done
        while not flight.done and flight.t < 30.0:
            flight.step()
        assert flight.touchdown is not None and not flight.touchdown.success


def test_pad_hold_keeps_feet_on_the_ground(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    heavy = make_rocket(dry_mass=3.0)
    flight = Flight(RocketDynamics(heavy, environment), 0.005)
    flight.inputs.ignite(0, 0.0)
    for _ in range(100):
        flight.step()
        assert flight.dynamics.lowest_point(flight.y) == pytest.approx(0.0, abs=1e-12)
    assert flight.y[IZ] > flight.dynamics.initial_state()[IZ]


def test_pad_hold_until_thrust_exceeds_weight(environment: EnvironmentConfig) -> None:
    heavy = make_rocket(dry_mass=3.0)  # 20 N cannot lift 3 kg at 10 m/s^2
    flight = Flight(RocketDynamics(heavy, environment), 0.005)
    flight.inputs.ignite(0, 0.0)
    z0 = flight.y[IZ]
    for _ in range(100):
        flight.step()
    assert not flight.lifted_off
    assert flight.y[IZ] >= z0  # the CG creeps toward the nose as propellant burns
    assert flight.dynamics.lowest_point(flight.y) == pytest.approx(0.0, abs=1e-12)
    assert flight.y[IBURNED] > 0.0


def test_touchdown_geometry_and_tip_over(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    dynamics = RocketDynamics(rocket, environment)
    y = dynamics.initial_state()
    assert dynamics.lowest_point(y) == pytest.approx(0.0)
    cg_height = 1.1 - 0.76 / 1.3
    # Four feet: the inscribed radius of the square is (span/2) * cos(45 deg).
    assert dynamics.tip_over_angle(y) == pytest.approx(math.atan2(0.15 * math.cos(math.pi / 4), cg_height))
    # Leaning toward +x lowers the foot on the +x side by (span/2) sin(theta) plus the cosine drop.
    y[IQ] = quat.from_axis_angle(ROTATE_Y, 0.2)
    expected = y[IZ] - cg_height * math.cos(0.2) - 0.15 * math.sin(0.2)
    assert dynamics.lowest_point(y) == pytest.approx(expected)
    y[IQ] = quat.from_axis_angle(ROTATE_Y, math.pi)  # upside down the nose is the lowest point
    assert dynamics.lowest_point(y) == pytest.approx(y[IZ] - 0.76 / 1.3)


def test_touchdown_grading(rocket: RocketConfig, environment: EnvironmentConfig) -> None:
    dynamics = RocketDynamics(rocket, environment)
    y = dynamics.initial_state()
    y[IX], y[IY], y[IVEL] = 3.0, 4.0, [0.3, 0.4, -1.0]
    y[IQ] = quat.from_axis_angle(ROTATE_X, 0.05)
    result = dynamics.grade_touchdown(5.0, y)
    assert result.success
    assert result.miss_distance == pytest.approx(5.0)
    assert result.lateral_speed == pytest.approx(0.5)
    assert result.tilt_limit == pytest.approx(math.radians(10.0))
    y[IVEL] = [1.5, 0.0, -3.0]
    y[IQ] = quat.from_axis_angle(ROTATE_X, 0.3)
    result = dynamics.grade_touchdown(5.0, y)
    assert result.failures == ("vertical speed", "lateral speed", "tilt")


def test_air_density_and_wind() -> None:
    env = load_sim_config(EXAMPLE_SIM).environment
    assert air_density(0.0, env) == pytest.approx(1.225 * math.exp(-300.0 / 8500.0))
    assert air_density(1000.0, env) < air_density(0.0, env)
    steady = Wind(WindConfig(steady=2.0, steady_direction=math.pi / 2, gust_std=0.0, gust_time_constant=1.0))
    assert steady.step(0.005) == pytest.approx([0.0, 2.0])
    gusty = Wind(WindConfig(0.0, 0.0, gust_std=1.0, gust_time_constant=0.5), np.random.default_rng(0))
    samples = np.array([gusty.step(0.005) for _ in range(40000)])
    assert samples.std(axis=0) == pytest.approx([1.0, 1.0], rel=0.15)


def test_example_rocket_open_loop_flight_is_plausible() -> None:
    rocket = load_rocket_config(EXAMPLE_ROCKET)
    sim = load_sim_config(EXAMPLE_SIM)
    flight = Flight(RocketDynamics(rocket, sim.environment), sim.simulation.dt)
    flight.inputs.ignite(0, rocket.motors[0].ignition_delay_mean)
    while not flight.done and flight.t < sim.simulation.max_flight_time:
        flight.step()
    assert flight.done
    assert 60.0 < flight.apogee < 200.0
    assert flight.touchdown is not None and not flight.touchdown.success


def test_replace_keeps_config_immutable(rocket: RocketConfig) -> None:
    changed = dataclasses.replace(rocket, name="other")
    assert rocket.name == "test" and changed.name == "other"
