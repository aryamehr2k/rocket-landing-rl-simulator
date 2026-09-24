"""Six degree of freedom rigid body dynamics of the rocket, integrated with RK4.

This module has no RL or Gymnasium imports so it can be tested on its own.
Frames, signs and the state layout are defined in docs/conventions.md.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from rocketsim import quaternion
from rocketsim.aero import aerodynamic_loads, air_density
from rocketsim.config import RocketConfig
from rocketsim.dragdevice import device_loads
from rocketsim.simconfig import EnvironmentConfig
from rocketsim.touchdown import TouchdownResult, grade_touchdown, lowest_point, tip_over_angle

IX, IY, IZ, IVX, IVY, IVZ = range(6)
IPOS = slice(0, 3)
IVEL = slice(3, 6)
IQ = slice(6, 10)
IW = slice(10, 13)
IBURNED = 13
NUM_RIGID_STATES = 13
DOWN = np.array([0.0, 0.0, -1.0])
RK4_WEIGHTS = (1.0, 2.0, 2.0, 1.0)
HALF = 0.5


def state_size(num_motors: int) -> int:
    return NUM_RIGID_STATES + num_motors


def thrust_direction(gimbal_pitch: float, gimbal_yaw: float) -> np.ndarray:
    """Unit thrust direction in the body frame for the two gimbal deflections."""
    direction = np.array([-math.tan(gimbal_pitch), -math.tan(gimbal_yaw), 1.0])
    return direction / np.linalg.norm(direction)


@dataclass(frozen=True)
class MassProperties:
    mass: float
    cg: float
    pitch_inertia: float
    roll_inertia: float


@dataclass
class MotorInputs:
    """What the outside world tells the physics about one motor."""

    ignition_time: float | None = None
    throttle: float = 1.0
    thrust_scale: float = 1.0


@dataclass
class Inputs:
    """Everything the dynamics need from actuators and weather for one step."""

    gimbal_pitch: float = 0.0
    gimbal_yaw: float = 0.0
    brake_fraction: float = 0.0
    wind: np.ndarray = field(default_factory=lambda: np.zeros(2))
    motors: list[MotorInputs] = field(default_factory=list)

    def ignite(self, index: int, time: float) -> bool:
        """Start a motor's burn at `time`. A motor that already burned is not restarted."""
        if self.motors[index].ignition_time is not None:
            return False
        self.motors[index].ignition_time = time
        return True


class RocketDynamics:
    """Forces, moments, mass properties and integration for one rocket configuration."""

    def __init__(self, rocket: RocketConfig, environment: EnvironmentConfig) -> None:
        self.rocket = rocket
        self.environment = environment
        self.motors = [motor.spec for motor in rocket.motors]
        self.gimbaled = [motor.gimbaled for motor in rocket.motors]
        self.stations = np.array([motor.position for motor in rocket.motors])
        self.motor_radii = np.array([motor.spec.diameter * HALF for motor in rocket.motors])
        self.area = rocket.reference_area
        self.device = rocket.drag_device

    def new_inputs(self) -> Inputs:
        return Inputs(motors=[MotorInputs() for _ in self.motors])

    def initial_state(self) -> np.ndarray:
        """Standing upright on the pad with full motors and the feet on the ground."""
        y = np.zeros(state_size(len(self.motors)))
        y[IQ] = quaternion.IDENTITY
        y[IZ] = self.rocket.feet_station - self.mass_properties(y).cg
        return y

    def mass_properties(self, y: np.ndarray) -> MassProperties:
        """Mass, centre of gravity and inertias from the parts and burned propellant."""
        airframe = self.rocket.airframe
        burned = np.clip(y[IBURNED:], 0.0, [m.propellant_mass for m in self.motors])
        motor_masses = np.array([m.total_mass for m in self.motors]) - burned
        mass = airframe.dry_mass + float(motor_masses.sum())
        cg = (airframe.dry_mass * airframe.dry_cg + float((motor_masses * self.stations).sum())) / mass
        pitch = airframe.dry_pitch_inertia + airframe.dry_mass * (cg - airframe.dry_cg) ** 2
        pitch += float((motor_masses * (cg - self.stations) ** 2).sum())
        roll = airframe.dry_roll_inertia + float((HALF * motor_masses * self.motor_radii ** 2).sum())
        return MassProperties(mass, cg, pitch, roll)

    def motor_thrust(self, index: int, t: float, y: np.ndarray, inputs: Inputs) -> float:
        """Actual thrust of one motor including throttle and the random thrust scale."""
        spec, command = self.motors[index], inputs.motors[index]
        if command.ignition_time is None:
            return 0.0
        thrust = spec.thrust_at(t - command.ignition_time)
        if spec.throttleable:
            if y[IBURNED + index] >= spec.propellant_mass:
                return 0.0
            thrust *= command.throttle
        return thrust * command.thrust_scale

    def derivatives(self, t: float, y: np.ndarray, inputs: Inputs) -> np.ndarray:
        """Time derivative of the state vector."""
        props = self.mass_properties(y)
        q, omega = y[IQ], y[IW]
        rotation = quaternion.to_matrix(q)
        wind = np.array([inputs.wind[0], inputs.wind[1], 0.0])
        airspeed_body = rotation.T @ (y[IVEL] - wind)
        density = air_density(y[IZ], self.environment)
        loads = aerodynamic_loads(self.rocket.aero, self.area, density, airspeed_body, props.cg)
        force, moment = loads.force.copy(), loads.moment.copy()
        if self.device is not None and inputs.brake_fraction > 0.0:
            device_force, device_moment = device_loads(
                self.device, inputs.brake_fraction, density, airspeed_body, omega, props.cg
            )
            force += device_force
            moment += device_moment

        pivot = np.array([0.0, 0.0, props.cg - self.rocket.gimbal.pivot])
        direction = thrust_direction(inputs.gimbal_pitch, inputs.gimbal_yaw)
        dy = np.zeros_like(y)
        for i in range(len(self.motors)):
            thrust = self.motor_thrust(i, t, y, inputs)
            if self.gimbaled[i]:
                force += thrust * direction
                moment += np.cross(pivot, thrust * direction)
            else:
                force += thrust * quaternion.BODY_AXIS
            dy[IBURNED + i] = self._mass_flow(i, t, y, inputs)

        inertia = np.array([props.pitch_inertia, props.pitch_inertia, props.roll_inertia])
        dy[IPOS] = y[IVEL]
        dy[IVEL] = rotation @ force / props.mass + self.environment.gravity * DOWN
        dy[IQ] = quaternion.derivative(q, omega)
        dy[IW] = (moment - np.cross(omega, inertia * omega)) / inertia
        return dy

    def device_drag(self, y: np.ndarray, inputs: Inputs) -> float:
        """Magnitude of the drag device force in the current state, for the log."""
        if self.device is None or inputs.brake_fraction <= 0.0:
            return 0.0
        rotation = quaternion.to_matrix(y[IQ])
        wind = np.array([inputs.wind[0], inputs.wind[1], 0.0])
        airspeed_body = rotation.T @ (y[IVEL] - wind)
        density = air_density(y[IZ], self.environment)
        force, _ = device_loads(self.device, inputs.brake_fraction, density, airspeed_body, y[IW], self.mass_properties(y).cg)
        return float(np.linalg.norm(force))

    def _mass_flow(self, index: int, t: float, y: np.ndarray, inputs: Inputs) -> float:
        # Follows the thrust curve; the random thrust scale does not change it.
        spec, command = self.motors[index], inputs.motors[index]
        if command.ignition_time is None:
            return 0.0
        curve_thrust = spec.thrust_at(t - command.ignition_time)
        if spec.throttleable:
            if y[IBURNED + index] >= spec.propellant_mass:
                return 0.0
            curve_thrust *= command.throttle
        return curve_thrust / spec.exhaust_velocity

    def step(self, t: float, y: np.ndarray, inputs: Inputs, dt: float) -> np.ndarray:
        """One RK4 step from t to t + dt with the inputs held constant."""
        k1 = self.derivatives(t, y, inputs)
        k2 = self.derivatives(t + HALF * dt, y + HALF * dt * k1, inputs)
        k3 = self.derivatives(t + HALF * dt, y + HALF * dt * k2, inputs)
        k4 = self.derivatives(t + dt, y + dt * k3, inputs)
        w = RK4_WEIGHTS
        y_next = y + dt / sum(w) * (w[0] * k1 + w[1] * k2 + w[2] * k3 + w[3] * k4)
        y_next[IQ] = quaternion.normalize(y_next[IQ])
        return y_next

    def lowest_point(self, y: np.ndarray) -> float:
        return lowest_point(self.rocket, self.mass_properties(y).cg, y[IPOS], y[IQ])

    def tip_over_angle(self, y: np.ndarray) -> float:
        return tip_over_angle(self.rocket, self.mass_properties(y).cg)

    def grade_touchdown(self, t: float, y: np.ndarray) -> TouchdownResult:
        return grade_touchdown(self.rocket, self.mass_properties(y).cg, t, y[IPOS], y[IVEL], y[IQ])


class Flight:
    """Advances one flight: holds the rocket on the pad until liftoff, then flies to touchdown."""

    def __init__(self, dynamics: RocketDynamics, dt: float) -> None:
        self.dynamics = dynamics
        self.dt = dt
        self.t = 0.0
        self.y = dynamics.initial_state()
        self.inputs = dynamics.new_inputs()
        self.lifted_off = False
        self.feet_clear = False
        self.touchdown: TouchdownResult | None = None
        self.apogee = self.y[IZ]

    @property
    def done(self) -> bool:
        return self.touchdown is not None

    def step(self) -> None:
        """Advance one physics step using the current inputs."""
        if self.done:
            return
        if not self.lifted_off:
            dy = self.dynamics.derivatives(self.t, self.y, self.inputs)
            if dy[IVZ] <= 0.0:
                # Held by the pad: only the propellant changes, and the feet stay on the ground
                # while the centre of gravity moves toward the nose.
                self.y[IBURNED:] += self.dt * dy[IBURNED:]
                self.y[IZ] = self.dynamics.rocket.feet_station - self.dynamics.mass_properties(self.y).cg
                self.t += self.dt
                return
            self.lifted_off = True
        self.y = self.dynamics.step(self.t, self.y, self.inputs, self.dt)
        self.t += self.dt
        self.apogee = max(self.apogee, self.y[IZ])
        lowest = self.dynamics.lowest_point(self.y)
        # Touchdown only counts once the feet have actually been off the ground.
        if lowest > 0.0:
            self.feet_clear = True
        elif self.feet_clear:
            self.touchdown = self.dynamics.grade_touchdown(self.t, self.y)
