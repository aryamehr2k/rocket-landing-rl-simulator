"""The closed loop: sensors, flight computer, actuators, physics and the flight log.

One `control_step` runs the flight computer once and the physics `steps_per_control` times.
The scripted flight and the Gymnasium environment both drive it this way.
"""

from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from rocketsim import quaternion
from rocketsim.actuators import Igniter, Servo
from rocketsim.aero import Wind
from rocketsim.commands import ControlCommand, PlaneAction
from rocketsim.config import RocketConfig
from rocketsim.flightcomputer import FlightComputer
from rocketsim.flightlog import FlightLogWriter
from rocketsim.physics import IBURNED, IPOS, IQ, IVEL, IW, IZ, Flight, RocketDynamics
from rocketsim.sensors import SensorSuite, Truth
from rocketsim.simconfig import SimConfig, WindConfig, physics_steps_per_control_step

ASCENT, LANDING = 0, 1
STATE_NAMES = ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "qw", "qx", "qy", "qz", "wx_radps", "wy_radps", "wz_radps")
SENSOR_AXES = ("bx", "by", "bz")


class Simulation:
    def __init__(
        self,
        rocket: RocketConfig,
        sim: SimConfig,
        seed: int | None = None,
        log_path: str | Path | None = None,
        wind: WindConfig | None = None,
    ) -> None:
        self.rocket = rocket
        self.sim = sim
        self.dt = sim.simulation.dt
        self.steps_per_control = physics_steps_per_control_step(sim.simulation, rocket.control)
        self.control_dt = self.dt * self.steps_per_control
        self.rng = np.random.default_rng(seed)
        self.dynamics = RocketDynamics(rocket, sim.environment)
        self.gravity_down = np.array([0.0, 0.0, -sim.environment.gravity])
        self.wind = Wind(wind if wind is not None else sim.wind, self.rng)
        self.sensors = SensorSuite(rocket.computer.sensors, self.dt, self.rng)
        self.flight_computer = FlightComputer(rocket, sim.environment, self.control_dt, sim.simulation.pad_hold_time)
        self.servos = (Servo(rocket.gimbal.pitch_servo, rocket.gimbal, self.dt), Servo(rocket.gimbal.yaw_servo, rocket.gimbal, self.dt))
        self.igniters = tuple(Igniter(i, motor, self.rng) for i, motor in enumerate(rocket.motors))
        self.log = FlightLogWriter(log_path) if log_path is not None else None
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Start a new flight. The same seed repeats the sensor errors, igniter delays and gusts exactly."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.wind.rng = self.sensors.rng = self.rng
            for igniter in self.igniters:
                igniter.rng = self.rng
        self.flight = Flight(self.dynamics, self.dt)
        self.wind.reset()
        self.sensors.reset()
        self.flight_computer.reset()
        for servo in self.servos:
            servo.reset()
        for igniter in self.igniters:
            igniter.reset()
        self.applied = ControlCommand()
        self.pending: deque[ControlCommand] = deque([ControlCommand()] * self.rocket.control.action_delay_steps)

    @property
    def t(self) -> float:
        return self.flight.t

    @property
    def done(self) -> bool:
        return self.flight.done or self.t >= self.sim.simulation.max_flight_time

    def control_step(self, action: PlaneAction | None = None) -> ControlCommand:
        """Run the flight computer once, then the physics for one control period."""
        command = ControlCommand()
        for i in range(self.steps_per_control):
            self._sense()
            if i == 0:
                command = self.flight_computer.control_step(self.t, action)
                self.pending.append(command)
                self._apply(self.pending.popleft())
            self._log_row()
            self._advance()
            if self.flight.done:
                self.flight_computer.landed(self.t)
                self._sense()
                self._log_row()
                break
        return command

    def run(self) -> None:
        """Fly with the flight computer's own PID and trigger until touchdown or the time limit."""
        while not self.done:
            self.control_step()

    def close(self) -> None:
        if self.log is not None:
            self.log.close()

    def _sense(self) -> None:
        y, t = self.flight.y, self.t
        rotation = quaternion.to_matrix(y[IQ])
        accel_world = np.zeros(3)
        if self.flight.lifted_off and not self.flight.done:
            accel_world = self.dynamics.derivatives(t, y, self.flight.inputs)[IVEL]
        specific_force = rotation.T @ (accel_world - self.gravity_down)
        self.sensors.record(Truth(specific_force, y[IW].copy(), float(y[IZ])))
        imu, baro = self.sensors.sample(t)
        if imu is not None:
            self.flight_computer.imu(imu)
        if baro is not None:
            self.flight_computer.baro(baro)

    def _apply(self, command: ControlCommand) -> None:
        self.applied = command
        self.servos[0].command(command.gimbal_pitch)
        self.servos[1].command(command.gimbal_yaw)
        if command.ignite_ascent:
            self.igniters[ASCENT].command(self.t, self.flight.inputs)
        if command.ignite_landing:
            self.igniters[LANDING].command(self.t, self.flight.inputs)

    def _advance(self) -> None:
        inputs = self.flight.inputs
        inputs.wind = self.wind.step(self.dt)
        inputs.gimbal_pitch = self.servos[0].step()
        inputs.gimbal_yaw = self.servos[1].step()
        self.flight.step()

    def _log_row(self) -> None:
        if self.log is None:
            return
        y, t, inputs = self.flight.y, self.t, self.flight.inputs
        row: dict[str, float | str] = {"time_s": t, "phase": self.flight_computer.phase.value}
        imu, baro = self.sensors.last_imu, self.sensors.last_baro
        if imu is not None:
            for axis, force, rate in zip(SENSOR_AXES, imu.specific_force, imu.angular_rate):
                row[f"accel_{axis}_mps2"] = force
                row[f"gyro_{axis}_radps"] = rate
        if baro is not None:
            row["baro_alt_m"] = baro.altitude
        estimate = self.flight_computer.estimate
        est_values = list(estimate.position) + list(estimate.velocity) + list(estimate.quaternion) + list(estimate.angular_rate)
        for name, est, true in zip(STATE_NAMES, est_values, list(y[IPOS]) + list(y[IVEL]) + list(y[IQ]) + list(y[IW])):
            row[f"est_{name}"] = est
            row[f"true_{name}"] = true
        row.update(
            {
                "true_mass_kg": self.dynamics.mass_properties(y).mass,
                "gimbal_pitch_cmd_rad": self.applied.gimbal_pitch,
                "gimbal_yaw_cmd_rad": self.applied.gimbal_yaw,
                "gimbal_pitch_act_rad": self.servos[0].angle,
                "gimbal_yaw_act_rad": self.servos[1].angle,
                "servo_pitch_us": self.servos[0].pulse,
                "servo_yaw_us": self.servos[1].pulse,
                "ascent_ignite_cmd": float(self.igniters[ASCENT].commanded),
                "ascent_thrust_n": self.dynamics.motor_thrust(ASCENT, t, y, inputs),
                "landing_ignite_cmd": float(self.igniters[LANDING].commanded),
                "landing_thrust_n": self.dynamics.motor_thrust(LANDING, t, y, inputs),
                "wind_x_mps": inputs.wind[0],
                "wind_y_mps": inputs.wind[1],
            }
        )
        self.log.write(row)


def with_wind(sim: SimConfig, steady: float | None, direction: float | None, gust_std: float | None) -> WindConfig:
    """The training YAML's wind with any of the three main values replaced."""
    wind = sim.wind
    if steady is not None:
        wind = replace(wind, steady=steady)
    if direction is not None:
        wind = replace(wind, steady_direction=direction)
    if gust_std is not None:
        wind = replace(wind, gust_std=gust_std)
    return wind
