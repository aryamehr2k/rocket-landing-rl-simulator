"""Closed loop of the electric vehicle: sensors, flight computer, servos, throttle, roll control, physics.

As in rocketsim.simulation, one `control_step` runs the flight computer once and the physics several times.
"""

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rocketsim import quaternion
from rocketsim.actuators import RollActuator, Servo, Throttle
from rocketsim.aero import Wind
from rocketsim.commands import PlaneAction
from rocketsim.flightlog import FlightLogWriter
from rocketsim.hop.computer import HopCommand, HopFlightComputer
from rocketsim.hop.mission import HopPhase, MissionConfig, MissionResult, MissionScore
from rocketsim.hop.vehicle import HopVehicleConfig
from rocketsim.physics import IPOS, IQ, IVEL, IW, IZ, Flight, RocketDynamics
from rocketsim.sensors import SensorSuite, Truth
from rocketsim.simconfig import SimConfig, WindConfig, physics_steps_per_control_step

MAIN = 0
STATE_NAMES = ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "qw", "qx", "qy", "qz", "wx_radps", "wy_radps", "wz_radps")
SENSOR_AXES = ("bx", "by", "bz")


@dataclass(frozen=True)
class HopErrors:
    """What the real vehicle gets wrong compared with its file, drawn per flight in training."""

    thrust_scale: float = 1.0
    dry_mass_offset: float = 0.0


class HopSimulation:
    def __init__(
        self, vehicle: HopVehicleConfig, mission: MissionConfig, sim: SimConfig, seed: int | None = None,
        log_path: str | Path | None = None, wind: WindConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.mission = mission
        self.sim = sim
        self.dt = sim.simulation.dt
        self.steps_per_control = physics_steps_per_control_step(sim.simulation, vehicle.body.control)
        self.control_dt = self.dt * self.steps_per_control
        self.rng = np.random.default_rng(seed)
        self.wind = Wind(wind if wind is not None else sim.wind, self.rng)
        self.sensors = SensorSuite(vehicle.sensors, self.dt, self.rng)
        self.computer = HopFlightComputer(vehicle, mission, sim.environment, self.control_dt, sim.simulation.pad_hold_time)
        gimbal = vehicle.body.gimbal
        self.servos = (Servo(gimbal.pitch_servo, gimbal, self.dt), Servo(gimbal.yaw_servo, gimbal, self.dt))
        self.throttle = Throttle(vehicle.motor.spec.throttle_lag, self.dt)
        self.roll = RollActuator(vehicle.roll.max_torque, vehicle.roll.time_constant, self.dt)
        self.log = FlightLogWriter(log_path) if log_path is not None else None
        self.reset(seed)

    def reset(self, seed: int | None = None, mission: MissionConfig | None = None, errors: HopErrors | None = None) -> None:
        """Start a new flight. The same seed, mission and errors repeat the flight exactly."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.wind.rng = self.sensors.rng = self.rng
        if mission is not None:
            self.mission = mission
            self.computer.set_mission(mission)
        self.errors = errors if errors is not None else HopErrors()
        airframe = self.vehicle.body.airframe
        body = replace(self.vehicle.body, airframe=replace(airframe, dry_mass=airframe.dry_mass + self.errors.dry_mass_offset))
        self.dynamics = RocketDynamics(body, self.sim.environment)
        self.flight = Flight(self.dynamics, self.dt)
        self.flight.inputs.motors[MAIN].thrust_scale = self.errors.thrust_scale
        self.flight.inputs.motors[MAIN].throttle = 0.0
        self.wind.reset()
        self.sensors.reset()
        self.computer.reset()
        for actuator in (*self.servos, self.throttle, self.roll):
            actuator.reset()
        self.score = MissionScore(self.mission)
        self.applied = HopCommand()
        self.pending: deque[HopCommand] = deque([HopCommand()] * self.vehicle.body.control.action_delay_steps)
        self._sense()
        self.computer.prepare(self.t)

    @property
    def t(self) -> float:
        return self.flight.t

    @property
    def done(self) -> bool:
        limit = min(self.mission.max_flight_time, self.sim.simulation.max_flight_time)
        return self.flight.done or self.t >= limit

    def feet_height(self) -> float:
        return self.dynamics.lowest_point(self.flight.y)

    def set_wind(self, speed: float, direction: float, gust_std: float) -> None:
        """Change the wind during a flight (live dashboard)."""
        self.wind.config = replace(self.wind.config, steady=speed, steady_direction=direction, gust_std=gust_std)
        self.wind.steady = speed * np.array([np.cos(direction), np.sin(direction)])

    def push(self, delta_v: np.ndarray) -> None:
        """Add a sideways velocity kick in m/s (a gust or a hand on the tether)."""
        self.flight.y[IVEL][:2] += delta_v

    def control_step(self, action: PlaneAction | None = None) -> HopCommand:
        """Run the flight computer once, then the physics for one control period."""
        command = HopCommand()
        for i in range(self.steps_per_control):
            if i == 0:
                command = self.computer.control_step(self.t, action)
                self.pending.append(command)
                self._apply(self.pending.popleft())
            self._log_row()
            self._advance()
            # Sensing after the step means a policy evaluated between steps sees the same estimate as the PID.
            self._sense()
            if self.flight.done:
                self.computer.landed(self.t)
                self._log_row()
                break
        self.computer.prepare(self.t)
        return command

    def run(self) -> MissionResult:
        """Fly with the vehicle file's own controllers until touchdown or the time limit."""
        while not self.done:
            self.control_step()
        return self.result

    def close(self) -> None:
        if self.log is not None:
            self.log.close()

    @property
    def result(self) -> MissionResult:
        touchdown = self.flight.touchdown
        y = self.flight.y
        miss = touchdown.miss_distance if touchdown is not None else float(np.hypot(y[0], y[1]))
        return self.score.result(
            landed=bool(touchdown is not None and touchdown.success),
            miss_distance=miss,
            touchdown_speed=touchdown.vertical_speed if touchdown is not None else float("nan"),
            aborted=self.computer.aborted,
        )

    def frame(self) -> dict[str, Any]:
        """The current state for a live display."""
        y, inputs = self.flight.y, self.flight.inputs
        tilt = float(np.degrees(quaternion.total_tilt(y[IQ])))
        return {
            "t": round(self.t, 3), "phase": self.computer.phase.value,
            "position": [round(float(v), 3) for v in y[IPOS]], "velocity": [round(float(v), 3) for v in y[IVEL]],
            "quaternion": [round(float(v), 5) for v in y[IQ]], "height": round(self.feet_height(), 3),
            "est_height": round(self.computer.height, 3), "tilt_deg": round(tilt, 2),
            "gimbal_deg": [round(float(np.degrees(inputs.gimbal_pitch)), 2), round(float(np.degrees(inputs.gimbal_yaw)), 2)],
            "throttle": round(inputs.motors[MAIN].throttle, 3),
            "thrust": round(self.dynamics.motor_thrust(MAIN, self.t, y, inputs), 2),
            "ref_height": round(self.computer.reference[0], 3), "ref_speed": round(self.computer.reference[1], 3),
            "wind": [round(float(v), 2) for v in inputs.wind], "max_height": round(self.score.max_height, 2),
            "hover_held": round(self.score.best_hold, 2), "done": self.done,
        }

    def _apply(self, command: HopCommand) -> None:
        self.applied = command
        self.servos[0].command(command.gimbal_pitch)
        self.servos[1].command(command.gimbal_yaw)
        self.throttle.command(command.throttle)
        self.roll.command(command.roll_torque)
        if command.arm and self.flight.inputs.motors[MAIN].ignition_time is None:
            self.flight.inputs.ignite(MAIN, self.t)  # an electric motor starts at once

    def _advance(self) -> None:
        inputs = self.flight.inputs
        inputs.wind = self.wind.step(self.dt)
        inputs.gimbal_pitch = self.servos[0].step()
        inputs.gimbal_yaw = self.servos[1].step()
        inputs.motors[MAIN].throttle = self.throttle.step()
        inputs.roll_torque = self.roll.step()
        self.flight.step()
        self.score.update(self.t, self.feet_height())

    def _sense(self) -> None:
        y, t = self.flight.y, self.t
        rotation = quaternion.to_matrix(y[IQ])
        accel_world = np.zeros(3)
        if self.flight.lifted_off and not self.flight.done:
            accel_world = self.dynamics.derivatives(t, y, self.flight.inputs)[IVEL]
        gravity_down = np.array([0.0, 0.0, -self.sim.environment.gravity])
        self.sensors.record(Truth(
            rotation.T @ (accel_world - gravity_down), y[IW].copy(), float(y[IZ]), y[IPOS][:2].copy(), y[IVEL][:2].copy()
        ))
        imu, baro = self.sensors.sample(t)
        gps = self.sensors.sample_gps(t)
        if imu is not None:
            self.computer.imu(imu)
        if baro is not None:
            self.computer.baro(baro)
        if gps is not None:
            self.computer.gps(gps)

    def _log_row(self) -> None:
        if self.log is None:
            return
        y, t, inputs = self.flight.y, self.t, self.flight.inputs
        row: dict[str, float | str] = {"time_s": t, "phase": self.computer.phase.value}
        imu, baro = self.sensors.last_imu, self.sensors.last_baro
        if imu is not None:
            for axis, force, rate in zip(SENSOR_AXES, imu.specific_force, imu.angular_rate):
                row[f"accel_{axis}_mps2"] = force
                row[f"gyro_{axis}_radps"] = rate
        if baro is not None:
            row["baro_alt_m"] = baro.altitude
        est = self.computer.estimate
        est_values = list(est.position) + list(est.velocity) + list(est.quaternion) + list(est.angular_rate)
        for name, e, true in zip(STATE_NAMES, est_values, list(y[IPOS]) + list(y[IVEL]) + list(y[IQ]) + list(y[IW])):
            row[f"est_{name}"] = e
            row[f"true_{name}"] = true
        row.update({
            "true_mass_kg": self.dynamics.mass_properties(y).mass,
            "gimbal_pitch_cmd_rad": self.applied.gimbal_pitch, "gimbal_yaw_cmd_rad": self.applied.gimbal_yaw,
            "gimbal_pitch_act_rad": self.servos[0].angle, "gimbal_yaw_act_rad": self.servos[1].angle,
            "servo_pitch_us": self.servos[0].pulse, "servo_yaw_us": self.servos[1].pulse,
            "wind_x_mps": inputs.wind[0], "wind_y_mps": inputs.wind[1],
            "throttle_cmd": self.applied.throttle, "throttle_act": inputs.motors[MAIN].throttle,
            "main_thrust_n": self.dynamics.motor_thrust(MAIN, t, y, inputs), "roll_torque_nm": inputs.roll_torque,
            "ref_height_m": self.computer.reference[0], "ref_vz_mps": self.computer.reference[1],
        })
        self.log.write(row)
