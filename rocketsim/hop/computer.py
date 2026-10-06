"""Flight computer of the electric vehicle: estimator, guidance, PID baseline, policy hook and safety.

Each control step it updates the reference from the mission, computes the PID commands, lets
a policy replace steering and/or throttle where the vehicle file says so, and passes the result
through the safety limits. A non-finite policy output falls back to the PID for that step.
"""

import math
from dataclasses import dataclass

from rocketsim.commands import PlaneAction
from rocketsim.estimator import Estimate, Estimator
from rocketsim.guidance_config import POLICY
from rocketsim.hop.mission import FLYING_PHASES, TIME_TOLERANCE, Guidance, HopPhase, MissionConfig
from rocketsim.hop.vehicle import RESIDUAL, HopVehicleConfig
from rocketsim.pid import TvcController, world_to_servo
from rocketsim.sensors import BaroSample, GpsSample, ImuSample
from rocketsim.simconfig import EnvironmentConfig

YAW_RATE_AXIS = 2
MIN_COS_TILT = 0.5  # the tilt compensation of the throttle stops growing beyond 60 degrees


@dataclass(frozen=True)
class HopCommand:
    gimbal_pitch: float = 0.0
    gimbal_yaw: float = 0.0
    throttle: float = 0.0
    roll_torque: float = 0.0
    arm: bool = False


class HopFlightComputer:
    def __init__(
        self, vehicle: HopVehicleConfig, mission: MissionConfig, environment: EnvironmentConfig,
        control_dt: float, launch_time: float,
    ) -> None:
        self.vehicle = vehicle
        self.mission = mission
        self.gravity = environment.gravity
        self.control_dt = control_dt
        self.launch_time = launch_time
        self.estimator = Estimator(
            vehicle.estimator, environment.gravity, vehicle.sensors.imu.dt, vehicle.body.pad_cg_height,
            baro_lag=vehicle.sensors.barometer.lag,
            gps_lag=vehicle.sensors.gps.lag if vehicle.sensors.gps is not None else 0.0,
        )
        self.guidance = Guidance(mission)
        self.tvc = TvcController(vehicle.attitude)
        self.controllers = vehicle.controllers
        self.hover_throttle = vehicle.mass * environment.gravity / vehicle.max_thrust
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.guidance = Guidance(self.mission)
        self.tvc.reset()
        self.launched = False
        self.aborted = False
        self.touched_down = False
        self.speed_integral = 0.0
        self.reference = (0.0, 0.0)
        self.policy_fallbacks = 0
        self.abort_reason = ""
        self.command = HopCommand()

    def set_mission(self, mission: MissionConfig) -> None:
        self.mission = mission
        self.guidance = Guidance(mission)

    @property
    def phase(self) -> HopPhase:
        if self.aborted:
            return HopPhase.ABORT
        if self.touched_down:
            return HopPhase.LANDED
        return self.guidance.phase

    @property
    def estimate(self) -> Estimate:
        return self.estimator.estimate

    @property
    def height(self) -> float:
        """Estimated height of the feet above the pad."""
        return float(self.estimator.position[2]) - self.vehicle.body.pad_cg_height

    @property
    def throttle(self) -> float:
        return self.command.throttle

    def imu(self, sample: ImuSample) -> None:
        self.estimator.update_imu(sample)

    def baro(self, sample: BaroSample) -> None:
        self.estimator.update_baro(sample)

    def gps(self, sample: GpsSample) -> None:
        self.estimator.update_gps(sample)

    def landed(self, t: float) -> None:
        self.touched_down = True

    def prepare(self, t: float) -> None:
        """Launch when the time has come: end the pad calibration and start the mission plan.

        Called before anything reads the estimate for the step at time t, so a policy and the PID
        see the same estimate and plan on the launch step."""
        if not self.launched and t + TIME_TOLERANCE >= self.launch_time:
            self.launched = True
            self.estimator.liftoff()
            self.guidance.start(t)

    def control_step(self, t: float, action: PlaneAction | None = None) -> HopCommand:
        """Decide the gimbal, throttle and roll commands for this control step."""
        self.prepare(t)
        phase, ref_height, ref_speed = self.guidance.update(t, self.control_dt)
        self.reference = (ref_height, ref_speed)
        estimate = self.estimate
        if self.phase not in FLYING_PHASES:
            self.command = HopCommand(arm=self.launched and self.phase != HopPhase.ABORT)
            return self.command
        self._check_abort(estimate)
        if self.aborted:
            self.command = HopCommand()
            return self.command
        delta_x, delta_y = self.tvc.command(estimate, self.control_dt, hold_position=True)
        throttle = self._altitude_pid(estimate, phase, ref_height, ref_speed)
        if action is not None and self.controllers.steering == POLICY:
            delta_x = self._policy_or_pid(action.delta_x, delta_x)
            delta_y = self._policy_or_pid(action.delta_y, delta_y)
        if action is not None and self.controllers.throttle == POLICY:
            throttle = self._policy_or_pid(action.throttle, throttle)
        pitch, yaw = world_to_servo(delta_x, delta_y, estimate.roll)
        roll = self.vehicle.roll
        roll_torque = -(roll.kp * estimate.roll + roll.kd * float(estimate.angular_rate[YAW_RATE_AXIS]))
        self.command = self._limited(pitch, yaw, throttle, roll_torque)
        return self.command

    def _altitude_pid(self, estimate: Estimate, phase: HopPhase, ref_height: float, ref_speed: float) -> float:
        gains = self.vehicle.altitude
        speed = float(estimate.velocity[2])
        wanted_speed = ref_speed
        if phase != HopPhase.LANDING:  # on the way down to touchdown only the speed matters
            correction = gains.height_gain * (ref_height - self.height)
            wanted_speed += min(max(correction, -gains.max_speed_correction), gains.max_speed_correction)
        error = wanted_speed - speed
        if gains.speed_integral_gain > 0.0:
            limit = gains.max_integral_accel / gains.speed_integral_gain
            self.speed_integral = min(max(self.speed_integral + error * self.control_dt, -limit), limit)
        accel = gains.speed_gain * error + gains.speed_integral_gain * self.speed_integral
        tilt = max(math.cos(estimate.total_tilt), MIN_COS_TILT)
        return self.hover_throttle * (1.0 + accel / self.gravity) / tilt

    def _policy_or_pid(self, wanted: float | None, pid_value: float) -> float:
        # Direct mode uses the policy's command, residual mode adds it to the PID's.
        if wanted is None:
            return pid_value
        if not math.isfinite(wanted):
            self.policy_fallbacks += 1
            return pid_value
        if self.controllers.mode == RESIDUAL:
            return pid_value + float(wanted)
        return float(wanted)

    def _check_abort(self, estimate: Estimate) -> None:
        safety = self.vehicle.safety
        distance = math.hypot(float(estimate.position[0]), float(estimate.position[1]))
        if estimate.total_tilt > safety.abort_tilt:
            self.aborted, self.abort_reason = True, "tilt beyond the abort limit"
        elif distance > safety.geofence_radius:
            self.aborted, self.abort_reason = True, "outside the geofence"

    def _limited(self, pitch: float, yaw: float, throttle: float, roll_torque: float) -> HopCommand:
        """Clip and rate limit every command against the last one that went out."""
        safety, gimbal = self.vehicle.safety, self.vehicle.body.gimbal
        step = safety.max_gimbal_rate * self.control_dt
        throttle_step = safety.max_throttle_rate * self.control_dt
        last = self.command

        def limit(value: float, previous: float, rate_step: float, low: float, high: float) -> float:
            value = min(max(value, low), high)
            return min(max(value, previous - rate_step), previous + rate_step)

        return HopCommand(
            gimbal_pitch=limit(pitch, last.gimbal_pitch, step, -gimbal.max_angle, gimbal.max_angle),
            gimbal_yaw=limit(yaw, last.gimbal_yaw, step, -gimbal.max_angle, gimbal.max_angle),
            throttle=limit(throttle, last.throttle, throttle_step, 0.0, 1.0),
            roll_torque=roll_torque,
            arm=True,
        )
