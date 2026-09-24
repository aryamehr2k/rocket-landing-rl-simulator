"""Servo and igniter models between the flight computer and the physics.

A servo applies, in this order: clip to the gimbal limit, deadband on the change of command,
pulse quantisation, a pure delay, and a rate limit. See docs/conventions.md.
"""

from collections import deque

import numpy as np

from rocketsim.config import GimbalConfig, MotorConfig, ServoCalibration
from rocketsim.dragdevice import FULLY_OPEN, SHUT, DragDeviceConfig
from rocketsim.physics import Inputs


class Servo:
    """One gimbal servo. `command` takes a new angle each control step, `step` moves the horn each physics step."""

    def __init__(self, calibration: ServoCalibration, gimbal: GimbalConfig, dt: float) -> None:
        self.calibration = calibration
        self.gimbal = gimbal
        self.delay_steps = int(round(gimbal.servo_delay / dt))
        self.rate_step = gimbal.servo_rate_limit * dt
        self.reset()

    def reset(self) -> None:
        self.last_command = 0.0
        self.pulse = self.calibration.quantize(self.calibration.pulse_for_gimbal(0.0))
        self.pending: deque[float] = deque([self.pulse] * self.delay_steps)
        self.angle = self.calibration.gimbal_for_pulse(self.pulse)

    def command(self, angle: float) -> float:
        """Accept a gimbal angle command and return the pulse width sent to the servo."""
        clipped = min(max(angle, -self.gimbal.max_angle), self.gimbal.max_angle)
        if abs(clipped - self.last_command) < self.gimbal.servo_deadband:
            clipped = self.last_command
        self.last_command = clipped
        self.pulse = self.calibration.quantize(self.calibration.pulse_for_gimbal(clipped))
        return self.pulse

    def step(self) -> float:
        """One physics step: the pulse from `servo_delay` ago becomes the target, the horn moves toward it."""
        if self.delay_steps:
            self.pending.append(self.pulse)
            target_pulse = self.pending.popleft()
        else:
            target_pulse = self.pulse
        target = self.calibration.gimbal_for_pulse(target_pulse)
        self.angle += min(max(target - self.angle, -self.rate_step), self.rate_step)
        return self.angle


class BrakeServo:
    """Drives the drag device opening: a pure delay, then a rate limit set by the deploy and retract times."""

    def __init__(self, device: DragDeviceConfig, delay: float, dt: float) -> None:
        self.device = device
        self.delay_steps = int(round(delay / dt))
        self.open_step = FULLY_OPEN / device.deploy_time * dt
        self.close_step = FULLY_OPEN / device.retract_time * dt
        self.reset()

    def reset(self) -> None:
        self.target = SHUT
        self.pending: deque[float] = deque([SHUT] * self.delay_steps)
        self.fraction = SHUT

    def command(self, fraction: float) -> float:
        """Accept a new opening fraction, clipped to [0, 1]."""
        self.target = min(max(fraction, SHUT), FULLY_OPEN)
        return self.target

    def step(self) -> float:
        """One physics step: the command from `delay` ago becomes the target, the petals move toward it."""
        if self.delay_steps:
            self.pending.append(self.target)
            target = self.pending.popleft()
        else:
            target = self.target
        if target > self.fraction:
            self.fraction = min(target, self.fraction + self.open_step)
        else:
            self.fraction = max(target, self.fraction - self.close_step)
        return self.fraction


class Igniter:
    """Turns an ignition command into a motor start after this flight's igniter delay."""

    def __init__(self, index: int, motor: MotorConfig, rng: np.random.Generator) -> None:
        self.index = index
        self.motor = motor
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        """Draw the delay for a new flight: uniform in mean +- spread, never negative."""
        low = self.motor.ignition_delay_mean - self.motor.ignition_delay_spread
        high = self.motor.ignition_delay_mean + self.motor.ignition_delay_spread
        self.delay = max(0.0, float(self.rng.uniform(low, high)))
        self.command_time: float | None = None

    @property
    def commanded(self) -> bool:
        return self.command_time is not None

    def command(self, t: float, inputs: Inputs) -> bool:
        """Send the ignition command. Returns False if this motor was commanded before."""
        if self.commanded:
            return False
        self.command_time = t
        return inputs.ignite(self.index, t + self.delay)
