"""Last checks on every command before it reaches the servos and igniters."""

from dataclasses import replace

from rocketsim.commands import ControlCommand
from rocketsim.config import GimbalConfig
from rocketsim.dragdevice import FULLY_OPEN, SHUT, DragDeviceConfig
from rocketsim.estimator import Estimate
from rocketsim.guidance_config import SafetyConfig
from rocketsim.phases import LANDING_IGNITION_PHASES, Phase

BRAKE_PHASES = (Phase.DESCENT, Phase.LANDING_BURN)
TAIL_DEVICE_PHASES = (Phase.LANDING_BURN,)


class Safety:
    def __init__(
        self,
        config: SafetyConfig,
        gimbal: GimbalConfig,
        device: DragDeviceConfig | None = None,
        descent_cg: float = 0.0,
        control_dt: float | None = None,
    ) -> None:
        self.config = config
        self.gimbal = gimbal
        # A device behind the descent centre of gravity tips a falling rocket over, so it may
        # only open once the gimbal has thrust to hold the attitude.
        self.device_behind_cg = device is not None and device.station > descent_cg
        # The commanded gimbal may move at most this much between two control steps.
        self.max_step = None if config.max_gimbal_rate is None or control_dt is None else config.max_gimbal_rate * control_dt
        self.reset()

    def reset(self) -> None:
        self.last_pitch = 0.0
        self.last_yaw = 0.0

    def _slew(self, wanted: float, last: float) -> float:
        if self.max_step is None:
            return wanted
        return min(max(wanted, last - self.max_step), last + self.max_step)

    def filter(
        self,
        command: ControlCommand,
        phase: Phase,
        t: float,
        liftoff_time: float | None,
        estimate: Estimate,
        height: float,
    ) -> tuple[ControlCommand, tuple[str, ...]]:
        """Return the command that may go out and the reasons for anything that was taken out."""
        limit = self.gimbal.max_angle
        pitch = self._slew(min(max(command.gimbal_pitch, -limit), limit), self.last_pitch)
        yaw = self._slew(min(max(command.gimbal_yaw, -limit), limit), self.last_yaw)
        refused: list[str] = []
        ascent = command.ignite_ascent
        if ascent and phase != Phase.PAD:
            ascent, _ = False, refused.append(f"ascent ignition refused in {phase.value}")
        landing = command.ignite_landing
        if landing:
            reason = self._landing_refusal(phase, t, liftoff_time, estimate, height)
            if reason:
                landing = False
                refused.append(reason)
        brake = min(max(command.brake_fraction, SHUT), FULLY_OPEN)
        if brake > SHUT:
            reason = self._brake_refusal(phase)
            if reason:
                brake = SHUT
                refused.append(reason)
        if phase == Phase.ABORT:
            pitch, yaw, ascent, landing, brake = 0.0, 0.0, False, False, SHUT
        self.last_pitch, self.last_yaw = pitch, yaw
        filtered = replace(
            command, gimbal_pitch=pitch, gimbal_yaw=yaw, ignite_ascent=ascent, ignite_landing=landing,
            brake_fraction=brake,
        )
        return filtered, tuple(refused)

    def _landing_refusal(
        self, phase: Phase, t: float, liftoff_time: float | None, estimate: Estimate, height: float
    ) -> str | None:
        config = self.config
        if phase not in LANDING_IGNITION_PHASES:
            return f"landing ignition refused in {phase.value}"
        if liftoff_time is None or t - liftoff_time < config.landing_ignition_lockout:
            return f"landing ignition refused: within {config.landing_ignition_lockout:g} s of liftoff"
        if estimate.total_tilt > config.max_landing_ignition_tilt:
            return "landing ignition refused: tilt too large"
        if height > config.max_landing_ignition_height:
            return "landing ignition refused: too high"
        return None

    def _brake_refusal(self, phase: Phase) -> str | None:
        if phase not in BRAKE_PHASES:
            return f"brake refused in {phase.value}"
        if self.device_behind_cg and phase not in TAIL_DEVICE_PHASES:
            return f"brake refused in {phase.value}: a device behind the centre of gravity tips the fall"
        return None
