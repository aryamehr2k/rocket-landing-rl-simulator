"""Last checks on every command before it reaches the servos and igniters."""

from dataclasses import replace

from rocketsim.commands import ControlCommand
from rocketsim.config import GimbalConfig
from rocketsim.estimator import Estimate
from rocketsim.guidance_config import SafetyConfig
from rocketsim.phases import LANDING_IGNITION_PHASES, Phase


class Safety:
    def __init__(self, config: SafetyConfig, gimbal: GimbalConfig) -> None:
        self.config = config
        self.gimbal = gimbal

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
        pitch = min(max(command.gimbal_pitch, -limit), limit)
        yaw = min(max(command.gimbal_yaw, -limit), limit)
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
        if phase == Phase.ABORT:
            pitch, yaw, ascent, landing = 0.0, 0.0, False, False
        return replace(command, gimbal_pitch=pitch, gimbal_yaw=yaw, ignite_ascent=ascent, ignite_landing=landing), tuple(refused)

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
