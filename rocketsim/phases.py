"""Flight phases and the rules that move between them. See docs/conventions.md."""

from enum import Enum

from rocketsim.estimator import Estimate
from rocketsim.guidance_config import PhaseConfig


class Phase(str, Enum):
    PAD = "PAD"
    BOOST = "BOOST"
    COAST = "COAST"
    DESCENT = "DESCENT"
    LANDING_BURN = "LANDING_BURN"
    LANDED = "LANDED"
    ABORT = "ABORT"


CONTROLLED_PHASES = (Phase.BOOST, Phase.LANDING_BURN)
LANDING_IGNITION_PHASES = (Phase.COAST, Phase.DESCENT)


class PhaseMachine:
    """Tracks the phase from the IMU and the estimate. Liftoff is detected on every IMU sample."""

    def __init__(self, config: PhaseConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.PAD
        self.liftoff_time: float | None = None
        self.landing_command_time: float | None = None
        self.history: list[tuple[float, Phase]] = [(0.0, Phase.PAD)]

    def imu_update(self, t: float, specific_force_bz: float) -> bool:
        """Liftoff check on one IMU sample. Returns True at the sample that shows liftoff."""
        if self.phase == Phase.PAD and specific_force_bz > self.config.liftoff_accel:
            self.liftoff_time = t
            self._transition(t, Phase.BOOST)
            return True
        return False

    def update(self, t: float, specific_force_bz: float, estimate: Estimate) -> Phase:
        """The slower transitions, once per control step."""
        if self.phase == Phase.BOOST and self.liftoff_time is not None:
            if estimate.total_tilt > self.config.abort_tilt:
                self._transition(t, Phase.ABORT)
            elif t - self.liftoff_time >= self.config.min_boost_time and specific_force_bz < self.config.burnout_accel:
                self._transition(t, Phase.COAST)
        elif self.phase == Phase.COAST and estimate.velocity[2] < self.config.apogee_vz:
            self._transition(t, Phase.DESCENT)
        return self.phase

    def landing_commanded(self, t: float) -> None:
        if self.phase in LANDING_IGNITION_PHASES:
            self.landing_command_time = t
            self._transition(t, Phase.LANDING_BURN)

    def landed(self, t: float) -> None:
        if self.phase != Phase.ABORT:
            self._transition(t, Phase.LANDED)

    def _transition(self, t: float, phase: Phase) -> None:
        self.phase = phase
        self.history.append((t, phase))
