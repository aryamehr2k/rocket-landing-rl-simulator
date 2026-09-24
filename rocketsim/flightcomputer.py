"""Everything that will run on the board: estimator, phases, landing trigger, PID and safety.

The simulation feeds it sensor samples and asks for a command once per control step. A policy
may hand in a PlaneAction instead of letting the PID and the trigger decide; phases and safety
still apply to it. See docs/conventions.md.
"""

from rocketsim.commands import ControlCommand, PlaneAction
from rocketsim.config import RocketConfig
from rocketsim.estimator import Estimate, Estimator
from rocketsim.landing_trigger import LandingTrigger
from rocketsim.phases import CONTROLLED_PHASES, Phase, PhaseMachine
from rocketsim.pid import TvcController, world_to_servo
from rocketsim.safety import Safety
from rocketsim.sensors import BaroSample, ImuSample
from rocketsim.simconfig import EnvironmentConfig

BODY_AXIS_INDEX = 2
HALF_STEP = 0.5  # on average the trigger condition comes true halfway between two control steps


class FlightComputer:
    def __init__(self, rocket: RocketConfig, environment: EnvironmentConfig, control_dt: float, launch_time: float) -> None:
        computer = rocket.computer
        self.rocket = rocket
        self.control_dt = control_dt
        self.launch_time = launch_time
        self.estimator = Estimator(
            computer.estimator, environment.gravity, computer.sensors.imu.dt, rocket.pad_cg_height,
            baro_lag=computer.sensors.barometer.lag,
        )
        self.phases = PhaseMachine(computer.phases)
        self.trigger = LandingTrigger(
            rocket, computer.landing_trigger, environment,
            decision_latency=(rocket.control.action_delay_steps + HALF_STEP) * control_dt,
        )
        self.pid = TvcController(computer.pid)
        self.safety = Safety(computer.safety, rocket.gimbal)
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.phases.reset()
        self.pid.reset()
        self.specific_force_bz = 0.0
        self.launched = False
        self.refusals: tuple[str, ...] = ()
        self.command = ControlCommand()

    @property
    def phase(self) -> Phase:
        return self.phases.phase

    @property
    def estimate(self) -> Estimate:
        return self.estimator.estimate

    @property
    def height(self) -> float:
        """Estimated height of the feet above the ground."""
        return float(self.estimator.position[2]) - self.rocket.pad_cg_height

    def imu(self, sample: ImuSample) -> None:
        self.specific_force_bz = float(sample.specific_force[BODY_AXIS_INDEX])
        if self.phases.imu_update(sample.time, self.specific_force_bz):
            self.estimator.liftoff()
        self.estimator.update_imu(sample)

    def baro(self, sample: BaroSample) -> None:
        self.estimator.update_baro(sample)

    def control_step(self, t: float, action: PlaneAction | None = None) -> ControlCommand:
        """Decide the servo and igniter commands for this control step."""
        estimate = self.estimate
        phase = self.phases.update(t, self.specific_force_bz, estimate)
        ignite_ascent = phase == Phase.PAD and not self.launched and t >= self.launch_time
        pitch = yaw = 0.0
        if phase in CONTROLLED_PHASES:
            if action is None:
                delta_x, delta_y = self.pid.command(estimate, self.control_dt, hold_position=True)
            else:
                delta_x, delta_y = action.delta_x, action.delta_y
            pitch, yaw = world_to_servo(delta_x, delta_y, estimate.roll)
        else:
            self.pid.reset()
        if action is not None:
            ignite_landing = action.ignite_landing  # safety decides whether this phase allows it
        else:
            speed_down = -float(estimate.velocity[2])
            ignite_landing = phase == Phase.DESCENT and self.trigger.should_ignite(self.height, speed_down)
        wanted = ControlCommand(pitch, yaw, ignite_ascent, ignite_landing)
        command, self.refusals = self.safety.filter(wanted, phase, t, self.phases.liftoff_time, estimate, self.height)
        if command.ignite_ascent:
            self.launched = True
        if command.ignite_landing:
            self.phases.landing_commanded(t)
        self.command = command
        return command

    def landed(self, t: float) -> None:
        self.phases.landed(t)
