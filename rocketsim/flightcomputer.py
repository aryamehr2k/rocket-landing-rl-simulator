"""Everything that will run on the board: estimator, phases, landing trigger, PID and safety.

The simulation feeds it sensor samples and asks for a command once per control step. A policy
may hand in a PlaneAction instead of letting the PID and the trigger decide; phases and safety
still apply to it. See docs/conventions.md.
"""

from rocketsim.commands import ControlCommand, PlaneAction
from rocketsim.config import RocketConfig
from rocketsim.dragdevice import FULLY_OPEN, SHUT
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
        self.safety = Safety(computer.safety, rocket.gimbal, rocket.drag_device, rocket.descent_cg)
        self.brake = computer.brake
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.phases.reset()
        self.pid.reset()
        self.trigger.reset()
        self.specific_force_bz = 0.0
        self.launched = False
        self.brake_deployed = False
        self.brake_deployed_at: float | None = None
        self.brake_retracted = False
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
        speed_down = -float(estimate.velocity[2])
        brake = self.brake_rule(phase, speed_down)
        if action is not None and action.brake is not None:
            brake = action.brake
        if self.brake_deployed and self.brake_deployed_at is None:
            self.brake_deployed_at = t
        if action is not None:
            ignite_landing = action.ignite_landing  # safety decides whether this phase allows it
        else:
            if phase == Phase.DESCENT and self.brake_fully_open(t):
                self.trigger.observe_descent(t, speed_down, self.specific_force_bz, estimate.total_tilt)
            ignite_landing = phase == Phase.DESCENT and self.trigger.should_ignite(
                self.height, speed_down, self.command.brake_fraction
            )
        wanted = ControlCommand(pitch, yaw, ignite_ascent, ignite_landing, brake)
        command, self.refusals = self.safety.filter(wanted, phase, t, self.phases.liftoff_time, estimate, self.height)
        if command.ignite_ascent:
            self.launched = True
        if command.ignite_landing:
            self.phases.landing_commanded(t)
        self.command = command
        return command

    def brake_fully_open(self, t: float) -> bool:
        """True once the device has had its deploy time plus the servo delay since the open command."""
        device = self.rocket.drag_device
        if device is None or self.brake_deployed_at is None:
            return False
        latency = self.rocket.control.action_delay_steps * self.control_dt + self.rocket.gimbal.servo_delay
        return t - self.brake_deployed_at >= device.deploy_time + latency

    def brake_rule(self, phase: Phase, speed_down: float) -> float:
        """Drag device opening from the rules: open once falling fast enough, hold in the burn, shut at hand-over."""
        rules = self.brake
        if rules is None:
            return SHUT
        if phase == Phase.DESCENT:
            if speed_down > rules.deploy_descent_speed:
                self.brake_deployed = True
            return FULLY_OPEN if self.brake_deployed else SHUT
        if phase == Phase.LANDING_BURN and self.brake_deployed:
            if speed_down < rules.retract_descent_speed:
                self.brake_retracted = True
            return SHUT if self.brake_retracted else rules.burn_fraction
        return SHUT

    def landed(self, t: float) -> None:
        self.phases.landed(t)
