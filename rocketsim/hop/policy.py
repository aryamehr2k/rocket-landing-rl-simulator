"""Turning the network's outputs into commands for the electric vehicle, in training and in flight."""

import math

import numpy as np

from rocketsim.commands import PlaneAction
from rocketsim.hop.computer import MIN_COS_TILT, HopFlightComputer
from rocketsim.hop.training import HopActionConfig
from rocketsim.hop.vehicle import RESIDUAL
from rocketsim.observation import PLANES, ObservationBuilder
from rocketsim.policy import MlpPolicy

GIMBAL, THROTTLE = 0, 1
ACTION_SIZE = 2
LOW, HIGH = -1.0, 1.0


def plane_action(outputs: list[np.ndarray | None], computer: HopFlightComputer, action: HopActionConfig) -> PlaneAction:
    """The flight computer's action from the two plane outputs, each a gimbal and a throttle vote in [-1, 1].

    The votes are averaged. Direct mode scales the gimbal by the gimbal limit and adds the vote times `throttle_range`
    to the hover throttle; residual mode adds both to the PID's commands. A missing or non-finite output leaves that
    plane to the PID.
    """
    residual = computer.controllers.mode == RESIDUAL
    gimbal_scale = action.residual_gimbal if residual else computer.vehicle.body.gimbal.max_angle
    gimbal: list[float | None] = [None, None]
    votes = []
    for plane, output in zip(PLANES, outputs):
        if output is None or not np.all(np.isfinite(output)):
            continue
        gimbal[plane] = float(np.clip(output[GIMBAL], LOW, HIGH)) * gimbal_scale
        votes.append(float(np.clip(output[THROTTLE], LOW, HIGH)))
    throttle = None
    if votes:
        throttle = float(np.mean(votes)) * action.throttle_range
        if not residual:
            tilt = max(math.cos(computer.estimate.total_tilt), MIN_COS_TILT)
            throttle += computer.hover_throttle / tilt
    return PlaneAction(delta_x=gimbal[0], delta_y=gimbal[1], throttle=throttle)


class HopPolicyController:
    """Runs a trained network for both planes on the flight computer's state."""

    def __init__(self, policy: MlpPolicy, observer: ObservationBuilder, action: HopActionConfig) -> None:
        if observer.names != policy.field_names:
            raise ValueError(f"model was trained on fields {policy.field_names}, the training file lists {observer.names}")
        self.policy = policy
        self.observer = observer
        self.action_config = action
        self.last_gimbal = np.zeros(len(PLANES))

    def reset(self) -> None:
        self.last_gimbal[:] = 0.0

    def action(self, computer: HopFlightComputer, t: float) -> PlaneAction:
        outputs = [self.policy.forward(self.observer.build(computer, plane, t, self.last_gimbal[plane])) for plane in PLANES]
        result = plane_action(list(outputs), computer, self.action_config)
        for plane, value in zip(PLANES, (result.delta_x, result.delta_y)):
            if value is not None:
                self.last_gimbal[plane] = value
        return result
