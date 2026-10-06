"""A trained policy as plain numpy: the network weights, the observation normalisation and a runner.

Saved as a .npz, so flying a model and the C export need no PyTorch.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rocketsim.commands import PlaneAction
from rocketsim.env import ACTION_HIGH, ACTION_LOW, GIMBAL, IGNITE
from rocketsim.flightcomputer import FlightComputer
from rocketsim.observation import PLANES, ObservationBuilder

if TYPE_CHECKING:  # torch is only needed when training
    from rocketsim.ppo import ActorCritic, ObservationNormalizer

POLICY_FILE = "policy.npz"


@dataclass(frozen=True)
class MlpPolicy:
    """Fully connected network with one activation, plus the input normalisation to apply first."""

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activation: str
    obs_mean: np.ndarray
    obs_std: np.ndarray
    obs_clip: float
    field_names: tuple[str, ...]

    def forward(self, observation: np.ndarray) -> np.ndarray:
        """Deterministic action in [-1, 1] for one plane observation (already divided by the field scales)."""
        x = np.clip((observation.astype(np.float32) - self.obs_mean) / self.obs_std, -self.obs_clip, self.obs_clip)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = x @ weight.T + bias
            x = np.tanh(x) if self.activation == "tanh" else np.maximum(x, 0.0)
        out = x @ self.weights[-1].T + self.biases[-1]
        return np.clip(out, ACTION_LOW, ACTION_HIGH).astype(np.float32)

    def save(self, path: str | Path) -> None:
        arrays: dict[str, Any] = {"obs_mean": self.obs_mean, "obs_std": self.obs_std, "obs_clip": np.array(self.obs_clip)}
        arrays["activation"] = np.array(self.activation)
        arrays["field_names"] = np.array(self.field_names)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"] = weight
            arrays[f"b{i}"] = bias
        np.savez(path, **arrays)


def load_policy(path: str | Path) -> MlpPolicy:
    """Read a policy .npz written by MlpPolicy.save."""
    data = np.load(path)
    layers = sum(1 for key in data.files if key.startswith("w"))
    return MlpPolicy(
        weights=tuple(data[f"w{i}"].astype(np.float32) for i in range(layers)),
        biases=tuple(data[f"b{i}"].astype(np.float32) for i in range(layers)),
        activation=str(data["activation"]),
        obs_mean=data["obs_mean"].astype(np.float32),
        obs_std=data["obs_std"].astype(np.float32),
        obs_clip=float(data["obs_clip"]),
        field_names=tuple(str(n) for n in data["field_names"]),
    )


def policy_from_training(model: "ActorCritic", normalizer: "ObservationNormalizer", field_names: tuple[str, ...]) -> MlpPolicy:
    """The actor's weights and the observation normalisation of a PPO run, as numpy."""
    linear = [layer for layer in model.actor if hasattr(layer, "weight")]
    weights = tuple(layer.weight.detach().cpu().numpy().astype(np.float32) for layer in linear)
    biases = tuple(layer.bias.detach().cpu().numpy().astype(np.float32) for layer in linear)
    return MlpPolicy(
        weights, biases, model.activation, normalizer.stats.mean.astype(np.float32), normalizer.std.astype(np.float32),
        float(normalizer.clip), field_names,
    )


class PolicyController:
    """Runs the network once per plane, as the firmware does; ignition comes from the mean of the two ignite outputs."""

    def __init__(self, policy: MlpPolicy, observer: ObservationBuilder, max_angle: float, ignite_threshold: float) -> None:
        if observer.names != policy.field_names:
            raise ValueError(f"policy trained on fields {policy.field_names}, YAML lists {observer.names}")
        self.policy = policy
        self.observer = observer
        self.max_angle = max_angle
        self.ignite_threshold = ignite_threshold
        self.last_command = np.zeros(len(PLANES))

    def reset(self) -> None:
        self.last_command[:] = 0.0

    def action(self, computer: FlightComputer, t: float) -> PlaneAction:
        """Run the network for both planes. A non-finite output hands that plane to the PID for this step."""
        gimbal: list[float | None] = [None, None]
        ignite_outputs = []
        for plane in PLANES:
            out = self.policy.forward(self.observer.build(computer, plane, t, self.last_command[plane]))
            if not np.all(np.isfinite(out)):
                continue
            gimbal[plane] = float(out[GIMBAL]) * self.max_angle
            ignite_outputs.append(float(out[IGNITE]))
            self.last_command[plane] = gimbal[plane]
        ignite = bool(np.mean(ignite_outputs) > self.ignite_threshold) if ignite_outputs else None
        return PlaneAction(gimbal[0], gimbal[1], ignite)
