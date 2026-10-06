"""PPO-Clip (Schulman et al. 2017, arXiv:1707.06347) with a diagonal Gaussian policy.

The normalisation and the defaults follow Stable-Baselines3, so a run can be checked against it.
"""

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import nn

from rocketsim.training_config import PpoConfig

OBS_CLIP = 10.0
REWARD_CLIP = 10.0
NORM_EPSILON = 1e-8  # added to variances before the square root
INITIAL_COUNT = 1e-4  # pseudo sample count of a fresh RunningMeanStd (mean 0, variance 1)
ADAM_EPSILON = 1e-5
MAX_GRAD_NORM = 0.5
VALUE_LOSS_WEIGHT = 0.5
ADVANTAGE_EPSILON = 1e-8
HIDDEN_GAIN = math.sqrt(2.0)
ACTOR_OUTPUT_GAIN = 0.01  # starts with near-zero mean actions, so early behaviour is set by log_std_init
CRITIC_OUTPUT_GAIN = 1.0
EPISODE_WINDOW = 100  # default number of finished episodes the logged mean return and length cover
LOG_TWO_PI = math.log(2.0 * math.pi)
ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}


class VectorEnv(Protocol):
    """Steps all environments at once; dones is a bool array, and a finished one has already reset."""

    num_envs: int
    observation_size: int
    action_size: int
    action_low: float
    action_high: float

    def reset(self) -> np.ndarray: ...

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]: ...


class RunningMeanStd:
    """Running mean and variance, batches merged with Chan et al.'s parallel formula."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = INITIAL_COUNT

    def update(self, batch: np.ndarray) -> None:
        batch_mean, batch_var, batch_count = batch.mean(axis=0), batch.var(axis=0), batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        squares = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.var = squares / total
        self.count = total


class ObservationNormalizer:

    def __init__(self, size: int, clip: float = OBS_CLIP) -> None:
        self.stats = RunningMeanStd((size,))
        self.clip = clip

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.stats.var + NORM_EPSILON)

    def __call__(self, observations: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.stats.update(observations)
        return np.clip((observations - self.stats.mean) / self.std, -self.clip, self.clip)


class RewardScaler:
    """Divides rewards by the std of the discounted return so they stay around unit size.

    No mean is subtracted: that would add a constant to every step and change what ending a flight
    early or late is worth.
    """

    def __init__(self, num_envs: int, gamma: float, clip: float = REWARD_CLIP) -> None:
        self.stats = RunningMeanStd()
        self.discounted_sum = np.zeros(num_envs)
        self.gamma = gamma
        self.clip = clip

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.discounted_sum = self.discounted_sum * self.gamma + rewards
        self.stats.update(self.discounted_sum)
        scaled = np.clip(rewards / np.sqrt(self.stats.var + NORM_EPSILON), -self.clip, self.clip)
        self.discounted_sum[np.asarray(dones, dtype=bool)] = 0.0
        return scaled


def mlp(sizes: list[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:
            layers.append(ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


def init_layers(network: nn.Sequential, output_gain: float) -> None:
    # orthogonal init as in the PPO implementation details (Engstrom et al. 2020), small gain on the output
    linear = [m for m in network if isinstance(m, nn.Linear)]
    for layer in linear:
        nn.init.orthogonal_(layer.weight, gain=output_gain if layer is linear[-1] else HIDDEN_GAIN)
        nn.init.zeros_(layer.bias)


def gaussian_log_prob(actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    z = (actions - mean) / log_std.exp()
    return (-0.5 * z**2 - log_std - 0.5 * LOG_TWO_PI).sum(dim=-1)


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    # negative once std < 1/sqrt(2 pi e) ~ 0.24, which is normal late in training
    return (0.5 + 0.5 * LOG_TWO_PI + log_std).sum(dim=-1)


class ActorCritic(nn.Module):
    """Actor (mean action) and critic (value) as separate MLPs, plus a learned log std per action."""

    def __init__(self, obs_size: int, action_size: int, hidden: tuple[int, ...], activation: str, log_std_init: float) -> None:
        super().__init__()
        self.activation = activation
        self.actor = mlp([obs_size, *hidden, action_size], activation)
        self.critic = mlp([obs_size, *hidden, 1], activation)
        self.log_std = nn.Parameter(torch.full((action_size,), float(log_std_init)))
        init_layers(self.actor, ACTOR_OUTPUT_GAIN)
        init_layers(self.critic, CRITIC_OUTPUT_GAIN)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """GAE advantages and value targets; arrays are (steps, envs) and dones[t] means step t ended a flight."""
    # PPO paper eq. 11-12, run backwards:
    #   delta_t = r_t + gamma * V(s_t+1) * (1 - done_t) - V(s_t)
    #   A_t     = delta_t + gamma * lam * (1 - done_t) * A_t+1
    # A timeout also stops the bootstrap: running out of time is penalised here, not a cut-off.
    advantages = np.zeros_like(rewards)
    next_advantage = np.zeros_like(last_values)
    next_value = last_values
    for t in reversed(range(len(rewards))):
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        next_advantage = delta + gamma * lam * alive * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + values


def clipped_surrogate_loss(ratio: torch.Tensor, advantages: torch.Tensor, clip_range: float) -> torch.Tensor:
    """L_clip from eq. 7 of the PPO paper, negated so it can be minimised.

    r = pi_new / pi_old. Once r moves past 1 +- eps in the direction A favours, the clipped term wins
    and has no gradient, so one batch can't drag the policy too far.
    """
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages
    return -torch.min(unclipped, clipped).mean()


def explained_variance(predicted: np.ndarray, target: np.ndarray) -> float:
    """1 = the critic predicts the returns perfectly, 0 = no better than predicting their mean."""
    spread = np.var(target)
    return float("nan") if spread == 0 else float(1.0 - np.var(target - predicted) / spread)


@dataclass
class Rollout:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class Ppo:

    def __init__(self, env: VectorEnv, config: PpoConfig, episode_window: int = EPISODE_WINDOW) -> None:
        torch.manual_seed(config.seed)
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.model = ActorCritic(env.observation_size, env.action_size, config.hidden_layers, config.activation,
                                 config.log_std_init)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=ADAM_EPSILON)
        self.normalizer = ObservationNormalizer(env.observation_size)
        self.reward_scaler = RewardScaler(env.num_envs, config.gamma)
        self.timesteps = 0
        self.episode_return = np.zeros(env.num_envs)
        self.episode_length = np.zeros(env.num_envs, dtype=int)
        self.finished_returns: deque[float] = deque(maxlen=episode_window)
        self.finished_lengths: deque[int] = deque(maxlen=episode_window)
        self.raw_obs = env.reset()
        self.obs = self.normalizer(self.raw_obs)

    def collect(self, on_step: Callable[[int, list[dict[str, Any]]], None] | None = None) -> Rollout:
        """n_steps of the current policy in every env. on_step(timesteps, infos) is called after each step."""
        steps, envs = self.config.n_steps, self.env.num_envs
        obs = np.zeros((steps, envs, self.env.observation_size), dtype=np.float32)
        actions = np.zeros((steps, envs, self.env.action_size), dtype=np.float32)
        log_probs, values, rewards, dones = (np.zeros((steps, envs), dtype=np.float32) for _ in range(4))
        for t in range(steps):
            obs[t] = self.obs
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs[t])
                mean = self.model.actor(obs_tensor)
                action = mean + self.model.log_std.exp() * torch.randn_like(mean)
                log_probs[t] = gaussian_log_prob(action, mean, self.model.log_std).numpy()
                values[t] = self.model.value(obs_tensor).numpy()
            actions[t] = action.numpy()
            # keep the unclipped action for the log-prob, clip only what the env gets
            raw_obs, raw_rewards, done, infos = self.env.step(np.clip(actions[t], self.env.action_low, self.env.action_high))
            self.timesteps += envs
            self._track_episodes(raw_rewards, done)
            rewards[t] = self.reward_scaler(raw_rewards, done)
            dones[t] = done
            self.raw_obs = raw_obs
            self.obs = self.normalizer(raw_obs)
            if on_step is not None:
                on_step(self.timesteps, infos)
        with torch.no_grad():
            last_values = self.model.value(torch.as_tensor(self.obs, dtype=torch.float32)).numpy()
        advantages, returns = compute_gae(rewards, values, dones, last_values, self.config.gamma, self.config.gae_lambda)
        return Rollout(*(_merge_steps(a) for a in (obs, actions, log_probs, values, advantages, returns)))

    def _track_episodes(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        self.episode_return += rewards
        self.episode_length += 1
        for i in np.flatnonzero(dones):
            self.finished_returns.append(float(self.episode_return[i]))
            self.finished_lengths.append(int(self.episode_length[i]))
            self.episode_return[i], self.episode_length[i] = 0.0, 0

    def update(self, rollout: Rollout) -> dict[str, float]:
        """n_epochs over shuffled minibatches. Returns mean losses and diagnostics for the log."""
        config = self.config
        data = {k: torch.as_tensor(v) for k, v in vars(rollout).items()}
        history: dict[str, list[float]] = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
        size = len(rollout.obs)
        for _ in range(config.n_epochs):
            order = self.rng.permutation(size)
            for start in range(0, size, config.batch_size):
                batch = {k: v[order[start : start + config.batch_size]] for k, v in data.items()}
                advantages = batch["advantages"]
                if len(advantages) > 1:  # normalised per minibatch like SB3; a single sample has no std
                    advantages = (advantages - advantages.mean()) / (advantages.std() + ADVANTAGE_EPSILON)
                log_prob = gaussian_log_prob(batch["actions"], self.model.actor(batch["obs"]), self.model.log_std)
                log_ratio = log_prob - batch["log_probs"]
                ratio = log_ratio.exp()
                policy_loss = clipped_surrogate_loss(ratio, advantages, config.clip_range)
                value_loss = nn.functional.mse_loss(self.model.value(batch["obs"]), batch["returns"])
                entropy = gaussian_entropy(self.model.log_std)
                loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss - config.ent_coef * entropy  # eq. 9
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
                with torch.no_grad():
                    # "k3" estimate of KL(old || new), http://joschu.net/blog/kl-approx.html
                    history["approx_kl"].append(float(((ratio - 1.0) - log_ratio).mean()))
                    history["clip_fraction"].append(float(((ratio - 1.0).abs() > config.clip_range).float().mean()))
                history["policy_loss"].append(policy_loss.item())
                history["value_loss"].append(value_loss.item())
                history["entropy"].append(entropy.item())
        stats = {k: float(np.mean(v)) for k, v in history.items()}
        stats["explained_variance"] = explained_variance(rollout.values, rollout.returns)
        stats["std"] = float(self.model.log_std.exp().mean())
        return stats

    def save(self, path: str | Path) -> None:
        """Weights, optimizer state and normaliser statistics; the config goes along for reference."""
        torch.save({
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "timesteps": self.timesteps,
            "observation_stats": _stats_state(self.normalizer.stats), "return_stats": _stats_state(self.reward_scaler.stats),
            "config": asdict(self.config),
        }, path)

    def load(self, path: str | Path) -> None:
        """Inverse of save(). Running reward sums and episode counters start from zero."""
        state = torch.load(path, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.timesteps = state["timesteps"]
        _load_stats(self.normalizer.stats, state["observation_stats"])
        _load_stats(self.reward_scaler.stats, state["return_stats"])
        self.obs = self.normalizer(self.raw_obs, update=False)


def _merge_steps(array: np.ndarray) -> np.ndarray:  # (steps, envs, ...) -> (steps * envs, ...)
    return array.reshape(-1, *array.shape[2:])


def _stats_state(stats: RunningMeanStd) -> dict[str, Any]:
    return {"mean": torch.as_tensor(stats.mean), "var": torch.as_tensor(stats.var), "count": float(stats.count)}


def _load_stats(stats: RunningMeanStd, state: dict[str, Any]) -> None:
    stats.mean, stats.var, stats.count = state["mean"].numpy(), state["var"].numpy(), state["count"]
