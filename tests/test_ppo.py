from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from rocketsim.env import ACTION_HIGH, ACTION_LOW
from rocketsim.policy import policy_from_training
from rocketsim.ppo import (
    NORM_EPSILON, ActorCritic, ObservationNormalizer, Ppo, RewardScaler, RunningMeanStd, clipped_surrogate_loss,
    compute_gae, gaussian_entropy, gaussian_log_prob,
)
from rocketsim.training_config import PpoConfig

TOY_ENVS = 8
TOY_EPISODE_STEPS = 20
TOY_SPEED = 0.2  # distance moved per step at full action
TOY_UPDATES = 25
TOY_SIZE = 2  # the point's x and y: both the observation and the action


def toy_config(**changes: Any) -> PpoConfig:
    values: dict[str, Any] = dict(
        total_timesteps=0, n_sims=TOY_ENVS, workers=1, n_steps=32, batch_size=64, n_epochs=10, learning_rate=1e-3,
        gamma=0.9, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, hidden_layers=(32, 32), activation="tanh", seed=1,
        log_std_init=-0.5, checkpoint_every=1,
    )
    values.update(changes)
    return PpoConfig(**values)


class PointToTarget:
    """A point in the plane that must be steered to a random target; the reward is minus the distance."""

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs, self.observation_size, self.action_size = num_envs, TOY_SIZE, TOY_SIZE
        self.action_low, self.action_high = ACTION_LOW, ACTION_HIGH
        self.rng = np.random.default_rng(seed)
        self.offset = np.zeros((num_envs, TOY_SIZE))
        self.steps = np.zeros(num_envs, dtype=int)
        self.largest_action = 0.0

    def _draw(self, count: int) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=(count, TOY_SIZE))

    def reset(self) -> np.ndarray:
        self.offset = self._draw(self.num_envs)
        self.steps[:] = 0
        return self.offset.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        self.largest_action = max(self.largest_action, float(np.abs(actions).max()))
        self.offset -= TOY_SPEED * actions
        self.steps += 1
        rewards = -np.linalg.norm(self.offset, axis=1)
        dones = self.steps >= TOY_EPISODE_STEPS
        self.offset[dones] = self._draw(int(dones.sum()))
        self.steps[dones] = 0
        return self.offset.copy(), rewards, dones, [{} for _ in range(self.num_envs)]


def test_gae_stops_at_a_done() -> None:
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.array([[0.5], [1.0], [1.5]])
    dones = np.array([[0.0], [1.0], [0.0]])
    gamma, lam, last = 0.9, 0.8, np.array([2.0])
    advantages, returns = compute_gae(rewards, values, dones, last, gamma, lam)
    delta2 = 3.0 + gamma * 2.0 - 1.5
    delta1 = 2.0 - 1.0  # step 1 ends the episode: no value of the next state
    delta0 = 1.0 + gamma * 1.0 - 0.5
    expected = [delta0 + gamma * lam * delta1, delta1, delta2]
    assert advantages[:, 0] == pytest.approx(expected)
    assert returns[:, 0] == pytest.approx(np.array(expected) + values[:, 0])


def test_running_mean_std_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, size=(n, 4)) for n in (5, 17, 1, 40)]
    stats = RunningMeanStd((4,))
    for batch in batches:
        stats.update(batch)
    everything = np.concatenate(batches)
    assert stats.mean == pytest.approx(everything.mean(axis=0), rel=1e-4)
    assert stats.var == pytest.approx(everything.var(axis=0), rel=1e-3)


def test_observation_normalizer_subtracts_the_mean_and_divides_by_the_std() -> None:
    observations = np.random.default_rng(0).normal(3.0, 2.0, size=(50, 3))
    normalized = ObservationNormalizer(3)(observations)
    expected = (observations - observations.mean(axis=0)) / np.sqrt(observations.var(axis=0) + NORM_EPSILON)
    assert normalized == pytest.approx(expected, abs=1e-3)  # the fresh statistics' prior weighs 1e-4 of a sample


def test_observation_normalizer_clips_both_sides() -> None:
    normalizer = ObservationNormalizer(1, clip=2.0)
    normalizer(np.array([[0.0], [1.0]]))
    assert normalizer(np.array([[100.0]]), update=False)[0, 0] == 2.0
    assert normalizer(np.array([[-100.0]]), update=False)[0, 0] == -2.0


def test_reward_scaler_divides_by_the_spread_of_the_discounted_sums() -> None:
    scaler = RewardScaler(num_envs=2, gamma=0.5)
    scaler(np.array([1.0, -1.0]), np.array([False, False]))
    scaler(np.array([2.0, 0.5]), np.array([True, False]))
    scaled = scaler(np.array([4.0, 1.0]), np.array([False, False]))
    # Sums seen: 1 and -1, then 0.5 + 2 = 2.5 and -0.5 + 0.5 = 0, then 4 (restarted after its done) and 1.
    # Their variance is 15.875 / 6.
    assert scaled == pytest.approx(np.array([4.0, 1.0]) / np.sqrt(15.875 / 6.0), rel=1e-3)
    assert scaler.discounted_sum == pytest.approx([4.0, 1.0])


def test_reward_scaler_clips() -> None:
    # After a single sum the spread is still near zero, so any reward is far past the clip.
    assert RewardScaler(num_envs=1, gamma=0.5, clip=2.0)(np.array([5.0]), np.array([False]))[0] == 2.0


def test_reward_scaler_accepts_integer_dones() -> None:
    scaler = RewardScaler(num_envs=4, gamma=0.9)
    scaler.discounted_sum = np.array([1.0, 2.0, 3.0, 4.0])
    scaler(np.zeros(4), np.array([0, 0, 1, 0]))
    assert scaler.discounted_sum == pytest.approx([0.9, 1.8, 0.0, 3.6])


def test_gaussian_log_prob_and_entropy_match_torch() -> None:
    torch.manual_seed(0)
    mean, log_std = torch.randn(6, 3), torch.randn(3) * 0.3
    actions = torch.randn(6, 3)
    reference = torch.distributions.Normal(mean, log_std.exp())
    assert torch.allclose(gaussian_log_prob(actions, mean, log_std), reference.log_prob(actions).sum(-1), atol=1e-5)
    assert torch.allclose(gaussian_entropy(log_std), reference.entropy()[0].sum(), atol=1e-5)


@pytest.mark.parametrize("ratio, advantage, has_gradient", [
    (1.5, 1.0, False), (0.5, -1.0, False), (1.5, -1.0, True), (0.5, 1.0, True), (1.1, 1.0, True),
])
def test_clipped_objective_stops_pushing_past_the_clip(ratio: float, advantage: float, has_gradient: bool) -> None:
    log_ratio = torch.tensor([float(np.log(ratio))], requires_grad=True)
    clipped_surrogate_loss(log_ratio.exp(), torch.tensor([advantage]), clip_range=0.2).backward()
    assert (log_ratio.grad is not None and log_ratio.grad.abs().item() > 0.0) == has_gradient


def test_exported_policy_equals_the_torch_actor() -> None:
    torch.manual_seed(3)
    model = ActorCritic(5, 2, (16, 8), "relu", log_std_init=0.0)
    for layer in model.actor:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, std=0.3)  # larger than the 0.01 output gain, so the test means something
    normalizer = ObservationNormalizer(5)
    rng = np.random.default_rng(1)
    normalizer(rng.normal(2.0, 3.0, size=(50, 5)))
    policy = policy_from_training(model, normalizer, tuple("abcde"))
    observations = rng.normal(2.0, 3.0, size=(20, 5))
    with torch.no_grad():
        expected = model.actor(torch.as_tensor(normalizer(observations, update=False), dtype=torch.float32)).numpy()
    exported = np.array([policy.forward(o) for o in observations])
    assert exported == pytest.approx(np.clip(expected, ACTION_LOW, ACTION_HIGH), abs=1e-5)


def test_environment_gets_clipped_actions_and_the_rollout_keeps_the_samples() -> None:
    env = PointToTarget(TOY_ENVS, seed=0)
    rollout = Ppo(env, toy_config(log_std_init=1.0)).collect()
    assert np.abs(rollout.actions).max() > ACTION_HIGH
    assert env.largest_action <= ACTION_HIGH


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    agent = Ppo(PointToTarget(TOY_ENVS, seed=0), toy_config(n_epochs=1))
    agent.update(agent.collect())
    agent.save(tmp_path / "model.pt")
    other = Ppo(PointToTarget(TOY_ENVS, seed=0), toy_config(n_epochs=1, seed=2))
    other.load(tmp_path / "model.pt")
    assert other.timesteps == agent.timesteps
    for name in ("mean", "var", "count"):
        assert getattr(other.normalizer.stats, name) == pytest.approx(getattr(agent.normalizer.stats, name))
        assert getattr(other.reward_scaler.stats, name) == pytest.approx(getattr(agent.reward_scaler.stats, name))
    for a, b in zip(agent.model.parameters(), other.model.parameters()):
        assert torch.equal(a, b)
    first = next(agent.model.parameters())
    loaded = next(other.model.parameters())
    assert torch.equal(agent.optimizer.state[first]["exp_avg"], other.optimizer.state[loaded]["exp_avg"])
    # The observation in hand is rescaled with the loaded statistics.
    assert other.obs == pytest.approx(other.normalizer(other.raw_obs, update=False))


def test_ppo_learns_to_steer_a_point_to_its_target() -> None:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.backends.mkldnn.flags(enabled=False):  # as in scripts/train.py: much faster on tiny layers
            agent = Ppo(PointToTarget(TOY_ENVS, seed=0), toy_config())
            agent.update(agent.collect())
            before = float(np.mean(agent.finished_returns))
            for _ in range(TOY_UPDATES):
                stats = agent.update(agent.collect())
    finally:
        torch.set_num_threads(threads)
    after = float(np.mean(list(agent.finished_returns)[-TOY_ENVS * 4 :]))
    # Standing still from a random start scores about -15; flying straight to the target about -3.
    assert after > before + 8.0, (before, after)
    # The actor improves even with an untrained critic, so check the critic separately (about 0.6 here).
    assert stats["explained_variance"] > 0.3, stats
