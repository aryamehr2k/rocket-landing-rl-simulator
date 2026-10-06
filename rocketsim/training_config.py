"""The training sections of a training YAML: observation, action, rewards, curriculum, episode, ppo."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from rocketsim.curriculum import CurriculumLevel, CurriculumStage
from rocketsim.guidance_config import ControllersConfig, load_controllers
from rocketsim.observation import FIELDS, ObservationField
from rocketsim.rewards import RewardConfig
from rocketsim.yaml_section import ConfigError, Section, read_yaml_mapping

ACTIVATIONS = ("tanh", "relu")
DEFAULT_CHECKPOINT_EVERY = 500_000  # samples between saved copies of the network during training
DEFAULT_LOG_STD_INIT = 0.0  # exploration noise of one action unit
DEFAULT_SCALE = 1.0
NEUTRAL_THRESHOLD = 0.0


@dataclass(frozen=True)
class ActionConfig:
    ignite_threshold: float


@dataclass(frozen=True)
class EpisodeConfig:
    random_wind_direction: bool
    random_wind_speed: bool


@dataclass(frozen=True)
class PpoConfig:
    total_timesteps: int
    n_sims: int
    workers: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    hidden_layers: tuple[int, ...]
    activation: str
    seed: int
    log_std_init: float
    checkpoint_every: int


@dataclass(frozen=True)
class TrainingConfig:
    observation: tuple[ObservationField, ...]
    action: ActionConfig
    rewards: RewardConfig
    curriculum: tuple[CurriculumStage, ...]
    episode: EpisodeConfig
    ppo: PpoConfig
    controllers: ControllersConfig | None
    source: str


def load_training_config(path: str | Path) -> TrainingConfig:
    """Read the training sections. The simulation sections of the same file are read by load_sim_config."""
    data, source = read_yaml_mapping(path)
    root = Section(data, source)
    for key in ("observation", "action", "rewards", "ppo"):
        if not root.has(key):
            raise ConfigError(f"{source}: {key} section is required for training")
    return TrainingConfig(
        observation=_observation(root.sub("observation")),
        action=_action(root.sub("action")),
        rewards=_rewards(root.sub("rewards")),
        curriculum=_curriculum(root.sub("curriculum")) if root.has("curriculum") else (),
        episode=_episode(root.sub("episode")) if root.has("episode") else EpisodeConfig(False, False),
        ppo=_ppo(root.sub("ppo")),
        controllers=load_controllers(root.sub("controllers")) if root.has("controllers") else None,
        source=source,
    )


def _observation(section: Section) -> tuple[ObservationField, ...]:
    section.only_keys("fields")
    raw = section.data.get("fields")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{section.where('fields')} must be a non-empty list of {{name, scale}} entries")
    result = []
    for i, entry in enumerate(raw):
        item = Section(entry if isinstance(entry, dict) else {}, section.source, f"{section.path}.fields[{i}]")
        item.only_keys("name", "scale")
        name = item.string("name")
        if name not in FIELDS:
            raise ConfigError(f"{item.where('name')} is not a known observation field; known: {sorted(FIELDS)}")
        result.append(ObservationField(name, item.number("scale", above=0.0, default=DEFAULT_SCALE)))
    return tuple(result)


def _action(section: Section) -> ActionConfig:
    section.only_keys("ignite_threshold")
    return ActionConfig(section.number("ignite_threshold", minimum=-1.0, maximum=1.0, default=NEUTRAL_THRESHOLD))


def _rewards(section: Section) -> RewardConfig:
    names = tuple(f.name for f in fields(RewardConfig))
    section.only_keys(*names)
    values: dict[str, Any] = {name: section.number(name, default=0.0) for name in names if section.has(name)}
    return RewardConfig(**values)


def _curriculum(section: Section) -> tuple[CurriculumStage, ...]:
    section.only_keys("stages")
    raw = section.data.get("stages")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{section.where('stages')} must be a non-empty list")
    stages = []
    last_until = 0.0
    for i, entry in enumerate(raw):
        item = Section(entry if isinstance(entry, dict) else {}, section.source, f"{section.path}.stages[{i}]")
        item.only_keys("until", "wind_mps", "gust_mps", "sensor_noise", "hidden_errors")
        until = item.number("until", minimum=0.0, maximum=1.0)
        if until < last_until:
            raise ConfigError(f"{item.where('until')} must not decrease along the stages")
        last_until = until
        level = CurriculumLevel(
            wind_max=item.number("wind_mps", minimum=0.0) if item.has("wind_mps") else None,
            gust_std=item.number("gust_mps", minimum=0.0) if item.has("gust_mps") else None,
            sensor_noise=item.number("sensor_noise", minimum=0.0, default=1.0),
            hidden_errors=item.number("hidden_errors", minimum=0.0, default=1.0),
        )
        stages.append(CurriculumStage(until, level))
    return tuple(stages)


def _episode(section: Section) -> EpisodeConfig:
    section.only_keys("random_wind_direction", "random_wind_speed")
    return EpisodeConfig(
        section.boolean("random_wind_direction", default=False),
        section.boolean("random_wind_speed", default=False),
    )


def _ppo(section: Section) -> PpoConfig:
    section.only_keys(
        "total_timesteps", "n_sims", "workers", "n_steps", "batch_size", "n_epochs", "learning_rate", "gamma",
        "gae_lambda", "clip_range", "ent_coef", "hidden_layers", "activation", "seed", "log_std_init", "checkpoint_every",
    )
    layers = section.data.get("hidden_layers")
    if not isinstance(layers, list) or not layers or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in layers):
        raise ConfigError(f"{section.where('hidden_layers')} must be a list of positive integers, for example [32, 32]")
    activation = section.string("activation", default="tanh")
    if activation not in ACTIVATIONS:
        raise ConfigError(f"{section.where('activation')} must be one of {ACTIVATIONS}")
    return PpoConfig(
        total_timesteps=section.integer("total_timesteps", minimum=1),
        n_sims=section.integer("n_sims", minimum=1),
        workers=section.integer("workers", minimum=1, default=1),
        n_steps=section.integer("n_steps", minimum=1),
        batch_size=section.integer("batch_size", minimum=1),
        n_epochs=section.integer("n_epochs", minimum=1),
        learning_rate=section.number("learning_rate", above=0.0),
        gamma=section.number("gamma", above=0.0, maximum=1.0),
        gae_lambda=section.number("gae_lambda", minimum=0.0, maximum=1.0),
        clip_range=section.number("clip_range", above=0.0),
        ent_coef=section.number("ent_coef", minimum=0.0, default=0.0),
        hidden_layers=tuple(int(n) for n in layers),
        activation=activation,
        seed=section.integer("seed", minimum=0, default=0),
        log_std_init=section.number("log_std_init", default=DEFAULT_LOG_STD_INIT),
        checkpoint_every=section.integer("checkpoint_every", minimum=1, default=DEFAULT_CHECKPOINT_EVERY),
    )
