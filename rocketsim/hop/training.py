"""Training file of the electric vehicle task: which vehicle and mission, what the policy sees and is paid for.

The simulation, environment and wind sections of the same file are read by load_sim_config.
The vehicle and mission paths are relative to the training file.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from rocketsim.curriculum import CurriculumStage
from rocketsim.hop.vehicle import HopControllers, load_hop_controllers
from rocketsim.observation import ObservationField
from rocketsim.training_config import EpisodeConfig, PpoConfig, _curriculum, _episode, _observation, _ppo
from rocketsim.yaml_section import ConfigError, Section, read_yaml_mapping

TASK_KEY = "task"
HOP_TASK = "hop"
SOLID_TASK = "solid_landing"
NO_RANGE = (1.0, 1.0)
DEFAULT_RESIDUAL_GIMBAL_DEG = 4.0


@dataclass(frozen=True)
class HopActionConfig:
    throttle_range: float  # network output +-1 moves the throttle this far from the hover throttle
    residual_gimbal: float  # rad, gimbal correction for an output of 1 in residual mode


@dataclass(frozen=True)
class HopRewardConfig:
    height_error_step: float = 0.0
    speed_error_step: float = 0.0
    lateral_position_step: float = 0.0
    lateral_speed_step: float = 0.0
    tilt_step: float = 0.0
    gimbal_change_step: float = 0.0
    throttle_change_step: float = 0.0
    landed: float = 0.0
    mission_success: float = 0.0
    crash: float = 0.0
    timeout: float = 0.0
    touchdown_vertical_speed: float = 0.0
    miss_distance: float = 0.0
    outside_radius: float = 0.0


@dataclass(frozen=True)
class HiddenErrorRanges:
    thrust_scale: tuple[float, float] = NO_RANGE
    dry_mass_offset: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class MissionVariation:
    target_altitude: tuple[float, float] | None = None
    hover_time: tuple[float, float] | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    flights: int = 20
    wind: float = 0.0
    gust_std: float = 0.0
    plots: bool = True
    animation: bool = False


@dataclass(frozen=True)
class HopTrainingConfig:
    vehicle_path: Path
    mission_path: Path
    observation: tuple[ObservationField, ...]
    action: HopActionConfig
    rewards: HopRewardConfig
    curriculum: tuple[CurriculumStage, ...]
    episode: EpisodeConfig
    hidden_errors: HiddenErrorRanges
    mission_variation: MissionVariation
    evaluation: EvaluationConfig
    controllers: HopControllers | None
    ppo: PpoConfig
    source: str


def training_task(path: str | Path) -> str:
    """The `task` named in a training YAML; files without one are the solid rocket landing."""
    data, _ = read_yaml_mapping(path)
    return str(data.get(TASK_KEY, SOLID_TASK))


def load_hop_training_config(path: str | Path) -> HopTrainingConfig:
    data, source = read_yaml_mapping(path)
    root = Section(data, source)
    if data.get(TASK_KEY) != HOP_TASK:
        raise ConfigError(f"{source}: {TASK_KEY} must be {HOP_TASK!r} for the electric vehicle")
    for key in ("vehicle", "mission", "observation", "action", "rewards", "ppo"):
        if not root.has(key):
            raise ConfigError(f"{source}: {key} is required for the electric vehicle task")
    base = Path(source).parent
    return HopTrainingConfig(
        vehicle_path=base / root.string("vehicle"),
        mission_path=base / root.string("mission"),
        observation=_observation(root.sub("observation")),
        action=_action(root.sub("action")),
        rewards=_rewards(root.sub("rewards")),
        curriculum=_curriculum(root.sub("curriculum")) if root.has("curriculum") else (),
        episode=_episode(root.sub("episode")) if root.has("episode") else EpisodeConfig(False, False),
        hidden_errors=_errors(root.sub("hidden_errors")) if root.has("hidden_errors") else HiddenErrorRanges(),
        mission_variation=_variation(root.sub("mission_variation")) if root.has("mission_variation") else MissionVariation(),
        evaluation=_evaluation(root.sub("evaluation")) if root.has("evaluation") else EvaluationConfig(),
        controllers=load_hop_controllers(root.sub("controllers")) if root.has("controllers") else None,
        ppo=_ppo(root.sub("ppo")),
        source=source,
    )


def _action(section: Section) -> HopActionConfig:
    section.only_keys("throttle_range", "residual_gimbal_deg")
    return HopActionConfig(
        throttle_range=section.number("throttle_range", above=0.0, maximum=1.0),
        residual_gimbal=section.number("residual_gimbal_deg", above=0.0, default=DEFAULT_RESIDUAL_GIMBAL_DEG),
    )


def _rewards(section: Section) -> HopRewardConfig:
    names = tuple(f.name for f in fields(HopRewardConfig))
    section.only_keys(*names)
    values: dict[str, Any] = {name: section.number(name) for name in names if section.has(name)}
    return HopRewardConfig(**values)


def _errors(section: Section) -> HiddenErrorRanges:
    section.only_keys("thrust_scale", "dry_mass_g")
    return HiddenErrorRanges(
        thrust_scale=section.pair("thrust_scale") if section.has("thrust_scale") else NO_RANGE,
        dry_mass_offset=section.pair("dry_mass_g") if section.has("dry_mass_g") else (0.0, 0.0),
    )


def _variation(section: Section) -> MissionVariation:
    section.only_keys("target_altitude_m", "hover_time_s")
    return MissionVariation(
        target_altitude=section.pair("target_altitude_m") if section.has("target_altitude_m") else None,
        hover_time=section.pair("hover_time_s") if section.has("hover_time_s") else None,
    )


def _evaluation(section: Section) -> EvaluationConfig:
    section.only_keys("flights", "wind_mps", "gust_mps", "plots", "animation")
    return EvaluationConfig(
        flights=section.integer("flights", minimum=1, default=20),
        wind=section.number("wind_mps", minimum=0.0, default=0.0),
        gust_std=section.number("gust_mps", minimum=0.0, default=0.0),
        plots=section.boolean("plots", default=True),
        animation=section.boolean("animation", default=False),
    )
