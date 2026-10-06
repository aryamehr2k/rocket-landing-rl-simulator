"""The training environment: many flights at once, each contributing its two plane views.

A Stable-Baselines3 VecEnv whose sub-environment 2k is the pitch plane of flight k and 2k+1 its
yaw plane, so one policy is trained on both planes with twice the samples per flight. Flights
can be spread over worker processes; the curriculum level is forwarded to all of them.
"""

import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from rocketsim.curriculum import CurriculumLevel
from rocketsim.env import ACTION_HIGH, ACTION_LOW, ACTION_SIZE, PlaneEpisode
from rocketsim.observation import PLANES

PLANES_PER_FLIGHT = len(PLANES)
PARENT_CHECK_S = 5.0
# Any episode with the PlaneEpisode interface: rocketsim.env.PlaneEpisode or rocketsim.hop.env.HopEpisode.
EpisodeFactory = Callable[[int], PlaneEpisode]


def _worker(pipe: Connection, make_episode: EpisodeFactory, indices: Sequence[int]) -> None:
    episodes = [make_episode(i) for i in indices]
    parent = os.getppid()
    while True:
        # A forked worker holds both ends of its pipe, so it would never see the trainer die; check instead.
        if not pipe.poll(PARENT_CHECK_S):
            if os.getppid() != parent:
                return
            continue
        command, payload = pipe.recv()
        if command == "reset":
            pipe.send(np.concatenate([e.reset() for e in episodes]))
        elif command == "step":
            pipe.send([_step_and_reset(e, payload[PLANES_PER_FLIGHT * k : PLANES_PER_FLIGHT * (k + 1)]) for k, e in enumerate(episodes)])
        elif command == "level":
            for e in episodes:
                e.set_level(payload)
            pipe.send(None)
        elif command == "close":
            pipe.send(None)
            return


def _step_and_reset(episode: PlaneEpisode, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
    # At the end of a flight the terminal observation is kept in the info and a new flight starts.
    observations, rewards, done, info = episode.step([actions[0], actions[1]])
    if done:
        info = {**info, "terminal_observation": observations}
        observations = episode.reset()
    return observations, rewards, done, info


class PlanePairVecEnv(VecEnv):
    def __init__(self, make_episode: EpisodeFactory, n_sims: int, workers: int = 1) -> None:
        probe = make_episode(0)
        size = probe.observer.size
        self.observation_names = probe.observer.names
        super().__init__(
            PLANES_PER_FLIGHT * n_sims,
            gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32),
            gym.spaces.Box(ACTION_LOW, ACTION_HIGH, shape=(ACTION_SIZE,), dtype=np.float32),
        )
        self.render_mode = None
        self.n_sims = n_sims
        self.level = CurriculumLevel()
        self._actions: np.ndarray | None = None
        self.episodes: list[PlaneEpisode] = []
        self.pipes: list[Connection] = []
        self.processes: list[mp.Process] = []
        self.slices: list[range] = []
        if workers <= 1:
            self.episodes = [probe] + [make_episode(i) for i in range(1, n_sims)]
            return
        context = mp.get_context("fork")
        bounds = np.linspace(0, n_sims, min(workers, n_sims) + 1).astype(int)
        for start, end in zip(bounds[:-1], bounds[1:]):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, make_episode, range(start, end)), daemon=True)
            process.start()
            self.pipes.append(parent)
            self.processes.append(process)
            self.slices.append(range(start, end))

    def reset(self) -> np.ndarray:
        if self.episodes:
            return np.concatenate([e.reset() for e in self.episodes])
        for pipe in self.pipes:
            pipe.send(("reset", None))
        return np.concatenate([pipe.recv() for pipe in self.pipes])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        assert self._actions is not None
        if self.episodes:
            results = [_step_and_reset(e, self._actions[PLANES_PER_FLIGHT * k : PLANES_PER_FLIGHT * (k + 1)]) for k, e in enumerate(self.episodes)]
        else:
            for pipe, flights in zip(self.pipes, self.slices):
                pipe.send(("step", self._actions[PLANES_PER_FLIGHT * flights.start : PLANES_PER_FLIGHT * flights.stop]))
            results = [r for pipe in self.pipes for r in pipe.recv()]
        observations = np.concatenate([r[0] for r in results])
        rewards = np.concatenate([r[1] for r in results]).astype(np.float32)
        dones = np.repeat([r[2] for r in results], PLANES_PER_FLIGHT)
        infos = [self._plane_info(r[3], plane) for r in results for plane in PLANES]
        return observations, rewards, dones, infos

    @staticmethod
    def _plane_info(info: dict[str, Any], plane: int) -> dict[str, Any]:
        if "terminal_observation" in info:
            return {**info, "terminal_observation": info["terminal_observation"][plane]}
        return info

    def set_level(self, level: CurriculumLevel) -> None:
        """Forward a curriculum level to every flight; it applies from each flight's next reset."""
        self.level = level
        for e in self.episodes:
            e.set_level(level)
        for pipe in self.pipes:
            pipe.send(("level", level))
        for pipe in self.pipes:
            pipe.recv()

    def close(self) -> None:
        for pipe in self.pipes:
            pipe.send(("close", None))
        for pipe in self.pipes:
            pipe.recv()
        for process in self.processes:
            process.join()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        return [getattr(self, attr_name, None)] * self._count(indices)

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> list[Any]:
        raise NotImplementedError("PlanePairVecEnv has no per-plane methods")

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        return [False] * self._count(indices)

    def seed(self, seed: int | None = None) -> list[int | None]:
        return [seed] * self.num_envs

    def _count(self, indices: Any) -> int:
        if indices is None:
            return self.num_envs
        return 1 if isinstance(indices, int) else len(indices)
