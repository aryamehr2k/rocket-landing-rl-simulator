"""The training environment: many flights at once, each contributing its two plane views.

Sub-environment 2k is the pitch plane of flight k and 2k+1 its yaw plane, so one policy learns from
both planes. Flights can be spread over worker processes; the curriculum level reaches all of them.
"""

import multiprocessing as mp
import os
import signal
from multiprocessing.connection import Connection
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from rocketsim.curriculum import CurriculumLevel
from rocketsim.env import ACTION_HIGH, ACTION_LOW, ACTION_SIZE
from rocketsim.observation import PLANES, ObservationBuilder

PLANES_PER_FLIGHT = len(PLANES)
PARENT_CHECK_S = 5.0
WORKER_EXIT_S = 5.0  # how long close() waits for a worker before terminating it
PIPE_ERRORS = (EOFError, OSError)  # a worker that already exited: closed pipe, reset connection


class Episode(Protocol):
    """One flight seen as two plane views: rocketsim.env.PlaneEpisode or rocketsim.hop.env.HopEpisode."""

    observer: ObservationBuilder

    def reset(self, seed: int | None = None) -> np.ndarray: ...

    def step(self, actions: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]: ...

    def set_level(self, level: CurriculumLevel) -> None: ...


EpisodeFactory = Callable[[int], Episode]


def _worker(pipe: Connection, make_episode: EpisodeFactory, indices: Sequence[int]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C stops the trainer, which then closes the workers
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
            pipe.send(_step_flights(episodes, payload))
        elif command == "level":
            for e in episodes:
                e.set_level(payload)
            pipe.send(None)
        elif command == "close":
            pipe.send(None)
            return


def _step_flights(episodes: list[Episode], actions: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]]:
    """Step each flight with its two plane actions; a finished flight starts again at once."""
    per_flight = actions.reshape(len(episodes), PLANES_PER_FLIGHT, -1)
    results = []
    for episode, plane_actions in zip(episodes, per_flight):
        observations, rewards, done, info = episode.step(list(plane_actions))
        if done:
            observations = episode.reset()
        results.append((observations, rewards, done, info))
    return results


class PlanePairVecEnv:
    """n_sims flights as 2 * n_sims plane environments that reset themselves when a flight ends."""

    def __init__(self, make_episode: EpisodeFactory, n_sims: int, workers: int = 1) -> None:
        probe = make_episode(0)
        self.observation_names = probe.observer.names
        self.observation_size = probe.observer.size
        self.action_size = ACTION_SIZE
        self.action_low, self.action_high = ACTION_LOW, ACTION_HIGH
        self.num_envs = PLANES_PER_FLIGHT * n_sims
        self.episodes: list[Episode] = []
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

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """One control step of every flight. A finished flight restarts; its summary is in the info."""
        actions = np.asarray(actions, dtype=np.float32)
        if self.episodes:
            results = _step_flights(self.episodes, actions)
        else:
            for pipe, flights in zip(self.pipes, self.slices):
                pipe.send(("step", actions[PLANES_PER_FLIGHT * flights.start : PLANES_PER_FLIGHT * flights.stop]))
            results = [r for pipe in self.pipes for r in pipe.recv()]
        observations = np.concatenate([r[0] for r in results])
        rewards = np.concatenate([r[1] for r in results]).astype(np.float32)
        dones = np.repeat([r[2] for r in results], PLANES_PER_FLIGHT)
        infos = [r[3] for r in results for _ in PLANES]
        return observations, rewards, dones, infos

    def set_level(self, level: CurriculumLevel) -> None:
        """Forward a curriculum level to every flight; it applies from each flight's next reset."""
        for e in self.episodes:
            e.set_level(level)
        for pipe in self.pipes:
            pipe.send(("level", level))
        for pipe in self.pipes:
            pipe.recv()

    def close(self) -> None:
        """Stop the workers. Safe after an interrupt or a crashed worker, so it never hides the real error."""
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
                if pipe.poll(WORKER_EXIT_S):
                    pipe.recv()
            except PIPE_ERRORS:
                pass
        for process in self.processes:
            process.join(WORKER_EXIT_S)
            if process.is_alive():
                process.terminate()
                process.join()
        self.pipes, self.processes = [], []
