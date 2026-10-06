"""Many flights of one setup on consecutive seeds, spread over a process pool as a background job.

Every flight records a thinned-out path (time, x, y, height) for the browser's plots instead of
a log file. With "compare with PID" the PID flies the same seeds, so the hidden errors, sensor
noise and gusts match flight by flight.
"""

import multiprocessing
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from rocketsim.dashboard.flightsetup import (
    PID_LABEL, FlightSetup, PreparedFlight, SetupError, control_step, describe, new_simulation, prepare,
)
from rocketsim.hop.env import flight_info
from rocketsim.hop.evaluate import draw_errors, summarize
from rocketsim.hop.simulation import HopSimulation

MAX_FLIGHTS = 100
RESERVED_CPUS = 4  # left free for the server, the browser and any training
PATH_EVERY = 10  # control steps between recorded path points
DIGITS = 3
KEEP_JOBS = 10
START_METHOD = "spawn"  # a fresh interpreter per worker; forking a threaded server is not safe

# Each worker process loads the files of a setup once, not once per flight.
_prepared_cache: dict[FlightSetup, PreparedFlight] = {}


class BusyError(RuntimeError):
    """Another batch is still running."""


@dataclass(frozen=True)
class BatchTask:
    setup: FlightSetup
    seed: int
    random_errors: bool


@dataclass
class _Job:
    id: str
    total: int
    controllers: list[str]
    info: dict[str, Any]
    started: float = field(default_factory=time.time)
    flights: list[dict[str, Any]] = field(default_factory=list)
    state: str = "running"
    error: str | None = None
    results: dict[str, Any] | None = None
    seconds: float | None = None


def pool_size(limit: int | None = None) -> int:
    """Worker processes for a batch: every CPU but RESERVED_CPUS, at least one, at most `limit`."""
    size = max(1, (os.cpu_count() or 1) - RESERVED_CPUS)
    return max(1, min(size, limit)) if limit is not None else size


def fly_task(task: BatchTask) -> dict[str, Any]:
    """Fly one seed of a batch (runs in a worker process) and return its summary and path."""
    prepared = _prepared_cache.get(task.setup)
    if prepared is None:
        prepared = _prepared_cache[task.setup] = prepare(task.setup)
    errors = draw_errors(prepared.error_ranges, task.seed) if task.random_errors else None
    simulation = new_simulation(prepared, task.seed, errors)
    path: dict[str, list[float]] = {"t": [], "x": [], "y": [], "h": []}
    steps = 0
    while not simulation.done:
        if steps % PATH_EVERY == 0:
            _record(path, simulation)
        control_step(prepared, simulation)
        steps += 1
    _record(path, simulation)
    simulation.close()
    return {"controller": prepared.controller_name, "seed": task.seed, **flight_info(simulation), "path": path}


def _record(path: dict[str, list[float]], simulation: HopSimulation) -> None:
    position = simulation.flight.y
    path["t"].append(round(simulation.t, DIGITS))
    path["x"].append(round(float(position[0]), DIGITS))
    path["y"].append(round(float(position[1]), DIGITS))
    path["h"].append(round(simulation.feet_height(), DIGITS))


class BatchRunner:
    def __init__(self, processes: int | None = None) -> None:
        self.processes = pool_size(processes)
        self.lock = threading.Lock()
        self.jobs: dict[str, _Job] = {}
        self.latest: str | None = None

    def submit(self, setup: FlightSetup, flights: int, compare_pid: bool, random_errors: bool) -> str:
        """Start a batch in the background and return its job id. Raises BusyError or SetupError."""
        if not 1 <= flights <= MAX_FLIGHTS:
            raise SetupError(f"flights must be from 1 to {MAX_FLIGHTS}")
        prepared = prepare(setup)  # fails here, in the request, if a file is wrong
        setups = [setup]
        if compare_pid and setup.model is not None:
            setups.append(replace(setup, model=None))
        tasks = [BatchTask(s, seed, random_errors) for s in setups for seed in range(setup.seed, setup.seed + flights)]
        controllers = [prepared.controller_name] + ([PID_LABEL] if len(setups) > 1 else [])
        info = {**describe(prepared, setup), "flights": flights, "random_errors": random_errors}
        with self.lock:
            if any(job.state == "running" for job in self.jobs.values()):
                raise BusyError("a batch is already running; wait for it to finish")
            job = _Job(uuid.uuid4().hex[:12], len(tasks), controllers, info)
            self.jobs[job.id] = job
            self.latest = job.id
            self._forget_old_jobs()
        threading.Thread(target=self._run, args=(job, tasks), name=f"batch-{job.id}", daemon=True).start()
        return job.id

    def job(self, job_id: str) -> dict[str, Any] | None:
        """Progress of a job, with the results once it is done; None for an unknown id."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return {
                "id": job.id, "state": job.state, "done": len(job.flights), "total": job.total,
                "controllers": job.controllers, "setup": job.info, "error": job.error, "results": job.results,
                "seconds": job.seconds if job.seconds is not None else round(time.time() - job.started, 1),
                "processes": self.processes,
            }

    def latest_job(self) -> dict[str, Any] | None:
        with self.lock:
            latest = self.latest
        return self.job(latest) if latest is not None else None

    def _run(self, job: _Job, tasks: list[BatchTask]) -> None:
        try:
            context = multiprocessing.get_context(START_METHOD)
            with context.Pool(min(self.processes, len(tasks))) as pool:
                for flight in pool.imap_unordered(fly_task, tasks):
                    with self.lock:
                        job.flights.append(flight)
            results = self._results(job)
            with self.lock:
                job.seconds = round(time.time() - job.started, 1)
                job.results, job.state = results, "done"
        except Exception as error:  # the browser shows the failure instead of a job stuck at "running"
            with self.lock:
                job.seconds = round(time.time() - job.started, 1)
                job.state, job.error = "failed", f"{type(error).__name__}: {error}"

    def _results(self, job: _Job) -> dict[str, Any]:
        results = {}
        for name in job.controllers:
            flights = sorted((f for f in job.flights if f["controller"] == name), key=lambda f: f["seed"])
            results[name] = {"summary": summarize(flights), "flights": flights}
        return results

    def _forget_old_jobs(self) -> None:
        finished = [job_id for job_id, job in self.jobs.items() if job.state != "running"]
        for job_id in finished[:max(0, len(self.jobs) - KEEP_JOBS)]:
            del self.jobs[job_id]
