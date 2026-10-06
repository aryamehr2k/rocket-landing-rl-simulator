"""The live flight: one simulation at a time in a background thread, paced to the wall clock.

The thread runs control steps until the simulated time catches up with the wall clock times the
playback speed, then records a frame for the browser, about FRAME_RATE_HZ times a second. The
simulator is only a few times faster than real time, so at a speed it cannot keep up with it
runs as fast as it can instead of falling further and further behind. Wind, pushes, speed and
pause arrive from other threads and take the same lock as the stepping, so they land between
two control steps.
"""

import math
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from rocketsim.dashboard.flightsetup import (
    NUMBERS, FlightSetup, PreparedFlight, control_step, describe, new_simulation, prepare,
)
from rocketsim.hop.simulation import HopSimulation
from rocketsim.units import deg_to_rad

FRAME_RATE_HZ = 25.0
SPEEDS = (0.5, 1.0, 2.0, 4.0)
AS_FAST_AS_POSSIBLE = "max"
PUSH_SPEED_MPS = 1.0
PUSH_DIRECTIONS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}
MAX_LAG_S = 0.5  # simulated seconds behind the clock before the pace starts again from now
RATE_WINDOW = 25  # frames the achieved speed is measured over
KEEPALIVE_S = 10.0
JOIN_TIMEOUT_S = 5.0


class LiveError(ValueError):
    """A control request that does not fit the current flight."""


@dataclass
class _Flight:
    number: int
    prepared: PreparedFlight
    simulation: HopSimulation
    info: dict[str, Any]
    speed: float | None
    paused: bool = False
    stop_requested: bool = False
    stopped: bool = False
    finished: bool = False
    error: str | None = None
    result: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)
    anchor_wall: float = 0.0
    anchor_sim: float = 0.0
    recent: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=RATE_WINDOW))


def parse_speed(raw: Any) -> float | None:
    """A playback speed from the browser: one of SPEEDS, or None for as fast as possible."""
    if raw == AS_FAST_AS_POSSIBLE:
        return None
    try:
        speed = float(raw)
    except (TypeError, ValueError) as error:
        raise LiveError(f"speed must be one of {SPEEDS} or {AS_FAST_AS_POSSIBLE!r}") from error
    if speed not in SPEEDS:
        raise LiveError(f"speed must be one of {SPEEDS} or {AS_FAST_AS_POSSIBLE!r}")
    return speed


class LiveFlight:
    def __init__(self) -> None:
        self.changed = threading.Condition(threading.Lock())
        self.start_lock = threading.Lock()
        self.flight: _Flight | None = None
        self.thread: threading.Thread | None = None
        self.count = 0

    def start(self, setup: FlightSetup, speed: float | None) -> dict[str, Any]:
        """Stop any running flight and launch a new one. Raises SetupError for a bad setup."""
        prepared = prepare(setup)
        simulation = new_simulation(prepared, setup.seed)
        with self.start_lock:
            self.stop(wait=True)
            with self.changed:
                self.count += 1
                info = {"id": self.count, "speed": speed, **describe(prepared, setup)}
                self.flight = _Flight(self.count, prepared, simulation, info, speed)
                flight = self.flight
            self.thread = threading.Thread(target=self._run, args=(flight,), name=f"live-flight-{flight.number}", daemon=True)
            self.thread.start()
        return info

    def stop(self, wait: bool = False) -> bool:
        """Ask the running flight to stop. Returns False when nothing was flying."""
        with self.changed:
            flight = self.flight
            if flight is None or flight.finished:
                return False
            flight.stop_requested = True
            self.changed.notify_all()
        if wait and self.thread is not None:
            self.thread.join(JOIN_TIMEOUT_S)
        return True

    def control(self, body: dict[str, Any]) -> dict[str, Any]:
        """Change the wind, push the vehicle, change the speed or pause, during a flight."""
        with self.changed:
            flight = self.flight
            if flight is None or flight.finished:
                raise LiveError("no flight is running")
            simulation = flight.simulation
            if "wind" in body:
                speed, direction, gust = _wind(body["wind"])
                simulation.set_wind(speed, deg_to_rad(direction), gust)
                flight.info["wind"] = {"speed": speed, "direction_deg": direction, "gust_std": gust}
            if "push" in body:
                if body["push"] not in PUSH_DIRECTIONS:
                    raise LiveError(f"push must be one of {sorted(PUSH_DIRECTIONS)}")
                simulation.push(PUSH_SPEED_MPS * np.array(PUSH_DIRECTIONS[body["push"]]))
            if "speed" in body:
                flight.speed = parse_speed(body["speed"])
                flight.info["speed"] = flight.speed
            if "paused" in body:
                flight.paused = bool(body["paused"])
            self._anchor(flight)
            self.changed.notify_all()
            return self._state(flight)

    def status(self) -> dict[str, Any]:
        with self.changed:
            if self.flight is None:
                return {"state": "idle"}
            return {**self._state(self.flight), "setup": self.flight.info, "result": self.flight.result}

    def follow(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """The current flight as events: its setup, every frame so far and to come, then its result.

        Yields ("ping", {}) when nothing happened for a while, so the caller can keep the
        connection alive, and ("idle", {}) alone when no flight has been started.
        """
        with self.changed:
            flight = self.flight
        if flight is None:
            yield "idle", {}
            return
        yield "setup", flight.info
        sent = 0
        while True:
            with self.changed:
                self.changed.wait_for(lambda: len(flight.frames) > sent or flight.finished, timeout=KEEPALIVE_S)
                new = flight.frames[sent:]
                finished = flight.finished
            sent += len(new)
            for frame in new:
                yield "frame", frame
            if finished:
                yield "result", flight.result or {}
                return
            if not new:
                yield "ping", {}

    def _state(self, flight: _Flight) -> dict[str, Any]:
        state = "finished" if flight.finished else "paused" if flight.paused else "running"
        speed = flight.speed if flight.speed is not None else AS_FAST_AS_POSSIBLE
        return {"state": state, "id": flight.number, "speed": speed, "paused": flight.paused, "frames": len(flight.frames)}

    def _run(self, flight: _Flight) -> None:
        try:
            self._fly(flight)
        except Exception as error:  # a crashed flight is reported to the browser instead of vanishing
            flight.error = f"{type(error).__name__}: {error}"
        finally:
            with self.changed:
                flight.result = self._result(flight)
                flight.finished = True
                self.changed.notify_all()

    def _fly(self, flight: _Flight) -> None:
        period = 1.0 / FRAME_RATE_HZ
        simulation = flight.simulation
        with self.changed:
            self._anchor(flight)
            self._record(flight)
        while True:
            deadline = time.monotonic() + period
            with self.changed:
                if flight.stop_requested:
                    flight.stopped = True
                    return
                if flight.paused:
                    self.changed.wait(KEEPALIVE_S)
                    continue
                target = self._target_time(flight)
            while not simulation.done and simulation.t < target and time.monotonic() < deadline:
                with self.changed:
                    control_step(flight.prepared, simulation)
            with self.changed:
                if target - simulation.t > MAX_LAG_S and flight.speed is not None:
                    self._anchor(flight)
                self._record(flight)
                if simulation.done:
                    return
            time.sleep(max(0.0, deadline - time.monotonic()))

    def _anchor(self, flight: _Flight) -> None:
        flight.anchor_wall = time.monotonic()
        flight.anchor_sim = flight.simulation.t

    def _target_time(self, flight: _Flight) -> float:
        if flight.speed is None:
            return math.inf
        return flight.anchor_sim + (time.monotonic() - flight.anchor_wall) * flight.speed

    def _record(self, flight: _Flight) -> None:
        frame = flight.simulation.frame()
        now = time.monotonic()
        flight.recent.append((now, flight.simulation.t))
        (wall0, sim0), (wall1, sim1) = flight.recent[0], flight.recent[-1]
        frame["rate"] = round((sim1 - sim0) / (wall1 - wall0), 2) if wall1 > wall0 else None
        flight.frames.append(frame)
        self.changed.notify_all()

    def _result(self, flight: _Flight) -> dict[str, Any]:
        simulation = flight.simulation
        result = simulation.result
        timeout = simulation.flight.touchdown is None and not result.aborted and not flight.stopped and flight.error is None
        return {
            **asdict(result), "success": result.success, "summary": result.summary(), "flight_time": simulation.t,
            "abort_reason": simulation.computer.abort_reason, "stopped": flight.stopped, "timeout": timeout,
            "error": flight.error,
        }


def _wind(raw: Any) -> tuple[float, float, float]:
    if not isinstance(raw, dict):
        raise LiveError("wind must be an object with speed, direction_deg and gust_std")
    values = []
    for key, name in (("speed", "wind_speed"), ("direction_deg", "wind_direction_deg"), ("gust_std", "gust_std")):
        _, low, high = NUMBERS[name]
        try:
            value = float(raw.get(key, 0.0))
        except (TypeError, ValueError) as error:
            raise LiveError(f"wind {key} must be a number") from error
        if not (math.isfinite(value) and low <= value <= high):
            raise LiveError(f"wind {key} must be between {low:g} and {high:g}")
        values.append(value)
    return values[0], values[1], values[2]

