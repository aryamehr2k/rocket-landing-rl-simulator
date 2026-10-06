"""HTTP server of the dashboard: the page's files and the JSON and event-stream routes.

GET  /                        the page (dashboard/index.html) and its .js and .css files
GET  /api/options             file lists, defaults and limits for the forms
GET  /api/fly/status          state of the live flight
GET  /api/fly/stream          the live flight as Server-Sent Events: setup, frame, ..., result
POST /api/fly/start           launch a live flight with the form values and a playback speed
POST /api/fly/control         wind, push, speed or pause during the flight
POST /api/fly/stop            stop the live flight
POST /api/batch               fly many flights in the background; returns the job id
GET  /api/batch/latest        the most recent batch job
GET  /api/batch/<id>          progress of a batch job, and its results when done
GET  /api/training/runs       run folders with their state, and the trainings started here
GET  /api/training/run?path=  one run's progress series and the end of its console log
POST /api/training/start      start scripts/train.py in the background
POST /api/training/stop       stop a training started here
GET  /api/models              exported models and finished hop runs
"""

import json
import re
import sys
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from rocketsim.dashboard import catalog
from rocketsim.dashboard.batch import MAX_FLIGHTS, BatchRunner, BusyError
from rocketsim.dashboard.flightsetup import SetupError, parse_setup
from rocketsim.dashboard.jsonsafe import dumps
from rocketsim.dashboard.live import AS_FAST_AS_POSSIBLE, LiveError, LiveFlight, parse_speed
from rocketsim.dashboard.paths import PathError, ProjectPaths
from rocketsim.dashboard.training import TrainingError, TrainingManager

MAX_BODY_BYTES = 64 * 1024
INDEX = "index.html"
STATIC_FILE = re.compile(r"^[a-z0-9_-]+\.(html|js|css)$")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
}
BATCH_PREFIX = "/api/batch/"


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class DashboardApp:
    """Everything the routes share: the folders, the live flight, the batch jobs and the trainings."""

    def __init__(self, paths: ProjectPaths, batch_processes: int | None = None) -> None:
        self.paths = paths
        self.live = LiveFlight()
        self.batches = BatchRunner(batch_processes)
        self.trainings = TrainingManager(paths)

    def close(self) -> None:
        self.live.stop(wait=True)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: DashboardApp) -> None:
        super().__init__(address, DashboardHandler)
        self.app = app


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        routes: dict[str, Callable[[dict[str, list[str]]], Any]] = {
            "/api/options": self._options,
            "/api/fly/status": lambda query: self.app.live.status(),
            "/api/batch/latest": lambda query: self.app.batches.latest_job() or {"state": "none"},
            "/api/training/runs": self._training_runs,
            "/api/training/run": self._training_run,
            "/api/models": lambda query: {"models": catalog.models(self.app.paths)},
        }
        if url.path == "/api/fly/stream":
            self._stream()
        elif url.path in routes:
            self._answer(lambda: routes[url.path](parse_qs(url.query)))
        elif url.path.startswith(BATCH_PREFIX):
            self._answer(lambda: self._batch_job(url.path[len(BATCH_PREFIX):]))
        else:
            self._static(url.path)

    def do_POST(self) -> None:
        routes: dict[str, Callable[[dict[str, Any]], Any]] = {
            "/api/fly/start": self._fly_start,
            "/api/fly/control": lambda body: self.app.live.control(body),
            "/api/fly/stop": lambda body: {"stopped": self.app.live.stop()},
            "/api/batch": self._batch_start,
            "/api/training/start": self._training_start,
            "/api/training/stop": lambda body: {"stopped": self.app.trainings.stop(str(body.get("name", "")))},
        }
        path = urlsplit(self.path).path
        if path not in routes:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"no route {path}"})
            return
        self._answer(lambda: routes[path](self._body()))

    @property
    def app(self) -> DashboardApp:
        return self.server.app

    def _options(self, query: dict[str, list[str]]) -> dict[str, Any]:
        return catalog.options(self.app.paths, self.app.batches.processes, MAX_FLIGHTS)

    def _fly_start(self, body: dict[str, Any]) -> dict[str, Any]:
        setup = parse_setup(body, self.app.paths)
        return self.app.live.start(setup, parse_speed(body.get("speed", AS_FAST_AS_POSSIBLE)))

    def _batch_start(self, body: dict[str, Any]) -> dict[str, Any]:
        setup = parse_setup(body, self.app.paths)
        flights = body.get("flights", 1)
        if isinstance(flights, bool) or not isinstance(flights, int):
            raise ApiError(HTTPStatus.BAD_REQUEST, "flights must be a whole number")
        job = self.app.batches.submit(setup, flights, bool(body.get("compare_pid")), bool(body.get("random_errors")))
        return {"id": job}

    def _batch_job(self, job_id: str) -> dict[str, Any]:
        job = self.app.batches.job(job_id)
        if job is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"no batch job {job_id!r}")
        return job

    def _training_runs(self, query: dict[str, list[str]]) -> dict[str, Any]:
        trainings = self.app.trainings
        return {"runs": trainings.runs(), "started_here": trainings.started_here()}

    def _training_run(self, query: dict[str, list[str]]) -> dict[str, Any]:
        folder = self.app.paths.resolve_inside(query.get("path", [""])[0], self.app.paths.runs)
        return self.app.trainings.run_detail(folder)

    def _training_start(self, body: dict[str, Any]) -> dict[str, Any]:
        training = self.app.paths.resolve_inside(str(body.get("training", "")), self.app.paths.training_files)
        timesteps = body.get("timesteps")
        if timesteps in (None, ""):
            timesteps = None
        elif isinstance(timesteps, bool) or not isinstance(timesteps, int):
            raise ApiError(HTTPStatus.BAD_REQUEST, "timesteps must be a whole number")
        return self.app.trainings.start(training, str(body.get("name", "")), timesteps)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"request body is not JSON: {error}") from error
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
        return body

    def _answer(self, compute: Callable[[], Any]) -> None:
        try:
            self._send_json(HTTPStatus.OK, compute())
        except ApiError as error:
            self._send_json(error.status, {"error": str(error)})
        except BusyError as error:
            self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
        except (SetupError, PathError, LiveError, TrainingError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:  # the page shows the message instead of a dropped connection
            print(f"dashboard: {self.command} {self.path} failed: {error!r}", file=sys.stderr)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})

    def _send_json(self, status: HTTPStatus, value: Any) -> None:
        self._send(status, dumps(value).encode(), "application/json")

    def _send(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, path: str) -> None:
        name = INDEX if path == "/" else path.lstrip("/")
        file = self.app.paths.static / name
        if not STATIC_FILE.match(name) or not file.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"no page {path}"})
            return
        self._send(HTTPStatus.OK, file.read_bytes(), CONTENT_TYPES[file.suffix])

    def _stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for event, data in self.app.live.follow():
                chunk = ": ping\n\n" if event == "ping" else f"event: {event}\ndata: {dumps(data)}\n\n"
                self.wfile.write(chunk.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page closed the stream

    def log_message(self, message_format: str, *args: Any) -> None:
        """Requests are not logged; failures are printed by _answer."""


def make_server(app: DashboardApp, host: str, port: int) -> DashboardServer:
    """A server bound to host:port (port 0 picks a free one) that has not started serving yet."""
    return DashboardServer((host, port), app)
