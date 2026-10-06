import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from rocketsim.dashboard import catalog
from rocketsim.dashboard.batch import BatchRunner
from rocketsim.dashboard.flightsetup import SetupError, parse_setup
from rocketsim.dashboard.jsonsafe import dumps
from rocketsim.dashboard.live import LiveFlight
from rocketsim.dashboard.paths import PathError, ProjectPaths
from rocketsim.dashboard.server import DashboardApp, make_server
from rocketsim.dashboard.training import RUNNING_WINDOW_S, TrainingError, TrainingManager, read_progress
from tests.conftest import REPO_ROOT

SHORT_HOP = {"target_altitude": 5, "hover_time": 1, "climb_speed": 2, "descent_speed": 2}
BATCH_TIMEOUT_S = 60.0
POLL_S = 0.2
PROGRESS = """time/fps,time/total_timesteps,flights/mission_success_rate,flights/landed_rate,curriculum/wind_max,flights/mean_miss
400,16384,0.0,0.25,0.0,4.5
420,32768,0.1,0.5,,3.0
410,49152,0.3,0.75,3.0,2.0
"""


@pytest.fixture
def paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths.for_project(REPO_ROOT, runs=tmp_path / "runs", models=tmp_path / "models")


def test_options_list_the_project_files(paths: ProjectPaths) -> None:
    options = catalog.options(paths, batch_processes=2, max_flights=100)
    vehicles = {v["path"]: v for v in options["vehicles"]}
    assert vehicles["configs/vehicles/electric_hopper.yaml"]["geometry"]["leg_count"] == 4
    assert "configs/missions/hop_50m.yaml" in [m["path"] for m in options["missions"]]
    assert "configs/training/hop.yaml" in [w["path"] for w in options["worlds"]]
    tasks = {t["name"]: t["task"] for t in options["trainings"]}
    assert tasks["hop"] == "hop" and tasks["default"] == "solid_landing"
    assert options["models"] == []
    json.loads(dumps(options))


def test_paths_outside_the_project_folders_are_refused(paths: ProjectPaths) -> None:
    for bad in ("../../etc/passwd", "configs/../../etc/passwd", "/etc/passwd", "runs/../../../etc/passwd", ""):
        with pytest.raises(PathError):
            paths.resolve(bad)
    assert paths.resolve("configs/missions/hop_50m.yaml").name == "hop_50m.yaml"
    with pytest.raises(SetupError):
        parse_setup({"vehicle": "../../etc/passwd"}, paths)
    with pytest.raises(SetupError):
        parse_setup({"controller": "configs/training"}, paths)
    with pytest.raises(SetupError):
        parse_setup({"wind_speed": 500}, paths)


def test_json_has_no_nan() -> None:
    assert json.loads(dumps({"speed": float("nan"), "values": [1.0, float("inf")]})) == {"speed": None, "values": [1.0, None]}


def test_live_flight_streams_frames_until_the_result(paths: ProjectPaths) -> None:
    live = LiveFlight()
    info = live.start(parse_setup({**SHORT_HOP, "wind_speed": 2.0}, paths), speed=None)
    assert info["mission"]["target_altitude"] == 5.0 and info["controller"] == "PID"
    events = []
    for name, data in live.follow():
        events.append((name, data))
        if name == "frame" and len(events) == 3:
            assert live.control({"push": "+x", "wind": {"speed": 3.0, "direction_deg": 90.0, "gust_std": 0.5}})["state"] == "running"
    names = [name for name, _ in events]
    assert names[0] == "setup" and names[-1] == "result"
    times = [data["t"] for name, data in events if name == "frame"]
    assert len(times) > 2 and times == sorted(times)
    result = events[-1][1]
    assert result["landed"] and not result["stopped"] and result["error"] is None
    assert live.status()["state"] == "finished"


def test_live_flight_can_pause_and_stop(paths: ProjectPaths) -> None:
    live = LiveFlight()
    live.start(parse_setup(SHORT_HOP, paths), speed=1.0)
    assert live.control({"paused": True})["paused"]
    assert live.stop(wait=True)
    result = live.status()["result"]
    assert result["stopped"] and not result["landed"]


def test_batch_of_two_pid_flights(paths: ProjectPaths) -> None:
    runner = BatchRunner(processes=2)
    job_id = runner.submit(parse_setup({**SHORT_HOP, "seed": 5}, paths), flights=2, compare_pid=False, random_errors=True)
    deadline = time.monotonic() + BATCH_TIMEOUT_S
    while (job := runner.job(job_id))["state"] == "running" and time.monotonic() < deadline:
        time.sleep(POLL_S)
    assert job["state"] == "done", job["error"]
    pid = job["results"]["PID"]
    assert pid["summary"]["flights"] == 2
    assert [f["seed"] for f in pid["flights"]] == [5, 6]
    path = pid["flights"][0]["path"]
    assert len(path["t"]) == len(path["x"]) == len(path["h"]) > 2
    json.loads(dumps(job))


def make_run(runs: Path, folder: str, age_s: float, finished: bool) -> Path:
    run = runs / folder
    run.mkdir(parents=True)
    (run / "progress.csv").write_text(PROGRESS)
    if finished:
        (run / "summary.json").write_text(json.dumps({"timesteps": 49152}))
    stamp = time.time() - age_s
    os.utime(run / "progress.csv", (stamp, stamp))
    return run


def test_training_runs_are_listed_from_their_progress_logs(paths: ProjectPaths) -> None:
    make_run(paths.runs, "20260101_120000_old", age_s=10 * RUNNING_WINDOW_S, finished=False)
    make_run(paths.runs, "20260102_120000_done", age_s=10 * RUNNING_WINDOW_S, finished=True)
    live = make_run(paths.runs, "20260103_120000_live", age_s=1.0, finished=False)
    paths.train_logs.mkdir()
    (paths.train_logs / "live.log").write_text("line 1\nline 2\n")
    manager = TrainingManager(paths)
    runs = manager.runs()
    assert [r["name"] for r in runs] == ["live", "done", "old"]
    assert [r["state"] for r in runs] == ["running", "finished", "stopped"]
    assert runs[0]["timesteps"] == 49152 and runs[0]["success"] == 0.3 and runs[0]["fps"] == 410
    series = read_progress(live / "progress.csv")
    assert series["wind"] == [0.0, 0.0, 3.0]  # the empty cell keeps the last curriculum value
    assert series["reward"] == [None, None, None]
    detail = manager.run_detail(live)
    assert detail["log"].endswith("line 2")


def test_training_start_checks_its_inputs(paths: ProjectPaths) -> None:
    manager = TrainingManager(paths)
    hop = paths.configs / "training" / "hop.yaml"
    with pytest.raises(TrainingError):
        manager.start(hop, "../escape", None)
    with pytest.raises(TrainingError):
        manager.start(paths.configs / "missions" / "hop_50m.yaml", "fine_name", None)
    with pytest.raises(TrainingError):
        manager.start(hop, "fine_name", 0)
    assert manager.processes == {}


def test_models_include_hop_runs_with_a_policy(paths: ProjectPaths) -> None:
    model = paths.models / "hop_v1"
    (model / "configs" / "training").mkdir(parents=True)
    shutil.copy(paths.configs / "training" / "hop.yaml", model / "configs" / "training" / "hop.yaml")
    (model / "policy.npz").write_bytes(b"")
    (model / "model.json").write_text(json.dumps({"inputs": [{"index": 0, "name": "height"}]}))
    (paths.models / "unfinished").mkdir()
    found = catalog.models(paths)
    assert [m["path"] for m in found] == ["models/hop_v1"]
    assert found[0]["info"]["inputs"][0]["name"] == "height"


def get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_server_routes(paths: ProjectPaths) -> None:
    server = make_server(DashboardApp(paths, batch_processes=1), "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert get(f"{base}/api/options")[0] == 200
        assert get(f"{base}/api/fly/status") == (200, {"state": "idle"})
        assert get(f"{base}/api/training/run?path=../../etc/passwd")[0] == 400
        assert get(f"{base}/api/batch/unknown")[0] == 404
        assert get(f"{base}/../../etc/passwd")[0] == 404
        with urllib.request.urlopen(f"{base}/", timeout=10) as response:
            assert b"Rocket control pad" in response.read()
        request = urllib.request.Request(f"{base}/api/fly/start", data=b'{"speed": "fast"}', method="POST")
        with pytest.raises(urllib.error.HTTPError) as answer:
            urllib.request.urlopen(request, timeout=10)
        assert answer.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
