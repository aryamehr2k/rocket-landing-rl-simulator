"""Serve the project over HTTP and print the viewer link for a flight log; on a remote machine, forward the port.

Usage: python scripts/serve_viewer.py runs/flight.csv --port 8000 (then VS Code or ssh -L 8000:localhost:8000)
"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
VIEWER = "viewer/flight_viewer.html"
DEFAULT_ROCKET = "configs/rockets/example_tvc.yaml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def project_relative(path: str) -> str:
    """Path relative to the project root, as the browser will request it."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise SystemExit(f"{path} is outside the project folder {ROOT}, so the server cannot see it")


def viewer_url(port: int, log: str, rocket: str) -> str:
    """The viewer page with the files given relative to the page itself."""
    query = urlencode({"log": "../" + project_relative(log), "rocket": "../" + project_relative(rocket)}, safe="/")
    return f"http://localhost:{port}/{VIEWER}?{query}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="flight log CSV")
    parser.add_argument("--rocket", default=DEFAULT_ROCKET, help="rocket YAML that gives the drawn geometry")
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to listen on; 0.0.0.0 opens it to the network")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if not Path(args.log).is_file():
        raise SystemExit(f"{args.log} does not exist; fly something first with scripts/fly_scripted.py")
    if not Path(args.rocket).is_file():
        raise SystemExit(f"{args.rocket} does not exist; --rocket needs a rocket YAML")
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"serving {ROOT} on port {args.port}; press Ctrl+C to stop", flush=True)
        print(f"open {viewer_url(args.port, args.log, args.rocket)}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopped")


if __name__ == "__main__":
    main()
