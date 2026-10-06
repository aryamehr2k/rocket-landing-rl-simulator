"""Control pad in the browser: fly the vehicle live, fly many flights, watch training, pick models.

Usage: python scripts/dashboard.py [--port 8060]; on a remote machine, forward the port (VS Code: Ports panel).
"""

import argparse
from pathlib import Path

from rocketsim.dashboard.paths import ProjectPaths
from rocketsim.dashboard.server import DashboardApp, make_server

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8050
DEFAULT_HOST = "127.0.0.1"


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to listen on; 0.0.0.0 opens it to the network")
    parser.add_argument("--runs", type=Path, help="training runs folder (default: runs/ in the project)")
    parser.add_argument("--models", type=Path, help="models folder (default: models/ in the project)")
    parser.add_argument("--processes", type=int, help="most worker processes for many flights (default: all CPUs but 4)")
    return parser.parse_args()


def main() -> None:
    args = parse()
    paths = ProjectPaths.for_project(PROJECT_ROOT, runs=args.runs, models=args.models)
    app = DashboardApp(paths, batch_processes=args.processes)
    server = make_server(app, args.host, args.port)
    host, port = server.server_address[:2]
    print(f"Dashboard running at http://{host}:{port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        app.close()
        server.server_close()


if __name__ == "__main__":
    main()
