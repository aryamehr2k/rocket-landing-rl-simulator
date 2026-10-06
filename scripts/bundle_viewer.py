"""Write one self-contained HTML file: the viewer with a flight log and the rocket geometry inside.

Copy or download that file to any computer and open it in a browser; nothing else is needed.
Usage: python scripts/bundle_viewer.py runs/flight.csv --out runs/flight.html
"""

import argparse
from pathlib import Path

VIEWER = Path(__file__).resolve().parent.parent / "viewer" / "flight_viewer.html"
DEFAULT_ROCKET = "configs/rockets/example_tvc.yaml"
LOG_BLOCK = '<script type="text/csv" id="embeddedLog">'
ROCKET_BLOCK = '<script type="text/yaml" id="embeddedRocket">'
TITLE = "<title>Flight viewer</title>"
FORBIDDEN = ("</script", "<!--")  # would end or escape the block the text is pasted into


def embed(page: str, block: str, source: Path) -> str:
    """Paste the file's text into the empty block and name the block after the file."""
    text = source.read_text()
    lowered = text.lower()
    for token in FORBIDDEN:
        if token in lowered:
            raise SystemExit(f"{source} contains '{token}' and cannot be embedded")
    empty = block + "</script>"
    if empty not in page:
        raise SystemExit(f"{VIEWER} has no {block} block")
    filled = block[:-1] + f' data-name="{source.name}">\n{text}</script>'
    return page.replace(empty, filled, 1)


def bundle(log: Path, rocket: Path) -> str:
    page = VIEWER.read_text()
    page = embed(page, LOG_BLOCK, log)
    page = embed(page, ROCKET_BLOCK, rocket)
    return page.replace(TITLE, f"<title>{log.stem.replace('_', ' ')} flight</title>", 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="flight log CSV")
    parser.add_argument("--rocket", default=DEFAULT_ROCKET, help="rocket YAML that gives the drawn geometry")
    parser.add_argument("--out", help="HTML file to write; defaults to the CSV name with .html")
    args = parser.parse_args()
    log = Path(args.log)
    out = Path(args.out) if args.out else log.with_suffix(".html")
    try:
        out.write_text(bundle(log, Path(args.rocket)))
    except FileNotFoundError as error:
        raise SystemExit(f"{error.filename}: {error.strerror}")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB); open it in any browser")


if __name__ == "__main__":
    main()
