"""Build the public demo site from the export's tiles inside an extent.

Usage: uv run python scripts/build_demo.py --bbox X0 Y0 X1 Y1 [--out DIR] [--base PATH]
Builds the map (`vite build`, review mode left out) under the Pages sub-path --base
(/boulder-scout-demo/) into --out (the demo repo's checkout, ../boulder-scout-demo next to this
repo), with the export's tiles whose nominal tile lies in the extent --bbox,
copied unchanged, a tiles.json listing them and the demo repo's README (web/demo/README.md).
Everything in --out but its .git is replaced.
The imagery basemap needs VITE_ARCGIS_KEY in web/.env.local; without it the demo has OSM only.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from pipeline.area import check_extent, tile_corner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bbox", type=float, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1")
    )
    parser.add_argument("--export", type=Path, default=ROOT / "data" / "public_export")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "boulder-scout-demo")
    parser.add_argument("--base", default="/boulder-scout-demo/")
    args = parser.parse_args(argv)
    try:
        x0, y0, x1, y1 = check_extent(tuple(args.bbox))
    except ValueError as e:
        sys.exit(str(e))

    tiles = json.loads((args.export / "tiles.json").read_text())
    chosen = [
        t
        for t in tiles
        if x0 <= tile_corner(t["tile"])[0] < x1 and y0 <= tile_corner(t["tile"])[1] < y1
    ]
    if not chosen:
        sys.exit(f"no tile of {args.export} lies in {args.bbox}")
    with tempfile.TemporaryDirectory() as public:
        (Path(public) / "tiles").mkdir()
        (Path(public) / "tiles.json").write_text(json.dumps(chosen))
        for t in chosen:
            for f in (args.export / "tiles").glob(f"{t['tile']}.*.bin"):
                shutil.copyfile(f, Path(public) / "tiles" / f.name)
        pnpm = shutil.which("pnpm") or "pnpm"
        build = [pnpm, "-C", str(ROOT / "web"), "build", "--base", args.base]
        build += ["--outDir", str(args.out.resolve()), "--emptyOutDir"]
        subprocess.run(build, env={**os.environ, "EXPORT_DIR": public}, check=True)
    shutil.copyfile(ROOT / "web" / "demo" / "README.md", args.out / "README.md")
    (args.out / ".nojekyll").touch()  # Pages serves the files as they are, no Jekyll
    print(f"{len(chosen)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
