"""Run every fixture crop (tests/fixtures) through the pipeline into one map export.

Usage: uv run python scripts/export_fixtures.py [--data data]
Writes its unit tables (one per crop) and patches to {data}/fixtures/, apart from a real
run's, and the export to {data}/public_export/ (tiles.json and one boulders file per crop with
the exported candidates' patches), which `pnpm -C web dev` serves from data/. Replaces
the map's export only; a run's export comes back with `uv run python -m scoring.export_geojson`.
Every crop's candidates scoring at least 2.0 are exported, with no height or area cutoff.
The web tests export into a temp --data directory.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from pipeline.run_unit import main as run_unit  # noqa: E402
from scoring.export_geojson import main as export_geojson  # noqa: E402
from tests.real_fixtures import BOULDERS, FIXTURES, NEGATIVES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    data = parser.parse_args(argv).data
    export = data / "public_export"
    work = data / "fixtures"
    shutil.rmtree(work, ignore_errors=True)
    units, patches = work / "units", work / "patches"  # export_geojson's defaults beside units
    for site in BOULDERS + NEGATIVES:
        run_unit(
            [str(FIXTURES / site["crops"][0]["file"]),
             "--out", str(units / f"{site['id']}.parquet"), "--patches", str(patches)]
        )  # fmt: skip
    export_geojson(
        ["--candidates", str(units),
         "--out", str(export), "--min-height", "0", "--min-area", "0"]
    )  # fmt: skip
    print(
        f"{len(BOULDERS + NEGATIVES)} crops, {pq.read_table(units).num_rows} candidates -> {export}"
    )


if __name__ == "__main__":
    main()
