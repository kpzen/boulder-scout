"""Download Lantmäteriet's orthophotos over `--bbox` or `--area` into
data/orthophotos/, for the photo stage (scripts/measure_photos.py).

Which files: a search of the STAC API https://api.lantmateriet.se/stac-bild/v1 (product
*Ortofoto Nedladdning*, CC BY 4.0) for 4-band photos (`spektraltyp` rgbi: red, green, blue, near
infrared) of the last YEARS years at 0.5 m/px or finer, one cloud-optimized GeoTIFF per square,
the regional ones 0.4 m per 5 km, ~460 MB each. All are taken at one resolution, the coarsest on
offer, since the stage cuts every photo at one: towns' finer photos (0.16 m) are left out. Per
square it takes the one flown earliest in the season, for less leaf cover (the newest of those
flown on the same day of the year), for every square that overlaps the extent's 2.5 km tiles (a
candidate whose 40 m crop reaches past them gets no photo). Files already there are skipped, so
an interrupted run resumes.

Run on your own computer, from the repository:
    uv run python scripts/fetch_photos.py              download whatever is missing
    uv run python scripts/fetch_photos.py --plan       list what it would download, then stop
    ... --bbox X0 Y0 X1 Y1 | --area area.geojson       another extent, as scripts/run_area.py
The search is open; the files need Geotorget credentials (as scripts/fetch_area.py) with
Ortofoto Nedladdning ordered on Geotorget, else they answer 403.
"""

import argparse
import datetime
import json
import sys
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.area import Bbox, check_extent, nominal_tiles, polygon_tiles  # noqa: E402
from scripts import fetch_area  # noqa: E402

STAC = "https://api.lantmateriet.se/stac-bild/v1"
PHOTOS = REPO / "data" / "orthophotos"
YEARS = 10  # the oldest photos taken: the author found the newer ones much better


def choose(items: list[dict[str, Any]], tiles: list[Bbox]) -> list[dict[str, Any]]:
    """Per square that overlaps a tile, its item flown earliest in the year (on the same day, the
    newest) at the coarsest resolution of those items, sorted by id."""
    near = []
    for it in items:
        b = it["properties"]["proj:bbox"]
        if it["properties"]["proj:code"] == "EPSG:3006" and any(
            b[0] < t[2] and t[0] < b[2] and b[1] < t[3] and t[1] < b[3] for t in tiles
        ):
            near.append(it)
    coarsest = max((it["properties"]["upplosning"] for it in near), default=None)
    earliest: dict[tuple[float, ...], tuple[tuple[str, int], dict[str, Any]]] = {}
    for it in near:
        p = it["properties"]
        box = tuple(p["proj:bbox"])
        season = (p["datetime"][5:10], -int(p["datetime"][:4]))  # month-day, then the newest
        if p["upplosning"] == coarsest and (box not in earliest or season < earliest[box][0]):
            earliest[box] = season, it
    return sorted((it for _, it in earliest.values()), key=lambda it: it["id"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true", help="list the downloads, then stop")
    extent = parser.add_mutually_exclusive_group(required=True)
    extent.add_argument("--bbox", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    extent.add_argument("--area", type=Path, help="GeoJSON polygon in WGS84: its tiles")
    args = parser.parse_args(argv)
    try:
        if args.area:
            tiles = polygon_tiles(json.loads(args.area.read_text(encoding="utf-8")))
        else:
            tiles = nominal_tiles(check_extent(tuple(args.bbox)))
    except ValueError as e:
        parser.error(str(e))
    x0, y0, x1, y1 = zip(*tiles, strict=True)
    area = (min(x0), min(y0), max(x1), max(y1))
    auth = fetch_area.auth_header()
    since = datetime.date.today().year - YEARS
    print(f"{len(tiles)} tiles; searching {STAC} for 4-band orthophotos since {since} ...")
    cql = f"spektraltyp='rgbi' AND upplosning<=0.5 AND flygar>={since}"
    chosen = choose(fetch_area.search(auth, area, stac=STAC, cql=cql), tiles)
    todo = []
    for it in chosen:
        asset = it["assets"]["data"]
        dest = PHOTOS / Path(urllib.parse.urlsplit(asset["href"]).path).name
        print(f"   {dest.name}  {it['collection']}  {int(asset.get('file:size', 0)) / 1e6:.0f} MB")
        if not dest.exists():
            todo.append((asset, dest))
    total = sum(int(a.get("file:size", 0)) for a, _ in todo)
    print(f"{len(chosen)} squares, {len(chosen) - len(todo)} already in {PHOTOS}; ", end="")
    print(f"to download: {len(todo)} files, {total / 1e9:.2f} GB")
    if args.plan:
        return
    PHOTOS.mkdir(parents=True, exist_ok=True)
    for n, (asset, dest) in enumerate(todo, 1):
        print(f"\n{n}/{len(todo)} {dest.name}")
        try:
            fetch_area.download(asset["href"], dest, asset, auth)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(
                    f"Lantmäteriet refused the file ({e.code}): the Geotorget account needs"
                    f" Ortofoto Nedladdning ordered. Check {fetch_area.CREDENTIALS} too."
                )
            raise
    print(f"\nDone. The photos are in {PHOTOS}; run scripts/measure_photos.py --rerun.")


if __name__ == "__main__":
    main()
