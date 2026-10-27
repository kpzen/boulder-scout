"""Write the exported boulders for the map, bundled per tile, in EPSG:4326.

Usage: uv run python -m scoring.export_geojson [--candidates data/units]
       [--photo-measures data/photo_measures.parquet] [--photos data/photos]
       [--patches data/patches] [--out data/public_export] [--min-score 2.0]
       [--min-height 3.0] [--min-area 8.0] [--no-photos] [--table scoring/score_table.json]
--candidates is the folder of unit tables or one table. The photo stage's measures
(scripts/measure_photos.py) are joined by candidate_id; a candidate without a row, or a
run without the file, has NaN photo terms, which add 0.
Each tile with an exported boulder gets two files in {out}/tiles/: {tile}.boulders.bin, its
exported boulders' point patches ({patches}/{id}.bin) one after another and then the
head, a GeoJSON FeatureCollection of the boulders; and {tile}.photos.bin, their colour crops
({photos}/{id}_rgb.png) as WebP quality 80 one after another. A feature's `patch`
and `photo` are its [offset, length] in those files; one without a patch or crop has none.
{out}/tiles.json lists every tile with its head's [offset, length]. --no-photos writes no photo
files, and the map shows the Min karta link only.
Candidates scoring below --min-score are left out; a review export at a lower cutoff
(--min-score 0 --min-height 0 --min-area 0) lets review mode label candidates below them: export
again with the defaults before building the site. So are candidates the unit stage masked and
those a quick filter dropped (`dropped`): only kept candidates are exported. --min-height and
--min-area also leave out those lower (`max_h`, m) or smaller (`area_m2`, m²) and those whose
value is missing; 0 turns a cutoff off, missing values included.
--table scores with a table fitted by scoring/fit_score.py instead of the committed one.
"""

import argparse
import io
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image
from pyproj import Transformer

from scoring.heuristic_scorer import LLR_TABLE, score

# The only place coordinates leave SWEREF 99 TM.
TO_WGS84 = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)

# Export cutoff: below it fell 39 of the 41 held-out vegetation labels and none of the 17
# boulders; the table's out-of-fold boulders all scored above 2.4.
MIN_SCORE = 2.0

# Export cutoffs on height and area, the author's choice for the climbable boulders.
MIN_HEIGHT = 3.0
MIN_AREA = 8.0

# Columns the map filters on, besides score; area_m2 also sizes the marker.
FILTER_COLUMNS = ["max_h", "area_m2"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, default=Path("data/units"))
    parser.add_argument("--photo-measures", type=Path)
    parser.add_argument("--photos", type=Path)
    parser.add_argument("--patches", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/public_export"))
    parser.add_argument("--min-score", type=float, default=MIN_SCORE)
    parser.add_argument("--min-height", type=float, default=MIN_HEIGHT, help="max_h, m; 0: off")
    parser.add_argument("--min-area", type=float, default=MIN_AREA, help="area_m2, m²; 0: off")
    parser.add_argument("--no-photos", action="store_true")
    parser.add_argument("--table", type=Path)
    args = parser.parse_args(argv)

    photo_path = args.photo_measures or args.candidates.with_name("photo_measures.parquet")
    photo = (
        {r.pop("candidate_id"): r for r in pq.read_table(photo_path).to_pylist()}
        if photo_path.exists()
        else {}
    )
    candidates = [
        r | photo.get(r["candidate_id"], {}) for r in pq.read_table(args.candidates).to_pylist()
    ]
    table = json.loads(args.table.read_text(encoding="utf-8")) if args.table else LLR_TABLE
    scored = [r | {"score": score(r, table)} for r in candidates]
    rows = [
        r
        for r in scored
        if r["score"] >= args.min_score
        and not r.get("masked", False)
        and not r.get("dropped")
        and (args.min_height <= 0 or r["max_h"] >= args.min_height)  # NaN fails
        and (args.min_area <= 0 or r["area_m2"] >= args.min_area)
    ]
    crops = args.photos or args.candidates.with_name("photos")
    patches = args.patches or args.candidates.with_name("patches")
    by_tile: dict[str, list[dict[str, Any]]] = {}
    lon, lat = TO_WGS84.transform([r["x"] for r in rows], [r["y"] for r in rows])
    for r, lo, la in zip(rows, lon, lat, strict=True):
        by_tile.setdefault(r["tile_id"], []).append(r | {"lon": lo, "lat": la})
    folder = args.out / "tiles"
    folder.mkdir(parents=True, exist_ok=True)
    for old in [*folder.glob("*.boulders.bin"), *folder.glob("*.photos.bin")]:
        old.unlink()  # nothing of an earlier export stays
    index = []
    for tile, tile_rows in sorted(by_tile.items()):
        blob, photos, features = bytearray(), bytearray(), []
        for r in tile_rows:
            props = {
                "candidate_id": r["candidate_id"],
                "score": r["score"],
                # Missing evidence (NaN) is null: JSON has no NaN.
                **{c: None if math.isnan(r[c]) else r[c] for c in FILTER_COLUMNS},
            }
            patch = patches / f"{r['candidate_id']}.bin"
            if patch.exists():
                props["patch"] = [len(blob), patch.stat().st_size]
                blob += patch.read_bytes()
            crop = crops / f"{r['candidate_id']}_rgb.png"
            if not args.no_photos and crop.exists():
                # About 13% of the lossless PNG's size, no loss visible at the viewer's size.
                webp = io.BytesIO()
                Image.open(crop).save(webp, "WEBP", quality=80, method=6)
                props["photo"] = [len(photos), webp.tell()]
                photos += webp.getvalue()
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                    "properties": props,
                }
            )
        fc = {"type": "FeatureCollection", "features": features}
        head = json.dumps(fc, allow_nan=False).encode()
        (folder / f"{tile}.boulders.bin").write_bytes(blob + head)
        if not args.no_photos:
            (folder / f"{tile}.photos.bin").write_bytes(photos)
        index.append({"tile": tile, "head": [len(blob), len(head)]})
    (args.out / "tiles.json").write_text(json.dumps(index))
    cuts = "".join(
        f", {name} >= {v}"
        for name, v in [("max_h", args.min_height), ("area_m2", args.min_area)]
        if v > 0
    )
    print(
        f"{len(rows)} of {len(scored)} candidates (score >= {args.min_score}{cuts}, not masked or"
        f" dropped) in {len(index)} tiles -> {folder}"
    )


if __name__ == "__main__":
    main()
