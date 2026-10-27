import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from pipeline.run_unit import main as run_unit
from scoring.export_geojson import FILTER_COLUMNS, MIN_SCORE, main
from scoring.heuristic_scorer import LLR_TABLE, score
from scripts.measure_photos import png
from tests.synthetic import make_synthetic_tile

# No height or area cutoff: these tests are about the rest of the export.
OFF = ["--min-height=0", "--min-area=0"]


def write_candidates(path: Path, xy: list[tuple[float, float]]) -> None:
    pq.write_table(
        pa.table(
            {
                "candidate_id": [f"t_{x:.1f}_{y:.1f}" for x, y in xy],
                "tile_id": ["t"] * len(xy),
                "x": [x for x, _ in xy],
                "y": [y for _, y in xy],
                "area_m2": [10.0] * len(xy),
                "max_h": [2.5] * len(xy),
                **{f: [float("nan")] * len(xy) for f in [*LLR_TABLE, *FILTER_COLUMNS]},
            }
        ),
        path,
    )


def exported(out: Path) -> list[dict[str, Any]]:
    """Every tile's features, from each head that tiles.json points at."""
    features = []
    for t in json.loads((out / "tiles.json").read_text()):
        offset, length = t["head"]
        head = (out / "tiles" / f"{t['tile']}.boulders.bin").read_bytes()[offset : offset + length]
        features += json.loads(head)["features"]
    return features


def test_known_points_convert_to_lon_lat(tmp_path: Path) -> None:
    cands, out = tmp_path / "c.parquet", tmp_path / "export"
    # SWEREF 99 TM: central meridian 15° E, false easting 500 000 m, northing 0 on the equator.
    # Reference boulder (b01): lon/lat to 5 decimals.
    write_candidates(cands, [(500000.0, 0.0), (497122.0, 6925108.0)])
    main(["--candidates", str(cands), "--out", str(out), "--min-score=-inf", *OFF])

    origin, boulder = (f["geometry"]["coordinates"] for f in exported(out))
    assert abs(origin[0] - 15.0) < 1e-6 and abs(origin[1]) < 1e-6
    assert abs(boulder[0] - 14.94422) < 1e-5 and abs(boulder[1] - 62.45714) < 1e-5


def test_nan_features_export_as_null(tmp_path: Path) -> None:
    cands, out = tmp_path / "c.parquet", tmp_path / "export"
    write_candidates(cands, [(500000.0, 0.0)])  # every LLR feature is NaN
    main(["--candidates", str(cands), "--out", str(out), "--min-score=-inf", *OFF])

    props = exported(out)[0]["properties"]
    assert all(props[c] is None for c in FILTER_COLUMNS)


def test_masked_candidates_are_left_out(tmp_path: Path) -> None:
    cands, out = tmp_path / "c.parquet", tmp_path / "export"
    write_candidates(cands, [(500000.0, 0.0), (500100.0, 0.0)])
    table = pq.read_table(cands)
    pq.write_table(table.append_column("masked", pa.array([True, False])), cands)
    main(["--candidates", str(cands), "--out", str(out), "--min-score=-inf", *OFF])

    ids = [f["properties"]["candidate_id"] for f in exported(out)]
    assert ids == ["t_500100.0_0.0"]


def test_only_exported_patches_are_copied(tmp_path: Path) -> None:
    # A candidate left out of the export keeps its patch in the unit stage's folder only;
    # the exported one's is in its tile's boulders file, before the head.
    cands, out, unit_patches = tmp_path / "c.parquet", tmp_path / "export", tmp_path / "up"
    write_candidates(cands, [(500000.0, 0.0), (500100.0, 0.0)])
    unit_patches.mkdir()
    for x in (500000.0, 500100.0):
        (unit_patches / f"t_{x:.1f}_0.0.bin").write_bytes(str(x).encode())
    table = pq.read_table(cands)
    pq.write_table(table.append_column("masked", pa.array([False, True])), cands)
    main(["--candidates", str(cands), "--patches", str(unit_patches), "--out", str(out),
          "--min-score=-inf", *OFF])  # fmt: skip

    ids = [f["properties"]["candidate_id"] for f in exported(out)]
    assert ids == ["t_500000.0_0.0"]
    head = json.loads((out / "tiles.json").read_text())[0]["head"]
    assert (out / "tiles" / "t.boulders.bin").read_bytes()[: head[0]] == b"500000.0"


def test_dropped_candidates_are_left_out(tmp_path: Path) -> None:
    # A quick filter's drop has no patch and never reaches the map, even in review mode.
    cands, out = tmp_path / "c.parquet", tmp_path / "export"
    write_candidates(cands, [(500000.0, 0.0), (500100.0, 0.0)])
    table = pq.read_table(cands)
    pq.write_table(table.append_column("dropped", pa.array(["halo", ""])), cands)
    main(["--candidates", str(cands), "--out", str(out), "--min-score=-inf", *OFF])

    ids = [f["properties"]["candidate_id"] for f in exported(out)]
    assert ids == ["t_500100.0_0.0"]


def test_candidates_below_the_cutoff_are_left_out(tmp_path: Path) -> None:
    cands, out = tmp_path / "c.parquet", tmp_path / "export"
    write_candidates(cands, [(500000.0, 0.0)])  # all NaN: score exactly 0
    for min_score, kept in [(0.0, 1), (0.01, 0), (MIN_SCORE, 0)]:
        main(["--candidates", str(cands), "--out", str(out), "--min-score", str(min_score), *OFF])
        assert len(exported(out)) == kept


def test_height_and_area_cutoffs_by_default_leave_out_lower_smaller_and_missing(
    tmp_path: Path,
) -> None:
    cands, out = tmp_path / "c.parquet", tmp_path / "export"
    xy = [(500000.0, 0.0), (500100.0, 0.0), (500200.0, 0.0), (500300.0, 0.0)]
    write_candidates(cands, xy)
    t = pq.read_table(cands)
    t = t.set_column(t.schema.get_field_index("max_h"), "max_h", [[3.5, 2.9, float("nan"), 3.0]])
    t = t.set_column(t.schema.get_field_index("area_m2"), "area_m2", [[20.0, 30.0, 30.0, 8.0]])
    pq.write_table(t, cands)
    for cut, kept in [
        ([], [0, 3]),  # the defaults, 3 m and 8 m²
        (OFF, [0, 1, 2, 3]),  # off, missing values too
        (["--min-height", "0", "--min-area", "10"], [0, 1, 2]),
        (["--min-area", "10"], [0]),
    ]:
        main(["--candidates", str(cands), "--out", str(out), "--min-score=-inf", *cut])
        ids = sorted(f["properties"]["candidate_id"] for f in exported(out))
        assert ids == sorted(f"t_{xy[i][0]:.1f}_0.0" for i in kept)


def test_synthetic_export_is_feature_collection(tmp_path: Path) -> None:
    tile, cands, out = tmp_path / "synth.laz", tmp_path / "c.parquet", tmp_path / "export"
    make_synthetic_tile(tile)
    unit_patches = tmp_path / "unit_patches"
    run_unit([str(tile), "--out", str(cands), "--patches", str(unit_patches)])
    (out / "tiles").mkdir(parents=True)
    (out / "tiles" / "old.boulders.bin").write_bytes(b"")  # an earlier export's tile does not stay
    main(["--candidates", str(cands), "--patches", str(unit_patches), "--out", str(out),
          *OFF])  # fmt: skip

    (t,) = json.loads((out / "tiles.json").read_text())
    data = (out / "tiles" / f"{t['tile']}.boulders.bin").read_bytes()
    assert sorted(p.name for p in (out / "tiles").iterdir()) == [
        f"{t['tile']}.boulders.bin", f"{t['tile']}.photos.bin"
    ]  # fmt: skip
    offset, length = t["head"]
    assert offset + length == len(data)
    fc = json.loads(data[offset:])
    rows = pq.read_table(cands).to_pylist()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == len(rows) == 2
    for f, r in zip(fc["features"], rows, strict=True):
        assert f["type"] == "Feature" and f["geometry"]["type"] == "Point"
        lon, lat = f["geometry"]["coordinates"]
        assert -180 <= lon <= 180 and -90 <= lat <= 90
        patch = f["properties"].pop("patch")
        assert f["properties"] == {
            "candidate_id": r["candidate_id"],
            "score": score(r),
            **{c: r[c] for c in FILTER_COLUMNS},
        }
        # The patch's bytes are the unit stage's, at its [offset, length].
        unit_patch = (unit_patches / f"{r['candidate_id']}.bin").read_bytes()
        assert data[patch[0] : patch[0] + patch[1]] == unit_patch
    # The size is P_obj's footprint (the extractor's): the 2.5 m block (radius 4 m) reads 22.7 m².
    block = min(fc["features"], key=lambda f: f["properties"]["max_h"])
    assert 15.0 < block["properties"]["area_m2"] < 25.0


def test_photo_measures_join_by_id(tmp_path: Path) -> None:
    # A candidate with a photo row gets its photo terms; one without scores as before.
    cands, out = tmp_path / "c.parquet", tmp_path / "export"
    write_candidates(cands, [(500000.0, 0.0), (500010.0, 0.0)])
    pq.write_table(
        pa.table({"candidate_id": ["t_500000.0_0.0"], "ndvi_p25": [0.6], "right_angles": [0.3]}),
        tmp_path / "photo_measures.parquet",
    )
    main(["--candidates", str(cands), "--out", str(out), "--min-score=-inf", *OFF])
    scores = {f["properties"]["candidate_id"]: f["properties"]["score"]
              for f in exported(out)}  # fmt: skip
    assert scores["t_500010.0_0.0"] == 0.0
    green_roof = LLR_TABLE["ndvi_p25"][-1][1] + LLR_TABLE["right_angles"][-1][1]
    assert scores["t_500000.0_0.0"] == green_roof


def test_colour_crops_are_bundled_in_the_tiles_photos_file(tmp_path: Path) -> None:
    # The exported candidate with a colour crop gets it as a WebP in its tile's
    # photos file and its [offset, length] as `photo`; one without a crop gets neither.
    # --no-photos writes no photos file and no `photo`.
    cands, out = tmp_path / "c.parquet", tmp_path / "export"
    write_candidates(cands, [(500000.0, 0.0), (500010.0, 0.0)])
    (tmp_path / "photos").mkdir()
    rock = np.full((160, 160, 3), (120, 125, 118), np.uint8)
    (tmp_path / "photos" / "t_500000.0_0.0_rgb.png").write_bytes(png(rock))
    (tmp_path / "photos" / "t_500000.0_0.0_ir.png").write_bytes(b"ir")
    main(["--candidates", str(cands), "--out", str(out), "--min-score=-inf", *OFF])
    props = {f["properties"]["candidate_id"]: f["properties"] for f in exported(out)}
    assert "photo" not in props["t_500010.0_0.0"]
    offset, length = props["t_500000.0_0.0"]["photo"]
    photos = (out / "tiles" / "t.photos.bin").read_bytes()
    assert offset + length == len(photos)
    with Image.open(io.BytesIO(photos[offset:])) as crop:
        assert crop.format == "WEBP"
        # Lossy, and WebP keeps colour as YUV, so the same picture within a few levels.
        assert np.abs(np.asarray(crop.convert("RGB"), int) - rock).max() <= 8

    main(["--candidates", str(cands), "--out", str(out), "--min-score=-inf", "--no-photos", *OFF])
    assert not any("photo" in f["properties"] for f in exported(out))
    assert [p.name for p in (out / "tiles").iterdir()] == ["t.boulders.bin"]


def test_table_option_scores_with_a_fitted_table(tmp_path: Path) -> None:
    # --table replaces the committed score table with one scoring/fit_score.py wrote.
    cands, out, table = tmp_path / "c.parquet", tmp_path / "export", tmp_path / "t.json"
    write_candidates(cands, [(500000.0, 0.0)])
    pq.write_table(
        pq.read_table(cands).set_column(
            pq.read_table(cands).schema.get_field_index("halo"), "halo", pa.array([0.5])
        ),
        cands,
    )
    table.write_text(json.dumps({"halo": [[0.0, 3.0], [1.0, 1.0]]}))
    main(["--candidates", str(cands), "--out", str(out), "--min-score=-inf", "--table", str(table),
          *OFF])  # fmt: skip
    assert exported(out)[0]["properties"]["score"] == 2.0
