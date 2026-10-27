"""The area run's worker count, resume, polygon area, file release, progress and downloads while
running.

The units' work is a stand-in (run_unit.run is tested on its own); threads stand in for worker
processes so the stand-in reaches them.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyproj import Transformer

from pipeline.area import Bbox, nominal_tiles, units_on_disk
from scripts import fetch_area, run_area

GB = 1 << 30
EXTENT = (490000.0, 6917500.0, 497500.0, 6925000.0)  # 3 x 3 tiles
DTM = ["m691_49.tif", "m692_49.tif"]


def name(tile: Bbox) -> str:
    """As at the source: the stem gives the tile's south-west corner in 100 m."""
    return f"y20e018-{tile[1] / 100:.0f}_{tile[0] / 100:.0f}_25.copc.laz"


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float]]:
    """Units run in threads, write one candidate each and check their files are all there."""
    started: list[tuple[str, float]] = []

    def run(
        tile: Path,
        out: Path,
        patches: Path,
        bbox: Bbox,
        neighbours: list[tuple[Path, Bbox]],
        dtm: list[Path],
        rings: Path,
        topo: Path,
    ) -> float:
        assert tile.exists() and all(p.exists() for p, _ in neighbours) and dtm
        assert topo.name == "topografi50"
        assert all(p.exists() for p in dtm)
        started.append((tile.name, time.perf_counter()))
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"candidate_id": [tile.stem], "x": [bbox[0]]}), out)
        return 2.0

    monkeypatch.setattr(run_area, "run", run)
    monkeypatch.setattr(run_area, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(run_area, "measure_photos", lambda argv: None)
    monkeypatch.setattr(run_area, "export_geojson", lambda argv: None)
    return started


def argv(tmp_path: Path, *more: str) -> list[str]:
    return [
        "--bbox", *(str(v) for v in EXTENT), "--workers", "2",
        "--raw", str(tmp_path / "raw"), "--dtm", str(tmp_path / "dtm"),
        "--data", str(tmp_path / "data"), *more,
    ]  # fmt: skip


def on_disk(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "dtm").mkdir()
    for tile in nominal_tiles(EXTENT):
        (tmp_path / "raw" / name(tile)).write_bytes(b"")
    for d in DTM:
        (tmp_path / "dtm" / d).write_bytes(b"")


def test_workers_fit_the_free_memory_and_leave_a_core() -> None:
    assert run_area.workers(17 * GB // 2, 16) == 2  # 16 GB with the usual apps open
    assert run_area.workers(14 * GB, 16) == 3
    assert run_area.workers(64 * GB, 4) == 3
    assert run_area.workers(1 * GB, 16) == 1  # never none


def test_a_file_is_released_once_every_unit_that_reads_it_is_done() -> None:
    a, b, c = Path("a"), Path("b"), Path("c")
    box = (0.0, 0.0, 1.0, 1.0)
    plan = [(a, box, [(b, box)]), (b, box, [(a, box), (c, box)]), (c, box, [(b, box)])]
    assert run_area.released(plan, {a}) == []
    assert run_area.released(plan, {a, b}) == [a]
    assert run_area.released(plan, {a, b, c}) == [a, b, c]


def test_eta_is_this_runs_pace() -> None:
    assert run_area.eta(elapsed=300, ran=10, left=3500) == "29 h 10 min"


def test_a_resumed_run_skips_the_units_already_done(
    tmp_path: Path, stand_in: list[tuple[str, float]], capsys: pytest.CaptureFixture[str]
) -> None:
    on_disk(tmp_path)
    tiles = nominal_tiles(EXTENT)
    units = tmp_path / "data" / "units"
    units.mkdir(parents=True)
    pq.write_table(
        pa.table({"candidate_id": ["earlier"], "x": [0.0]}),
        units / f"{name(tiles[4]).split('.')[0]}.parquet",
    )
    # An earlier run's unit east of the extent stays in the export.
    east = (EXTENT[2], EXTENT[1], EXTENT[2] + 2500.0, EXTENT[1] + 2500.0)
    outside = units / f"{name(east).split('.')[0]}.parquet"
    pq.write_table(pa.table({"candidate_id": ["east"], "x": [0.0]}), outside)
    run_area.main(argv(tmp_path))
    assert sorted(n for n, _ in stand_in) == sorted(name(t) for i, t in enumerate(tiles) if i != 4)
    assert outside.exists()
    kept = pq.read_table(units)  # kept after the run, read as one table
    assert kept.num_rows == 10 and "earlier" in kept["candidate_id"].to_pylist()
    out = capsys.readouterr().out
    assert "Resuming: 1 units done" in out
    assert "[9/9]" in out and " left" in out  # progress and an ETA
    assert all((tmp_path / "raw" / name(t)).exists() for t in tiles)  # kept without --delete-laz


def test_delete_laz_removes_each_file_once_no_unit_reads_it(
    tmp_path: Path, stand_in: list[tuple[str, float]]
) -> None:
    on_disk(tmp_path)
    run_area.main(argv(tmp_path, "--delete-laz"))  # the stand-in fails if a file went early
    assert len(stand_in) == 9
    assert list((tmp_path / "raw").iterdir()) == []


def test_units_run_while_later_files_download(
    tmp_path: Path, stand_in: list[tuple[str, float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    tiles = nominal_tiles(EXTENT)
    tiles_items = [
        {
            "id": name(t),
            "properties": {"proj:bbox": list(t)},
            "assets": {"data": {"href": f"https://x/{name(t)}", "type": "laz"}},
        }
        for t in tiles
        if t != tiles[8]  # the north-east tile is sea: no file, no unit
    ]
    dtm_items = [{"assets": {"data": {"href": f"https://x/{d}"}}} for d in [*DTM, "m700_10.tif"]]
    monkeypatch.setattr(fetch_area, "auth_header", lambda: "auth")
    monkeypatch.setattr(
        fetch_area,
        "search",
        lambda auth, area, collection="": dtm_items if collection == "dtm-cog" else tiles_items,
    )
    downloaded: list[tuple[str, float]] = []

    def download(href: str, dest: Path, asset: dict[str, Any], auth: str, quiet: bool) -> None:
        assert threading.current_thread() is not threading.main_thread()
        time.sleep(0.2)
        dest.write_bytes(b"")
        downloaded.append((dest.name, time.perf_counter()))

    monkeypatch.setattr(fetch_area, "download", download)
    run_area.main(argv(tmp_path, "--fetch"))
    assert len(stand_in) == 8
    assert {n for n, _ in downloaded} == {name(t) for t in tiles[:8]} | set(DTM)  # not m700_10
    assert stand_in[0][1] < downloaded[-1][1]  # the first unit ran before the last download


def test_a_ring_is_deleted_once_every_unit_that_reads_its_file_is_done(
    tmp_path: Path, stand_in: list[tuple[str, float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    on_disk(tmp_path)
    plan = units_on_disk(tmp_path / "raw", EXTENT)
    unit = run_area.run  # type: ignore[attr-defined]
    written: set[str] = set()

    def with_rings(*args: Any) -> float:
        tile, neighbours, rings = args[0], args[4], args[6]
        done = {tmp_path / "raw" / n for n, _ in stand_in}  # one worker: in order
        present = {p.name for p in rings.glob("*.npy")}
        gone = {f"{p.name.split('.')[0]}.npy" for p in run_area.released(plan, done)}
        assert present == written - gone
        rings.mkdir(parents=True, exist_ok=True)
        for path in [tile, *(p for p, _ in neighbours)]:  # as run_unit caches them
            (rings / f"{path.name.split('.')[0]}.npy").write_bytes(b"")
            written.add(f"{path.name.split('.')[0]}.npy")
        return unit(*args)

    monkeypatch.setattr(run_area, "run", with_rings)
    run_area.main(argv(tmp_path, "--workers", "1"))
    assert len(stand_in) == 9
    assert len(written) == 9


def test_a_resume_after_delete_laz_keeps_the_units_whose_files_are_gone(
    tmp_path: Path, stand_in: list[tuple[str, float]]
) -> None:
    on_disk(tmp_path)
    tiles = nominal_tiles(EXTENT)
    units = tmp_path / "data" / "units"
    units.mkdir(parents=True)
    # A stopped --delete-laz run: the south-west unit and its neighbours are done, its file gone.
    for i in (0, 1, 3, 4):
        stem = name(tiles[i]).split(".")[0]
        pq.write_table(pa.table({"candidate_id": [stem], "x": [0.0]}), units / f"{stem}.parquet")
    (tmp_path / "raw" / name(tiles[0])).unlink()
    run_area.main(argv(tmp_path, "--delete-laz"))
    assert len(stand_in) == 5
    assert pq.read_table(units).num_rows == 9


def test_rerun_runs_the_extents_done_units_again(
    tmp_path: Path, stand_in: list[tuple[str, float]]
) -> None:
    on_disk(tmp_path)
    units = tmp_path / "data" / "units"
    units.mkdir(parents=True)
    stem = name(nominal_tiles(EXTENT)[4]).split(".")[0]
    pq.write_table(pa.table({"candidate_id": ["earlier"], "x": [0.0]}), units / f"{stem}.parquet")
    east = (EXTENT[2], EXTENT[1], EXTENT[2] + 2500.0, EXTENT[1] + 2500.0)
    outside = units / f"{name(east).split('.')[0]}.parquet"
    pq.write_table(pa.table({"candidate_id": ["east"], "x": [0.0]}), outside)
    run_area.main(argv(tmp_path, "--rerun"))
    assert len(stand_in) == 9
    ids = pq.read_table(units)["candidate_id"].to_pylist()
    assert "earlier" not in ids and "east" in ids  # the extent's tiles only


def test_area_runs_the_tiles_whose_centre_is_inside_the_polygon(
    tmp_path: Path, stand_in: list[tuple[str, float]], capsys: pytest.CaptureFixture[str]
) -> None:
    on_disk(tmp_path)
    tiles = nominal_tiles(EXTENT)
    # An L over the south-west tile, the one east of it and the one north of it, in WGS84.
    x0, y0, t = EXTENT[0], EXTENT[1], 2500.0
    corners = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 0)]
    to_wgs84 = Transformer.from_crs(3006, 4326, always_xy=True)
    ring = [list(to_wgs84.transform(x0 + i * t, y0 + j * t)) for i, j in corners]
    area = tmp_path / "area.geojson"
    area.write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}))
    run_area.main([*argv(tmp_path)[5:], "--area", str(area)])
    assert sorted(n for n, _ in stand_in) == sorted(name(tiles[i]) for i in (0, 1, 3))
    assert "3 work units; 0 tiles have no file" in capsys.readouterr().out


def test_rerun_photos_passes_on_to_the_photo_stage(
    tmp_path: Path, stand_in: list[tuple[str, float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    on_disk(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(run_area, "measure_photos", calls.append)
    run_area.main(argv(tmp_path))
    run_area.main(argv(tmp_path, "--rerun-photos"))
    assert ["--rerun" in c for c in calls] == [False, True]
