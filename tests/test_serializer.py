from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pipeline.ingest.laz_reader import POINT_DTYPE, read_tile
from pipeline.run_unit import main
from pipeline.serializer import MIN_RADIUS, PATCH_DTYPE, patch
from tests.real_fixtures import BOULDERS, FIXTURES
from tests.synthetic import make_synthetic_tile


def test_block_round_trip_within_1_cm(tmp_path: Path) -> None:
    (block, *_) = make_synthetic_tile(tmp_path / "t.laz")
    points = read_tile(tmp_path / "t.laz")
    disc = points[np.hypot(points["x"] - block.x, points["y"] - block.y) < MIN_RADIUS]
    origin = np.array([block.x, block.y, np.median(disc["z"])])
    # No canopy cut: the block is 10 m clear of trees, so every point in R comes back.
    rock = disc["classification"] == 1
    data = patch(disc, np.zeros(len(disc)), rock, origin, max_h=block.height)

    assert len(data) == 8 * len(disc)
    decoded = np.frombuffer(data, dtype=PATCH_DTYPE)
    for name, coord, o in zip(("dx", "dy", "dz"), ("x", "y", "z"), origin, strict=True):
        assert np.abs(decoded[name] / 100.0 + o - disc[coord]).max() <= 0.01
    assert (decoded["point_type"] == np.where(rock, 1, 0)).all()  # no vegetation here
    print(f"block patch: {len(disc)} points, {len(data)} bytes")


def test_canopy_and_point_types() -> None:
    # Ground; rock hit through canopy (last of 2 returns); a P_obj point the ground filter
    # labelled class 2; single-return foliage outside P_obj; canopy above H95 + 1 m.
    points = np.zeros(5, dtype=POINT_DTYPE)
    points["classification"] = [2, 1, 2, 1, 1]
    points["num_returns"] = [1, 2, 1, 1, 2]
    points["intensity"] = [100, 6400, 30000, 0, 0]
    z_norm = np.array([0.0, 3.0, 2.5, 2.0, 4.2])
    obj = np.array([False, True, True, False, False])
    data = patch(points, z_norm, obj, np.zeros(3), max_h=3.1)
    decoded = np.frombuffer(data, dtype=PATCH_DTYPE)

    assert decoded["point_type"].tolist() == [0, 1, 1, 2]
    assert decoded["intensity"].tolist() == [1, 100, 255, 0]


def test_run_unit_writes_one_patch_per_candidate(tmp_path: Path) -> None:
    make_synthetic_tile(tmp_path / "t.laz")
    main(
        [
            str(tmp_path / "t.laz"),
            "--out",
            str(tmp_path / "c.parquet"),
            "--patches",
            str(tmp_path / "p"),
        ]
    )

    ids = pq.read_table(tmp_path / "c.parquet").column("candidate_id").to_pylist()
    sizes = sorted((tmp_path / "p" / f"{i}.bin").stat().st_size for i in ids)
    print(f"synthetic patch sizes: {sizes}")
    assert len(sizes) == len(ids) == len(list((tmp_path / "p").iterdir()))
    assert not any(set(i) & set('<>:"/\\|?*') for i in ids)  # valid Windows file names
    assert all(3000 <= s <= 5000 for s in sizes)  # R = 8 m at 2 pulses/m²: ~400 points


def test_real_rock_patch_sizes(tmp_path: Path) -> None:
    # On the 20 author-confirmed rocks (b09 left out).
    sizes = []
    for b in BOULDERS:
        if b["id"].startswith("b09"):
            continue
        out, patches = tmp_path / f"{b['id']}.parquet", tmp_path / b["id"]
        main([str(FIXTURES / b["crops"][0]["file"]), "--out", str(out), "--patches", str(patches)])
        rows = pq.read_table(out).to_pylist()
        for e, n in [(b["e"], b["n"])] + [(r["e"], r["n"]) for r in b.get("nearby_rocks", [])]:
            row = min(rows, key=lambda r: np.hypot(r["x"] - e, r["y"] - n))
            assert np.hypot(row["x"] - e, row["y"] - n) < 3.0
            sizes.append((patches / f"{row['candidate_id']}.bin").stat().st_size)
    print(f"real rock patches: median {np.median(sizes):.0f} B, max {max(sizes)} B")
    assert len(sizes) == 20
    assert np.median(sizes) <= 6000 and max(sizes) <= 20000
