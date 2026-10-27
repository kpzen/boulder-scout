import math
import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.measure_photos import CUT, RES, blocks, corner, decode, main, measure, png

# Two synthetic blocks per layer side by side over E 1000-1100, N 2000-2100, split at E 1050.
X0, X1, Y0, Y1, SPLIT = 1000.0, 1100.0, 2000.0, 2100.0, 1050.0


def _area(res: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = round((Y1 - Y0) / res)
    return rng.integers(0, 256, (k, round((X1 - X0) / res), 3), dtype=np.uint8)


def _source(tmp_path: Path) -> dict[str, np.ndarray]:
    source = tmp_path / "orthophotos"
    source.mkdir()
    whole = {}
    for seed, (layer, res) in enumerate(RES.items()):
        whole[layer] = image = _area(res, seed)
        split = round((SPLIT - X0) / res)
        for a, b, part in ((X0, SPLIT, image[:, :split]), (SPLIT, X1, image[:, split:])):
            (source / f"{layer}_{a:g}_{Y0:g}_{b:g}_{Y1:g}.png").write_bytes(png(part))
    return whole


def _candidates(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "units.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _row(
    cid: str, x: float, y: float, area: float = 12.0, dropped: str = "", masked: bool = False
) -> dict[str, object]:
    return {
        "candidate_id": cid, "x": x, "y": y, "area_m2": area, "dropped": dropped, "masked": masked
    }  # fmt: skip


def _run(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    main(["--candidates", str(_candidates(tmp_path, rows)),
          "--source", str(tmp_path / "orthophotos"), "--out", str(tmp_path / "out")])  # fmt: skip


def test_crop_across_blocks_matches_source(tmp_path: Path) -> None:
    whole = _source(tmp_path)
    _run(tmp_path, [_row("t_1050.3_2050.0", 1050.3, 2050.0)])  # straddles E 1050
    for layer, res in RES.items():
        left, top = corner(1050.3, 2050.0, res)
        c, r, k = round((left - X0) / res), round((Y1 - top) / res), round(CUT / res)
        got = decode(
            (tmp_path / "out" / "photos" / f"t_1050.3_2050.0_{layer}.png").read_bytes(), ""
        )
        assert np.array_equal(got, whole[layer][r : r + k, c : c + k])
    measures = pq.read_table(tmp_path / "out" / "photo_measures.parquet").to_pylist()
    assert [m["candidate_id"] for m in measures] == ["t_1050.3_2050.0"]
    assert not math.isnan(measures[0]["ndvi_p25"]) and not math.isnan(measures[0]["right_angles"])


def test_no_object_dropped_and_uncovered(tmp_path: Path) -> None:
    _source(tmp_path)
    _run(
        tmp_path,
        [
            _row("t_1070.0_2050.0", 1070.0, 2050.0, area=math.nan),  # no P_obj: no measures
            _row("t_1050.0_2030.0", 1050.0, 2030.0, dropped="halo"),  # not measured at all
            _row("t_1030.0_2050.0", 1030.0, 2050.0, masked=True),  # nor is a masked one
            _row("t_1095.0_2050.0", 1095.0, 2050.0),  # the crop reaches past the photos
        ],
    )
    measures = pq.read_table(tmp_path / "out" / "photo_measures.parquet").to_pylist()
    assert [m["candidate_id"] for m in measures] == ["t_1070.0_2050.0"]
    assert math.isnan(measures[0]["ndvi_p25"]) and math.isnan(measures[0]["right_angles"])
    kept = sorted(p.name for p in (tmp_path / "out" / "photos").iterdir())
    assert kept == ["t_1070.0_2050.0_ir.png", "t_1070.0_2050.0_rgb.png"]


def test_no_photos_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(tmp_path, [_row("t_1050.0_2050.0", 1050.0, 2050.0)])
    assert not (tmp_path / "out").exists()
    assert "no photos" in capsys.readouterr().out


def test_measures_read_the_circle() -> None:
    # A bare disc of the candidate's size in green forest reads low ndvi_p25 only when the
    # circle is on it; a square roof in the ring reads high right_angles, rough foliage low.
    rng = np.random.default_rng(0)
    k_ir, k_rgb = round(CUT / RES["ir"]), round(CUT / RES["rgb"])
    x = y = 20.0  # the crops' centre; corner() puts their top-left at (0, 40)
    ir = np.zeros((k_ir, k_ir, 3), np.uint8)
    ir[..., 0], ir[..., 1] = 200, 40  # NIR, red: vegetation, NDVI 0.67
    yy, xx = np.mgrid[0:k_ir, 0:k_ir]
    rock = np.hypot((xx + 0.5) * RES["ir"] - x, 40 - (yy + 0.5) * RES["ir"] - y) < 2.0
    ir[rock, 0], ir[rock, 1] = 90, 90  # NDVI 0
    foliage = rng.integers(0, 256, (k_rgb, k_rgb, 3), dtype=np.uint8)
    roof = foliage.copy()
    roof[40:120, 40:120] = 220  # a 20 m square
    roof[60:100, 60:100] = 40
    on = measure(foliage, ir, x, y, x, y, math.pi * 4.0)
    off = measure(foliage, ir, x, y, x + 10.0, y, math.pi * 4.0)
    assert on["ndvi_p25"] < 0.1 < 0.6 < off["ndvi_p25"]
    assert measure(roof, ir, x, y, x, y, 80.0)["right_angles"] > 0.5
    assert on["right_angles"] < 0.2


def test_a_later_run_measures_only_new_and_rerun_units(tmp_path: Path) -> None:
    _source(tmp_path)
    units, out = tmp_path / "units", tmp_path / "out"
    units.mkdir()
    a, b = units / "a.parquet", units / "b.parquet"
    pq.write_table(pa.Table.from_pylist([_row("a_1030.0_2050.0", 1030.0, 2050.0)]), a)
    pq.write_table(pa.Table.from_pylist([_row("b_1070.0_2050.0", 1070.0, 2050.0)]), b)

    def run(*more: str) -> list[str]:
        main(["--candidates", str(units), "--source", str(tmp_path / "orthophotos"),
              "--out", str(out), *more])  # fmt: skip
        measured = out / "photo_measures.parquet"
        stamp = measured.stat().st_mtime
        os.utime(measured, (stamp + 10, stamp + 10))  # the tables are older than the measures
        return sorted(pq.read_table(measured)["candidate_id"].to_pylist())

    def crops() -> list[str]:
        return sorted(p.name.split("_")[0] for p in (out / "photos").glob("*_rgb.png"))

    assert run() == ["a_1030.0_2050.0", "b_1070.0_2050.0"]
    for p in (out / "photos").iterdir():
        p.unlink()
    assert run() == ["a_1030.0_2050.0", "b_1070.0_2050.0"] and crops() == []  # nothing redone
    # b ran again since, now with another candidate too: only b is measured again.
    pq.write_table(
        pa.Table.from_pylist([_row("b_1060.0_2050.0", 1060.0, 2050.0)]), b
    )  # b_1070's candidate is gone
    later = (out / "photo_measures.parquet").stat().st_mtime + 10
    os.utime(b, (later, later))
    assert run() == ["a_1030.0_2050.0", "b_1060.0_2050.0"] and crops() == ["b"]
    assert run("--rerun") == ["a_1030.0_2050.0", "b_1060.0_2050.0"] and crops() == ["a", "b"]


def _geotiff(
    path: Path, pixels: np.ndarray, x0: float, y1: float, res: float, compression: int = 8
) -> None:
    """A GeoTIFF laid out as Lantmäteriet's orthophotos: 4 bands of 8 bits, tiled (64 px here,
    so crops cross tiles), deflate with the horizontal predictor, pixel-is-area EPSG:3006."""
    side, (h, w, _) = 64, pixels.shape
    data, offsets, counts = bytearray(), [], []
    for r in range(0, h, side):
        for c in range(0, w, side):
            tile = np.zeros((side, side, 4), np.uint8)
            part = pixels[r : r + side, c : c + side]
            tile[: part.shape[0], : part.shape[1]] = part
            tile[:, 1:] -= tile[:, :-1].copy()  # uint8 wraps mod 256, as the predictor wants
            offsets.append(8 + len(data))
            data += zlib.compress(tile.tobytes())
            counts.append(len(data) + 8 - offsets[-1])
    keys = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3006)
    entries = sorted(
        [
            (256, 4, [w]), (257, 4, [h]), (258, 3, [8] * 4), (259, 3, [compression]),
            (262, 3, [2]), (277, 3, [4]), (284, 3, [1]), (317, 3, [2]), (322, 3, [side]),
            (323, 3, [side]), (324, 4, offsets), (325, 4, counts), (338, 3, [0]),
            (33550, 12, [res, res, 0.0]), (33922, 12, [0.0, 0.0, 0.0, x0, y1, 0.0]),
            (34735, 3, list(keys)),
        ]
    )  # fmt: skip
    ifd = 8 + len(data)
    extra = ifd + 2 + 12 * len(entries) + 4  # values over 4 bytes go after the IFD
    table, values = bytearray(struct.pack("<H", len(entries))), bytearray()
    for tag, typ, vals in entries:
        raw = struct.pack(f"<{len(vals)}{dict([(3, 'H'), (4, 'I'), (12, 'd')])[typ]}", *vals)
        if len(raw) <= 4:
            table += struct.pack("<HHI", tag, typ, len(vals)) + raw.ljust(4, b"\0")
        else:
            table += struct.pack("<HHII", tag, typ, len(vals), extra + len(values))
            values += raw
    path.write_bytes(b"II*\0" + struct.pack("<I", ifd) + data + table + b"\0" * 4 + values)


def test_cogs_are_cut_at_their_own_res_before_png_blocks(tmp_path: Path) -> None:
    # Two 4-band GeoTIFFs at 0.4 m side by side as Lantmäteriet's (red, green, blue, NIR), the
    # right one without data past E 1090; PNG blocks beside them are not read.
    _source(tmp_path)  # the crops below match the GeoTIFFs' pixels, not these blocks'
    res = 0.4
    four = np.random.default_rng(9).integers(1, 256, (250, 250, 4), dtype=np.uint8)
    four[:, 225:] = 0  # E 1090-1100: no data
    split = round((SPLIT - X0) / res)
    for x, part in ((X0, four[:, :split]), (SPLIT, four[:, split:])):
        _geotiff(tmp_path / "orthophotos" / f"o_{x:g}.tif", part, x, Y1, res)
    _run(
        tmp_path, [_row("t_1050.3_2050.0", 1050.3, 2050.0), _row("t_1075.0_2050.0", 1075.0, 2050.0)]
    )
    left, top = corner(1050.3, 2050.0, res)
    c, r, k = round((left - X0) / res), round((Y1 - top) / res), round(CUT / res)
    for layer, bands in (("rgb", [0, 1, 2]), ("ir", [3, 0, 1])):
        got = decode(
            (tmp_path / "out" / "photos" / f"t_1050.3_2050.0_{layer}.png").read_bytes(), ""
        )
        assert got.shape == (k, k, 3) and np.array_equal(got, four[r : r + k, c : c + k, bands])
    measures = pq.read_table(tmp_path / "out" / "photo_measures.parquet").to_pylist()
    assert [m["candidate_id"] for m in measures] == ["t_1050.3_2050.0"]  # t_1075 reaches no data


def test_a_cog_in_another_layout_is_refused(tmp_path: Path) -> None:
    pixels = np.ones((64, 64, 4), np.uint8)
    _geotiff(tmp_path / "lzw.tif", pixels, X0, Y1, 0.4, compression=5)
    with pytest.raises(ValueError, match="not a tiled deflated 4-band"):
        blocks(tmp_path)
