import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from pipeline.ingest.dtm_reader import read_dtm

TILE = 16


def write_geotiff(path: Path, values: np.ndarray, x0: float, y1: float, predictor: int = 3) -> None:
    """A Markhöjdmodell-like file: classic little-endian TIFF, float32, TILE x TILE tiles,
    deflate, the floating-point predictor, pixel-is-area EPSG:3006, nodata -9999."""
    h, w = values.shape
    across, down = -(-w // TILE), -(-h // TILE)
    tiles = []
    for tr in range(down):
        for tc in range(across):
            t = np.full((TILE, TILE), -9999.0, np.float32)
            part = values[tr * TILE : (tr + 1) * TILE, tc * TILE : (tc + 1) * TILE]
            t[: part.shape[0], : part.shape[1]] = part
            # Adobe TN3: big-endian byte planes per row, then byte-wise horizontal differences.
            planes = t.astype(">f4").view(np.uint8).reshape(TILE, TILE, 4)
            rows = planes.transpose(0, 2, 1).reshape(TILE, 4 * TILE)
            diff = np.diff(rows, axis=1, prepend=np.uint8(0)).astype(np.uint8)
            tiles.append(zlib.compress(diff.tobytes()))
    geo_keys = (1, 1, 0, 2, 1025, 0, 1, 1, 3072, 0, 1, 3006)
    entries: list[tuple[int, int, bytes]] = [  # (tag, type, packed values)
        (256, 3, struct.pack("<H", w)),
        (257, 3, struct.pack("<H", h)),
        (258, 3, struct.pack("<H", 32)),
        (259, 3, struct.pack("<H", 8)),
        (277, 3, struct.pack("<H", 1)),
        (317, 3, struct.pack("<H", predictor)),
        (322, 3, struct.pack("<H", TILE)),
        (323, 3, struct.pack("<H", TILE)),
        (324, 4, b""),  # offsets, filled below
        (325, 4, struct.pack(f"<{len(tiles)}I", *map(len, tiles))),
        (339, 3, struct.pack("<H", 3)),
        (33550, 12, struct.pack("<3d", 1.0, 1.0, 0.0)),
        (33922, 12, struct.pack("<6d", 0, 0, 0, x0, y1, 0)),
        (34735, 3, struct.pack(f"<{len(geo_keys)}H", *geo_keys)),
        (42113, 2, b"-9999\0"),
    ]
    # Header, then the tiles, then the IFD and its out-of-line values.
    sizes = {2: 1, 3: 2, 4: 4, 12: 8}
    offsets = np.cumsum([8] + [len(t) for t in tiles])[:-1]
    entries[8] = (324, 4, struct.pack(f"<{len(tiles)}I", *offsets))
    ifd_at = 8 + sum(map(len, tiles))
    extra_at = ifd_at + 2 + 12 * len(entries) + 4
    ifd, extra = struct.pack("<H", len(entries)), b""
    for tag, typ, v in entries:
        count = len(v) // sizes[typ]
        if len(v) <= 4:
            ifd += struct.pack("<HHI", tag, typ, count) + v.ljust(4, bytes(1))
        else:
            ifd += struct.pack("<HHII", tag, typ, count, extra_at + len(extra))
            extra += v
    head = b"II*" + bytes(1) + struct.pack("<I", ifd_at)
    path.write_bytes(head + b"".join(tiles) + ifd + struct.pack("<I", 0) + extra)


def test_reads_windows_across_tiles_and_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    west = (300 + rng.random((40, 37)) * 10).astype(np.float32)  # not whole tiles
    east = (300 + rng.random((40, 30)) * 10).astype(np.float32)
    west[5, 6] = -9999.0
    write_geotiff(tmp_path / "w.tif", west, 1000.0, 2040.0)
    write_geotiff(tmp_path / "e.tif", east, 1037.0, 2040.0)
    files = [tmp_path / "w.tif", tmp_path / "e.tif"]

    whole = read_dtm(files, (1000.0, 2000.0, 1067.0, 2040.0))
    assert (whole.x0, whole.y1, whole.cell) == (1000.0, 2040.0, 1.0)
    expected = np.hstack([west, east])
    expected[5, 6] = np.nan
    np.testing.assert_array_equal(whole.values, expected)

    # Snapped outward to whole cells; past the files' edges is unknown.
    part = read_dtm(files, (1030.4, 2010.2, 1070.0, 2025.5))
    assert (part.x0, part.y1, part.values.shape) == (1030.0, 2026.0, (16, 40))
    np.testing.assert_array_equal(part.values[:, :37], expected[14:30, 30:67])
    assert np.isnan(part.values[:, 37:]).all()


def test_rejects_other_layouts(tmp_path: Path) -> None:
    write_geotiff(tmp_path / "p2.tif", np.zeros((16, 16), np.float32), 0.0, 16.0, predictor=2)
    with pytest.raises(ValueError, match="predictor 2"):
        read_dtm([tmp_path / "p2.tif"], (0.0, 0.0, 16.0, 16.0))
    (tmp_path / "big.tif").write_bytes(b"MM\0*" + bytes(12))
    with pytest.raises(ValueError, match="little-endian"):
        read_dtm([tmp_path / "big.tif"], (0.0, 0.0, 16.0, 16.0))
