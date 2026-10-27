"""Run an extent through the pipeline into one map export.

Usage: uv run python scripts/run_area.py (--bbox X0 Y0 X1 Y1 | --area area.geojson) [--fetch]
       [--workers N] [--rerun] [--rerun-photos] [--delete-laz] [--raw data/raw] [--dtm data/dtm]
       [--data data]
Runs each work unit of the extent (--bbox, or with --area the tiles whose
centre lies inside a GeoJSON polygon; a tile with no file has none)
with its neighbours' 50 m edge strips, the ground model and the Topografi 50 mask
in {data}/topografi50 into one table per unit, {data}/units/{unit}.parquet,
crops and measures the photos in {data}/orthophotos when there are any
(scripts/measure_photos.py: only the candidates of units run since, unless
--rerun-photos) and writes {data}/public_export/, per tile: the exported
boulders with their point patches and their colour photo crops, from {data}/patches and
{data}/photos, plus tiles.json,
which `pnpm -C web dev` serves from data/.
Replaces any previous export there, including the fixture export.

As many units run at once as the free memory and the cores allow, unless --workers says.
With --fetch it downloads the extent's files (scripts/fetch_area.py) while it runs, starting each
unit once its file, its neighbours' and its ground model are on disk; without, it runs the files
already in {raw}. The unit tables are kept after a run: a unit whose table is in {data}/units is
not run again unless --rerun (or delete it to rerun the unit, or the folder to start over), and
the tables of tiles outside the extent are kept, so the export holds every tile run so far
(delete their tables to drop them). A file's edge ring ({data}/rings) is deleted
once every unit that reads the file is done, and with --delete-laz the file itself.
"""

import argparse
import ctypes
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from pipeline.area import (  # noqa: E402
    Bbox,
    check_extent,
    nominal_tiles,
    overlapping,
    polygon_tiles,
    select_work_units,
    units_on_disk,
    with_neighbours,
)
from pipeline.proposer.raster_relief import MIN_PULSE_DENSITY  # noqa: E402
from pipeline.run_unit import EDGE_BUFFER, run  # noqa: E402
from scoring.export_geojson import main as export_geojson  # noqa: E402
from scripts import fetch_area  # noqa: E402
from scripts.measure_photos import main as measure_photos  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PEAK = 4 << 30  # bytes a unit may hold: the densest unit peaks at 3.95 GB
# Bytes left beside the units for what the rest of the machine starts meanwhile. On 16 GB with
# 8.5 GB free, 2 units ran at 1.12x their solo time (624 s for the v1 area, 1,039 s solo) and
# 3 at 1.9x (712 s), so 2 is what fits there.
HEADROOM = 1 << 29

Plan = list[tuple[Path, Bbox, list[tuple[Path, Bbox]]]]


def timed_run(*args: Any) -> tuple[float, float]:
    """run_unit.run in a worker process: the unit's pulse density and its seconds."""
    t = time.perf_counter()
    return run(*args), time.perf_counter() - t


def free_memory() -> int:
    """Bytes of physical memory available now (on Windows including the standby cache)."""
    if sys.platform == "win32":

        class Status(ctypes.Structure):  # MEMORYSTATUSEX
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total", "avail", "pf", "pf_avail", "vm", "vm_avail", "ext")
            ]

        status = Status()
        status.length = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.avail)
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def workers(free: int, cores: int) -> int:
    """Units to run at once: each may peak at PEAK beside HEADROOM, and one core stays free."""
    return max(1, min(cores - 1, (free - HEADROOM) // PEAK))


def released(plan: Plan, done: set[Path]) -> list[Path]:
    """The files no unit still to run reads: their own unit and all its neighbours are done."""
    return [tile for tile, _, ns in plan if tile in done and all(p in done for p, _ in ns)]


def eta(elapsed: float, ran: int, left: int) -> str:
    """Time left at this run's pace so far."""
    minutes = round(elapsed / ran * left / 60)
    return f"{minutes // 60} h {minutes % 60:02d} min"


def window(bbox: Bbox) -> Bbox:
    """The ground a unit reads: its tile, its neighbours' edge strips and read_dtm's 1 m."""
    m = EDGE_BUFFER + 1
    return (bbox[0] - m, bbox[1] - m, bbox[2] + m, bbox[3] + m)


def fetch(order: list[Path], downloads: dict[Path, tuple[str, dict[str, Any]]], auth: str) -> None:
    """Download `order` one file at a time (the run's background thread)."""
    try:
        for n, dest in enumerate(order, 1):
            href, asset = downloads[dest]
            dest.parent.mkdir(parents=True, exist_ok=True)
            fetch_area.download(href, dest, asset, auth, quiet=True)
            print(f"downloaded {n}/{len(order)} {dest.name}", flush=True)
    except BaseException as e:  # sys.exit in a thread would end it without a word
        print(f"downloads stopped: {e}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    extent = parser.add_mutually_exclusive_group(required=True)
    extent.add_argument("--bbox", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    extent.add_argument("--area", type=Path, help="GeoJSON polygon in WGS84: its tiles")
    parser.add_argument("--fetch", action="store_true", help="download the files while running")
    parser.add_argument("--workers", type=int, help="units at once (default: from free memory)")
    parser.add_argument("--rerun", action="store_true", help="rerun the extent's done units")
    parser.add_argument("--rerun-photos", action="store_true", help="measure every candidate again")
    parser.add_argument("--delete-laz", action="store_true", help="delete each file once read")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--dtm", type=Path, default=ROOT / "data" / "dtm")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args(argv)
    try:
        if args.area:
            keep = polygon_tiles(json.loads(args.area.read_text(encoding="utf-8")))
            x0, y0, x1, y1 = zip(*keep, strict=True)
            area = (min(x0), min(y0), max(x1), max(y1))
        else:
            area = check_extent(tuple(args.bbox))
            keep = nominal_tiles(area)
    except ValueError as e:
        parser.error(str(e))

    downloads: dict[Path, tuple[str, dict[str, Any]]] = {}
    if args.fetch:
        auth = fetch_area.auth_header()
        items = fetch_area.search(auth, area)
        units = select_work_units(items, area)
        assets = {a["href"]: a for it in items for a in it["assets"].values()}
        downloads = {args.raw / u.name: (u.href, assets[u.href]) for u in units}
        dtm_items = fetch_area.search(auth, area, fetch_area.DTM_COLLECTION)
        dtm_assets = fetch_area.dtm_files(dtm_items, area, args.dtm)
        downloads |= {path: (a["href"], a) for path, a in dtm_assets.items()}
        plan = with_neighbours([(args.raw / u.name, u.bbox) for u in units])
        dtm = sorted(dtm_assets)
    else:
        plan = units_on_disk(args.raw, area)
        dtm = sorted(args.dtm.glob("*.tif"))
        if not dtm:
            sys.exit(f"no ground model in {args.dtm}: run scripts/fetch_area.py")
    tiles = set(keep)  # --area's extent is the polygon's tiles, not its bounding box
    plan = [(p, t, [(q, n) for q, n in ns if n in tiles]) for p, t, ns in plan if t in tiles]
    where = "at the source" if args.fetch else f"in {args.raw}"
    print(f"{len(plan)} work units; {len(keep) - len(plan)} tiles have no file {where}")

    export = args.data / "public_export"
    units = args.data / "units"
    rings = args.data / "rings"  # each file's edge strips, so a file is read whole at most twice
    outs = {tile: units / f"{tile.name.split('.')[0]}.parquet" for tile, _, _ in plan}
    if args.rerun:
        for out in outs.values():
            out.unlink(missing_ok=True)
    done = {tile for tile, out in outs.items() if out.exists()}
    if done:
        print(f"Resuming: {len(done)} units done in {units} (--rerun to run them again)")
    elif not any(units.glob("*.parquet")):  # other extents' patches stay with their tables
        shutil.rmtree(args.data / "patches", ignore_errors=True)
        shutil.rmtree(rings, ignore_errors=True)
    needs = {
        tile: [tile, *(p for p, _ in ns), *overlapping(dtm, window(b))] for tile, b, ns in plan
    }
    todo = [entry for entry in plan if entry[0] not in done]

    fetcher = None
    if args.fetch:
        # Unit by unit, so the first units can run while the rest download.
        order = {p: None for tile, _, _ in todo for p in needs[tile] if not p.exists()}
        print(f"{len(order)} files to download")
        fetcher = threading.Thread(target=fetch, args=(list(order), downloads, auth), daemon=True)
        fetcher.start()

    cores = os.process_cpu_count() or 1
    cap = args.workers or workers(free_memory(), cores)
    print(f"{cap} workers ({free_memory() / 2**30:.1f} GB free, {cores} cores)", flush=True)
    start = time.perf_counter()
    density: dict[Path, float] = {}
    seconds: dict[Path, float] = {}
    gone: set[Path] = set()
    running: dict[Future[tuple[float, float]], Path] = {}
    with ProcessPoolExecutor(cap) as pool:
        while todo or running:
            ready = next((e for e in todo if all(p.exists() for p in needs[e[0]])), None)
            # Another unit only while one more peak fits beside what the machine holds now.
            if ready and len(running) < cap and (not running or free_memory() - HEADROOM >= PEAK):
                todo.remove(ready)
                tile, bbox, ns = ready
                job = (
                    tile,
                    outs[tile],
                    args.data / "patches",
                    bbox,
                    ns,
                    overlapping(dtm, window(bbox)),
                )
                topo = args.data / "topografi50"
                running[pool.submit(timed_run, *job, rings, topo)] = tile
                continue
            if not running:
                if fetcher is None or not fetcher.is_alive():
                    missing = [p.name for p in needs[todo[0][0]] if not p.exists()]
                    sys.exit(f"{len(todo)} units cannot run; {todo[0][0].name} lacks {missing}")
                time.sleep(2)  # waiting for downloads
                continue
            for future in wait(running, timeout=5, return_when=FIRST_COMPLETED)[0]:
                tile = running.pop(future)
                density[tile], seconds[tile] = future.result()
                done.add(tile)
                elapsed = time.perf_counter() - start
                left = len(todo) + len(running)
                print(
                    f"[{len(done)}/{len(plan)}] {tile.name.split('.')[0]} {seconds[tile]:.0f} s;"
                    f" {elapsed:.0f} s so far, {eta(elapsed, len(seconds), left)} left",
                    flush=True,
                )
                for path in released(plan, done):
                    if path not in gone:
                        gone.add(path)
                        (rings / f"{path.name.split('.')[0]}.npy").unlink(missing_ok=True)
                        if args.delete_laz:
                            path.unlink(missing_ok=True)

    measure_photos(
        ["--candidates", str(units),
         "--source", str(args.data / "orthophotos"), "--out", str(args.data),
         *(["--rerun"] if args.rerun_photos else [])]
    )  # fmt: skip
    export_geojson(
        ["--candidates", str(units),
         "--out", str(export)]
    )  # fmt: skip

    print(f"\n{'tile':<24}{'pulses/m²':>10}{'candidates':>12}{'seconds':>9}")
    ran = {tile.name.split(".")[0]: tile for tile in seconds}
    # Every unit in the export, also those whose file --delete-laz removed before a resume.
    rows = {out.stem: pq.read_metadata(out).num_rows for out in sorted(units.glob("*.parquet"))}
    for stem, n in rows.items():
        unit = ran.get(stem)
        pd, sec = (f"{density[unit]:.2f}", f"{seconds[unit]:.0f}") if unit else ("-", "-")
        print(f"{stem:<24}{pd:>10}{n:>12}{sec:>9}")
    low = [(tile, d) for tile, d in density.items() if d < MIN_PULSE_DENSITY]
    elapsed = time.perf_counter() - start
    print(f"\n{len(rows)} units, {sum(rows.values())} candidates, {elapsed:.0f} s")
    print(f"Below MIN_PULSE_DENSITY ({MIN_PULSE_DENSITY}/m²): {len(low)} tiles (run this time)")
    for tile, d in low:
        print(f"   {tile.name.split('.')[0]}  {d:.2f}/m²")


if __name__ == "__main__":  # required: propose's process pool re-imports this script
    main()
