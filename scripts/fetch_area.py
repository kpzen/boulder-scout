"""Download the LAZ files of `--bbox` into data/raw/
and the ground model over it into data/dtm/.

Which files: the rule in pipeline/area.py over a fresh STAC search — the whole 2.5 km tiles
of the extent that have a file; about 110 MB a file. Files already there are
skipped, so an interrupted run resumes where it stopped. scripts/run_area.py --fetch downloads
the same files while it processes them.

Run on your own computer, from the repository:
    uv run python scripts/fetch_area.py --bbox X0 Y0 X1 Y1          download whatever is missing
    uv run python scripts/fetch_area.py --bbox X0 Y0 X1 Y1 --plan   list the downloads, then stop
The extent is EPSG:3006, whole 2.5 km tiles.
Credentials come from GEOTORGET_USERNAME and GEOTORGET_PASSWORD, or from those names in
~/.boulder-scout.env, or it asks for them. Cloud sessions cannot reach api.lantmateriet.se.
"""

import argparse
import base64
import getpass
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import truststore
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # a standalone script, but the file list is pipeline code

from pipeline.area import Bbox, check_extent, overlapping, select_work_units  # noqa: E402
from pipeline.area import files_to_fetch as plan_for  # noqa: E402

# Trust the OS certificate store (antivirus/company proxies add their own root certificate there).
truststore.inject_into_ssl()

STAC = "https://api.lantmateriet.se/stac-hojd/v1"
COLLECTION = "dsm-skoglig-copc"
RAW = REPO / "data" / "raw"
# Markhöjdmodell, a 1 m ground model GeoTIFF per 10 km square: the proposer's DTM.
DTM_COLLECTION = "dtm-cog"
DTM = REPO / "data" / "dtm"
# Credentials live outside the repository. Variables set with
# setx reach a new terminal but not an already-running one, so a file is the reliable route.
CREDENTIALS = Path.home() / ".boulder-scout.env"

TO_WGS84 = Transformer.from_crs(3006, 4326, always_xy=True)


class DropAuthOnOtherHost(urllib.request.HTTPRedirectHandler):
    # Downloads redirect to a storage host that rejects (and must not see) our password.
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            new
            and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc
        ):
            new.remove_header("Authorization")
        return new


opener = urllib.request.build_opener(DropAuthOnOtherHost)


def get(url: str, auth: str) -> Any:  # the opener's response
    req = urllib.request.Request(url, headers={"Authorization": auth, "Accept": "*/*"})
    for attempt in range(8):
        try:
            return opener.open(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == 7:
                raise
            # Rate limited: wait as long as the server asks, else back off 5, 10, 20 ... 120 s.
            retry_after = e.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else min(5 * 2**attempt, 120)
            print(f"\n   rate limited ({e.code}), waiting {wait} s ...", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def auth_header() -> str:
    """Credentials from the environment, else from CREDENTIALS, else asked for."""
    env = dict(os.environ)
    if CREDENTIALS.exists():
        for line in CREDENTIALS.read_text("utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.lstrip().startswith("#"):
                env.setdefault(key.strip(), value.strip())  # a real variable wins over the file
    user = env.get("GEOTORGET_USERNAME") or input("Geotorget username: ").strip()
    password = env.get("GEOTORGET_PASSWORD") or getpass.getpass(
        "Geotorget password (hidden while typing): "
    )
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def search(
    auth: str, area: Bbox, collection: str = COLLECTION, stac: str = STAC, cql: str = ""
) -> list[dict[str, Any]]:
    """Every item whose bbox intersects the area, de-duplicated and sorted by id; with `cql`
    (a CQL2 text filter) every matching item of every collection instead of `collection`'s."""
    lons, lats = TO_WGS84.transform(
        [area[0], area[2], area[0], area[2]], [area[1], area[1], area[3], area[3]]
    )
    box = (min(lons), min(lats), max(lons), max(lats))
    query = urllib.parse.urlencode(
        ({"filter": cql, "filter-lang": "cql2-text"} if cql else {"collections": collection})
        | {"bbox": ",".join(f"{v:.6f}" for v in box), "limit": 100}
    )
    items: dict[str, dict[str, Any]] = {}
    url = f"{stac}/search?{query}"
    while url:
        page = json.load(get(url, auth))
        for it in page["features"]:
            items[it["id"]] = it
        url = next((lk["href"] for lk in page.get("links", []) if lk.get("rel") == "next"), "")
    return [items[i] for i in sorted(items)]


def dtm_files(
    dtm_items: list[dict[str, Any]], area: Bbox, directory: Path
) -> dict[Path, dict[str, Any]]:
    """Each ground model square the units read (their points' box plus read_dtm's 1 m) in
    `directory`, and its asset; the search also returns squares only its lon/lat box reaches."""
    assets = {}
    for it in dtm_items:
        asset = it["assets"]["data"]
        assets[directory / Path(urllib.parse.urlsplit(asset["href"]).path).name] = asset
    box = (area[0] - 1, area[1] - 1, area[2] + 1, area[3] + 1)
    return {path: assets[path] for path in overlapping(sorted(assets), box)}


def download(href: str, dest: Path, asset: dict[str, Any], auth: str, quiet: bool = False) -> None:
    """Stream to `dest.part`, check what arrived, then rename: a partial file is never kept."""
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    expected = int(asset.get("file:size", 0))
    done = 0
    with get(href, auth) as r, open(part, "wb") as f:
        size = int(r.headers.get("Content-Length", 0)) or expected
        while chunk := r.read(8 << 20):
            f.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if not quiet:
                print(f"\r   {done >> 20} / {size >> 20} MB", end="", flush=True)
    if not quiet:
        print()
    if expected and done != expected:
        part.unlink()
        sys.exit(f"   {dest.name}: got {done} bytes, the catalogue says {expected}")
    # file:checksum is a multihash: "1220" prefixes a 32-byte sha2-256 digest.
    checksum = str(asset.get("file:checksum", ""))
    if checksum.startswith("1220") and digest.hexdigest() != checksum[4:]:
        part.unlink()
        sys.exit(f"   {dest.name}: checksum mismatch, the file arrived corrupt")
    part.rename(dest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true", help="list the downloads, then stop")
    parser.add_argument(
        "--bbox", type=float, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1")
    )
    args = parser.parse_args(argv)
    try:
        area = check_extent(tuple(args.bbox))
    except ValueError as e:
        parser.error(str(e))
    auth = auth_header()
    print(f"Area (EPSG:3006): E {area[0]:.0f}-{area[2]:.0f}, N {area[1]:.0f}-{area[3]:.0f}")
    print(f"Searching {COLLECTION} and {DTM_COLLECTION} ...")
    try:
        items = search(auth, area)
        dtm_items = search(auth, area, DTM_COLLECTION)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            sys.exit(f"Geotorget rejected the credentials ({e.code}). Check {CREDENTIALS}.")
        raise
    units = len(select_work_units(items, area))
    todo = plan_for(items, RAW, area)
    assets = {a["href"]: a for it in items for a in it["assets"].values()}
    sizes = {dest.name: int(assets[u.href].get("file:size", 0)) for u, dest in todo}
    print(f"{len(items)} items intersect the area; {units} work units, ", end="")
    print(f"{units - len(todo)} already in {RAW}")
    for _, dest in todo:
        print(f"   {dest.name}  {sizes[dest.name] / 1e6:.0f} MB")
    dtm_todo = []
    for dest, asset in dtm_files(dtm_items, area, DTM).items():
        if not dest.exists():
            dtm_todo.append((asset, dest))
            print(f"   {dest.name}  {int(asset.get('file:size', 0)) / 1e6:.0f} MB")
    total = sum(sizes.values()) + sum(int(a.get("file:size", 0)) for a, _ in dtm_todo)
    print(f"to download: {len(todo)} + {len(dtm_todo)} files, {total / 1e9:.2f} GB")
    if args.plan:
        return

    RAW.mkdir(parents=True, exist_ok=True)
    for n, (unit, dest) in enumerate(todo, 1):
        print(f"\n{n}/{len(todo)} {dest.name}  (tile E {unit.bbox[0]:.0f}, N {unit.bbox[1]:.0f})")
        download(unit.href, dest, assets[unit.href], auth)
    DTM.mkdir(parents=True, exist_ok=True)
    for n, (asset, dest) in enumerate(dtm_todo, 1):
        print(f"\n{n}/{len(dtm_todo)} {dest.name}")
        download(asset["href"], dest, asset, auth)
    print(f"\nDone. All {units} files are in {RAW}, the ground model in {DTM}.")


if __name__ == "__main__":
    main()
