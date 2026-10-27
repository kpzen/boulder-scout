# Boulder Scout

Finds likely climbable boulders in Swedish forest from Lantmäteriet's LiDAR, so climbers can
scout new areas from home.

**Live demo** (about 3,300 km² of forest around Överturingen):
https://kpzen.github.io/boulder-scout-demo/

Each pin is a candidate boulder, coloured by a score for how likely it is to be one. Filter by
height, size and score; click a pin for its values, a 3D view of its LiDAR points, an aerial
photo crop and a link to Min karta. **Save** keeps a boulder in your browser;
**Export as KML** downloads the saved ones for Google My Maps or a GPS app. The map is a
static site: no backend, no accounts.

You can run the same pipeline on any area in Sweden and get your own map: see below.

## What you need
- Python 3.13 with [uv](https://docs.astral.sh/uv/), and Node ≥ 22.12 with pnpm.
- About 4 GB of RAM per tile processed in parallel.
- Disk, per 2.5 km tile: about 110 MB of LiDAR, 25 MB of ground model, and 120 MB of photos
  if you use them.
- A [Geotorget](https://geotorget.lantmateriet.se) account with access to *Laserdata
  Nedladdning, skog* and *Markhöjdmodell*. Optional: *Ortofoto Nedladdning* for photos.
- Optional: an ArcGIS Location Platform API key for the aerial imagery background (else the
  map shows OpenStreetMap only).

## Setup
Install the dependencies:

```bash
uv sync
```

```bash
pnpm -C web install
```

**Geotorget credentials.** Put `GEOTORGET_USERNAME=…` and `GEOTORGET_PASSWORD=…` on two lines
in `~/.boulder-scout.env`, or set them as environment variables; otherwise the scripts ask.

**Topografi 50** (required; masks buildings, roads, water and built land). Order *Topografi 50
Nedladdning, vektor* on Geotorget and unzip its `byggnadsverk_*.zip`, `kommunikation_*.zip`
and `mark_*.zip` into `data/topografi50/` as GeoPackages.

**ArcGIS key** (optional):
1. Sign up for a free ArcGIS Location Platform account at https://location.arcgis.com.
2. In your portal (*My portal* on the dashboard), create API key credentials: *Content > My
   content > New item > Developer credentials > API key credentials*, as a *Public
   application*, with no item access and the basemaps privilege under location services. Set
   the referrers to where you will host the map and `http://localhost:5173`, and an expiry
   date (at most a year). Copy the key: it is shown only once.
3. Put it in `web/.env.local` as `VITE_ARCGIS_KEY=…`.

The key is readable by anyone in the built site, and Esri checks the referrer on only part of
the requests, so treat it as public and keep an eye on your account's usage.

## Mapping a new area
**1. Draw the area.** Start the dev server (`pnpm -C web dev`) and open
http://localhost:5173/area.html. Click the corners of an area, close it on its first corner
(or Enter), draw more if you like, and press **Save area**: this writes `data/area.geojson`.
Tiles already run show green, new ones orange with a time estimate.

**2. Fetch the photos (optional).** Without them the score leaves out the photo evidence and
the map shows no photo crops. Needs *Ortofoto Nedladdning*; about 480 MB per 5 km square
(`--plan` lists the files first):

```bash
uv run python scripts/fetch_photos.py --area data/area.geojson
```

**3. Run the pipeline.** It downloads the LiDAR and ground model as it goes (`--fetch`), finds
and scores the candidates, measures the photos if there are any, and writes the map's files
to `data/public_export/`. Expect about 35 s a tile.

```bash
uv run python scripts/run_area.py --area data/area.geojson --fetch
```

A stopped run resumes where it stopped, and later runs skip tiles already done, so you can
grow an area by drawing more and running again. `--bbox X0 Y0 X1 Y1` (EPSG:3006, whole 2.5 km
tiles) works in place of `--area`. `--delete-laz` deletes each LiDAR file once used; `--rerun`
and `--rerun-photos` redo the tiles or the photo measures after a pipeline change.

**4. Look at the map** in the dev server, http://localhost:5173.

**5. Host it (optional).** Build the site and serve `web/dist/` from any static host that
answers HTTP Range requests:

```bash
pnpm -C web build
```

## Labels and refitting (optional)
The committed score table (`scoring/score_table.json`) is fitted on the author's labels and works as it
is. To fit one to your own area:

**Label.** Open http://localhost:5173/?review and mark candidates (`boulder`, `vegetation`, …)
and boulders the pipeline missed; labels go to `data/labels/labels.json`. The map leaves out
candidates scoring below 2.0, lower than 3 m or smaller than 8 m²; to label those too, export
everything first (and export again with the defaults before hosting):

```bash
uv run python -m scoring.export_geojson --min-score 0 --min-height 0 --min-area 0
```

**Refit.** Check the fit out-of-fold, fit the table, and export with it:

```bash
uv run python -m scoring.fit_score --cv
```

```bash
uv run python -m scoring.fit_score
```

```bash
uv run python -m scoring.export_geojson --table data/score_table.json
```

## Developing
`./check` runs lint, typecheck and tests for Python and web. It is a bash script (on Windows,
run it from Git Bash), needs port 5173 free, and needs Playwright's browser once:
`pnpm -C web exec playwright install chromium`. Without any LiDAR,
`uv run python scripts/export_fixtures.py` exports the committed test crops for the dev
server.

## Licence and data
The code is MIT licensed (LICENSE). The LiDAR, ground model, Topografi 50 and aerial photos
are © Lantmäteriet (CC BY 4.0): anything shown or shared from them carries
"© Lantmäteriet".
