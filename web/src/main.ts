import {
  type ExpressionSpecification,
  type FilterSpecification,
  LngLatBounds,
  Map,
  NavigationControl,
  ScaleControl,
  addProtocol,
  setWorkerUrl,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { type Range, readRange } from "./bundle";
import { photoButton } from "./photos";
import { startReview } from "./review";
import { openPatch } from "./view3d";

// maplibre finds its worker relative to its own module, which Vite's bundling moves; point at it.
setWorkerUrl(workerUrl);

// Each slider sets a minimum on one property; candidates with missing evidence (null) always
// pass. Height, size and score only: the shape features are in the score.
const SLIDERS = ["max_h", "area_m2", "score"] as const;
type Filter = (typeof SLIDERS)[number];

// The parts of the tiles' heads the map reads (written by scoring/export_geojson.py), all
// tiles in one collection; `tile` is added here, the tile the boulder was read from.
export type Boulders = {
  type: "FeatureCollection";
  features: {
    type: "Feature";
    geometry: { type: "Point"; coordinates: [number, number] };
    properties: { candidate_id: string; tile: string; patch?: Range; photo?: Range } & Record<Filter, number | null>;
  }[];
};

// Esri World Imagery is the default background, with the ArcGIS key from web/.env.local
// (VITE_ARCGIS_KEY); OSM is the alternative, and without a key the only basemap.
// Esri has z19 in some places only; elsewhere its z19 tile is a grey "Map data not yet available"
// square. Its tilemap says which z19 tiles exist (1 or 0 per tile, asked per 32 x 32 block); a
// missing one fails to load, and MapLibre then enlarges the z18 tile above it.
const KEY = import.meta.env.VITE_ARCGIS_KEY;
const ESRI = "https://ibasemaps-api.arcgis.com/arcgis/rest/services/World_Imagery/MapServer";
const z19Blocks: Record<string, Promise<number[]>> = {};
addProtocol("esri", async ({ url }, abort) => {
  const [z, y, x] = url.slice("esri://".length).split("/").map(Number);
  if (z === 19) {
    const block = `${y - (y % 32)}/${x - (x % 32)}`;
    z19Blocks[block] ??= fetch(`${ESRI}/tilemap/19/${block}/32/32?token=${KEY}`)
      .then((r) => r.json())
      .then((t: { data: number[] }) => t.data)
      .catch(() => []);
    if ((await z19Blocks[block])[(y % 32) * 32 + (x % 32)] !== 1) throw new Error(`no Esri z19 tile at ${y}/${x}`);
  }
  const response = await fetch(`${ESRI}/tile/${z}/${y}/${x}?token=${KEY}`, { signal: abort.signal });
  if (!response.ok) throw new Error(`Esri tile ${z}/${y}/${x}: HTTP ${response.status}`);
  return { data: await response.arrayBuffer() };
});
const BASEMAPS = {
  ...(KEY ? {
    imagery: {
      maxzoom: 19,
      tiles: ["esri://{z}/{y}/{x}"],
      attribution:
        "Powered by Esri · Source: Esri, Vantor, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community",
    },
  } : {}),
  osm: {
    maxzoom: 19,
    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
};
type Basemap = keyof typeof BASEMAPS;

// The candidate marker: a map pin (a round head with a hole, tapering to a tip) whose tip sits
// PIN_GAP px above the candidate, so it never covers the boulder. It is a signed distance field
// (as MapLibre's TinySDF: 0.75 at the edge, 1/RADIUS per pixel), so icon-color tints it per
// score and icon-halo-color outlines it.
const PIN_R = 11; // head radius, px
const PIN_L = 22; // head centre to tip, px
const PIN_GAP = 6; // tip to candidate, px
const PIN_SCALE = 2; // image pixels per screen pixel
// Pins grow with √area (fixtures: 4–258 m²): icon-size 0.75 at √area 2, 1.3 at 16.
const PIN_SIZE = [2, 0.75, 16, 1.3] as const;
// Pins from this zoom in, dots below it; the map's two candidate layers.
const PIN_ZOOM = 12;
const LAYERS = ["dots", "boulders"];
// Whether (x, y) px from the candidate (y down) is on a pin of icon-size 1; the head's hole
// counts only if withHole.
function onPin(x: number, y: number, withHole: boolean): boolean {
  const dy = y + PIN_GAP + PIN_L; // from the head's centre
  const r2 = x * x + dy * dy;
  if (r2 < 4 * 4) return withHole;
  if (r2 <= PIN_R * PIN_R) return true;
  // The sides are the tangents from the tip to the head; they touch it at (±tx, ty).
  const sin = PIN_R / PIN_L;
  const [tx, ty] = [PIN_R * Math.sqrt(1 - sin * sin), PIN_R * sin];
  return dy >= ty && y <= -PIN_GAP && Math.abs(x) <= (tx * (-PIN_GAP - y)) / (PIN_L - ty);
}
function pinImage(): { width: number; height: number; data: Uint8Array } {
  const pad = 4; // room for the halo
  const [w, h] = [2 * (pad + PIN_R), pad + PIN_R + PIN_L + PIN_GAP]; // the bottom edge is the candidate
  const [width, height] = [w * PIN_SCALE, h * PIN_SCALE];
  const n = width * height;
  const isIn = Array.from({ length: n }, (_, i) =>
    onPin(((i % width) + 0.5) / PIN_SCALE - w / 2, (Math.floor(i / width) + 0.5) / PIN_SCALE - h, false),
  );
  // Pixels next to one of the other class; the nearest such pixel gives the distance.
  const edge = [...Array(n).keys()].filter((i) => {
    const x = i % width;
    return [x > 0 ? i - 1 : -1, x < width - 1 ? i + 1 : -1, i - width, i + width].some(
      (j) => j >= 0 && j < n && isIn[j] !== isIn[i],
    );
  });
  const RADIUS = 8;
  const data = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    let d = Infinity;
    for (const j of edge) {
      if (isIn[j] === isIn[i]) continue;
      d = Math.min(d, Math.hypot((i % width) - (j % width), Math.floor(i / width) - Math.floor(j / width)));
    }
    const signed = isIn[i] ? -(d - 0.5) : d - 0.5;
    data.fill(Math.max(0, Math.min(255, Math.round(255 * (1 - (signed / RADIUS + 0.25))))), i * 4, i * 4 + 4);
  }
  return { width, height, data };
}

const map = new Map({
  container: "app",
  style: {
    version: 8,
    sources: Object.fromEntries(
      Object.entries(BASEMAPS).map(([id, b]) => [
        id,
        { type: "raster", tileSize: 256, ...b },
      ]),
    ),
    layers: Object.keys(BASEMAPS).map((id) => ({
      id,
      type: "raster",
      source: id,
      layout: { visibility: id === Object.keys(BASEMAPS)[0] ? "visible" : "none" },
    })),
  },
  center: [15.0, 62.5],
  zoom: 5,
  attributionControl: { compact: true }, // the full line covers the basemap switch
});

// Without a key OSM is the only basemap, so there is nothing to switch.
if (!KEY) document.getElementById("basemaps")!.remove();
document.getElementById("basemaps")?.addEventListener("change", (e) => {
  const input = e.target as HTMLInputElement;
  const active = input.value as Basemap;
  for (const id of Object.keys(BASEMAPS))
    map.setLayoutProperty(id, "visibility", id === active ? "visible" : "none");
});

map.on("load", async () => {
  // Every tile's head, read as text first: review mode identifies the run by their hash.
  // Paths are under the site's base: / in the dev server, the Pages sub-path in the demo.
  const base = import.meta.env.BASE_URL;
  const tiles: { tile: string; head: Range }[] = await (await fetch(`${base}tiles.json`)).json();
  const heads = await Promise.all(
    tiles.map(async ({ tile, head }) => new TextDecoder().decode(await readRange(`${base}tiles/${tile}.boulders.bin`, head))),
  );
  const text = heads.join("");
  const data: Boulders = {
    type: "FeatureCollection",
    features: heads.flatMap((h, i) =>
      (JSON.parse(h) as Boulders).features.map((f) => ({ ...f, properties: { ...f.properties, tile: tiles[i].tile } })),
    ),
  };
  map.addSource("boulders", {
    type: "geojson",
    data,
    promoteId: "candidate_id", // feature-state marks saved candidates
    attribution: 'LiDAR © <a href="https://www.lantmateriet.se">Lantmäteriet</a> (CC BY 4.0)',
  });
  map.addImage("pin", pinImage(), { sdf: true, pixelRatio: PIN_SCALE });
  // Red at the export cutoff, yellow mid-way, green from 5 (25 km run: median 5.7);
  // saved candidates are blue whatever their score.
  const colour: ExpressionSpecification = [
    "case", ["boolean", ["feature-state", "saved"], false], "#1e78ff",
    ["interpolate", ["linear"], ["get", "score"],
      2, "#d73027",
      3.5, "#fee08b",
      5, "#1a9850",
    ],
  ];
  // Zoomed out, tens of thousands of overlapping, sorted pins are slow to place and draw as one
  // blob; dots are cheap and show where the candidates are.
  map.addLayer({
    id: "dots",
    type: "circle",
    source: "boulders",
    maxzoom: PIN_ZOOM,
    layout: { "circle-sort-key": ["get", "score"] }, // likelier candidates draw on top
    paint: {
      // Small and unoutlined zoomed far out, where the outlines would hide the colours.
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 8, 2, 11, 4],
      "circle-color": colour,
      "circle-stroke-color": "#fff",
      "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 9, 0, 11, 1],
    },
  });
  map.addLayer({
    id: "boulders",
    type: "symbol",
    source: "boulders",
    minzoom: PIN_ZOOM,
    layout: {
      "icon-image": "pin",
      "icon-anchor": "bottom", // the image's bottom edge, PIN_GAP below the tip, is the candidate
      "icon-allow-overlap": true,
      "icon-ignore-placement": true,
      "symbol-sort-key": ["get", "score"], // likelier candidates draw on top
      "icon-size": ["interpolate", ["linear"], ["sqrt", ["coalesce", ["get", "area_m2"], 4]], ...PIN_SIZE],
    },
    paint: {
      "icon-color": colour,
      "icon-halo-color": "#fff",
      "icon-halo-width": 1.5,
    },
  });

  // Sliders span the data's range. Height and area start at the author's defaults (2 m, 8 m²),
  // or at the data's minimum when that is higher; score at its minimum.
  const START: Partial<Record<Filter, number>> = { max_h: 2, area_m2: 8 };
  const panel = document.getElementById("filters")!;
  const sliders = SLIDERS.map((prop) => {
    const values = data.features
      .map((f) => f.properties[prop])
      .filter((v): v is number => v !== null);
    const min = values.length ? Math.floor(Math.min(...values) * 100) / 100 : 0;
    const max = values.length ? Math.ceil(Math.max(...values) * 100) / 100 : 0;
    const row = document.createElement("label");
    row.innerHTML = `${prop} ≥ <output></output><input type="range" name="${prop}" min="${min}" max="${max}" step="0.01" value="${Math.min(Math.max(START[prop] ?? min, min), max)}">`;
    panel.append(row);
    const input = row.querySelector("input")!;
    const output = row.querySelector("output")!;
    input.addEventListener("input", () => {
      output.textContent = input.value;
      applyFilter();
    });
    output.textContent = input.value;
    return { prop, input };
  });
  // The candidates every slider passes; the map draws only these and the fit button frames them.
  // Review mode's score band, when one is chosen, takes the score slider's place.
  let shown = data.features;
  let band: [number, number] | null = null;
  // ?ids=a,b,… keeps only those candidates, for reviewing a chosen sample.
  const only = new URLSearchParams(location.search).get("ids")?.split(",");
  function applyFilter(): void {
    shown = data.features.filter((f) =>
      (!only || only.includes(f.properties.candidate_id)) &&
      sliders.every((s) => {
        const v = f.properties[s.prop];
        if (band && s.prop === "score") return v !== null && v >= band[0] && v < band[1];
        const limit = Number(s.input.value);
        return v === null || v >= limit;
      }),
    );
    // The same test as a filter on the values, so a slider move sends MapLibre a few numbers,
    // not every shown id; a null becomes ±Infinity to pass a minimum and fail a band.
    // ?ids= stays a match, a table lookup per feature.
    const filter: FilterSpecification = [
      "all",
      ...(only ? [["match", ["get", "candidate_id"], only, true, false] as ExpressionSpecification] : []),
      ...sliders.map(({ prop, input }): ExpressionSpecification =>
        band && prop === "score"
          ? ["all", [">=", ["coalesce", ["get", prop], -Infinity], band[0]], ["<", ["coalesce", ["get", prop], -Infinity], band[1]]]
          : [">=", ["coalesce", ["get", prop], Infinity], Number(input.value)],
      ),
    ];
    for (const layer of LAYERS) map.setFilter(layer, filter);
  }
  applyFilter(); // ?ids= applies from the start

  // Count the candidates drawn inside the current view, after each re-render.
  const count = document.getElementById("count")!;
  map.on("idle", () => {
    const ids = new Set(
      map.queryRenderedFeatures({ layers: LAYERS }).map((f) => f.properties.candidate_id),
    );
    count.textContent = `${ids.size} of ${data.features.length} shown`;
  });

  // Padding keeps candidates out from under the slider panel (top left, ~210 px wide).
  function fit(features: Boulders["features"], duration?: number): void {
    if (features.length === 0) return;
    const bounds = new LngLatBounds();
    for (const f of features) bounds.extend(f.geometry.coordinates);
    const padding = { top: 60, bottom: 60, left: 260, right: 80 };
    map.fitBounds(bounds, { padding, maxZoom: 17, duration });
  }
  fit(only ? shown : data.features, 0);

  // Bottom-right controls stack upwards from the attribution: scale bar, zoom + compass, fit.
  map.addControl(new ScaleControl(), "bottom-right");
  map.addControl(new NavigationControl(), "bottom-right");
  map.addControl(
    {
      onAdd: () => {
        const group = document.createElement("div");
        group.className = "maplibregl-ctrl maplibregl-ctrl-group";
        group.innerHTML = `<button type="button" id="fit" title="Fit to shown candidates" aria-label="Fit to shown candidates">⤢</button>`;
        group.querySelector("button")!.onclick = () => fit(shown);
        return group;
      },
      onRemove: () => {},
    },
    "bottom-right",
  );

  // A click opens the info panel; its buttons open the 3D panel and the aerial photos,
  // which then follow each clicked candidate until closed. Review mode (?review) adds a labelling panel under the info
  // and takes its own clicks first.
  const info = document.getElementById("info")!;
  const view3d = document.getElementById("view3d")!;
  const fmt = (v: unknown) => (typeof v === "number" ? v.toFixed(2) : "–");
  function openInfo(f: Boulders["features"][number]): void {
    const p: Record<string, unknown> = f.properties;
    const id = f.properties.candidate_id;
    const rows = ["score", "area_m2", "max_h"].map(
      (k) => `<tr data-prop="${k}"><th>${k}</th><td>${fmt(p[k])}</td></tr>`,
    );
    info.innerHTML = `<b>${id}</b><table>${rows.join("")}</table><div class="buttons"><button id="save"></button><button id="open3d">3D view</button><button id="closeinfo">Close</button></div>`;
    info.hidden = false;
    const [E, N] = id.split("_").slice(-2).map(Number); // {tile_id}_{x:.1f}_{y:.1f}, EPSG:3006
    document.getElementById("open3d")!.after(photoButton(E, N, f.properties));
    document.getElementById("closeinfo")!.onclick = () => (info.hidden = true);
    const save = document.getElementById("save")!;
    const showSaved = () => {
      save.textContent = saved[id] ? "Saved" : "Save";
      save.classList.toggle("on", id in saved);
    };
    showSaved();
    save.onclick = () => {
      if (saved[id]) delete saved[id];
      else {
        const [lon, lat] = f.geometry.coordinates;
        saved[id] = { lon, lat, max_h: f.properties.max_h, area_m2: f.properties.area_m2 };
      }
      map.setFeatureState({ source: "boulders", id }, { saved: id in saved });
      storeSaved();
      showSaved();
    };
    const open3d = () =>
      void openPatch(
        f.properties.tile,
        f.properties.patch,
        `${id} · score ${fmt(p.score)} · max_h ${typeof p.max_h === "number" ? `${fmt(p.max_h)} m` : "–"}`,
      );
    // A second click closes the 3D panel.
    document.getElementById("open3d")!.onclick = () => (view3d.hidden ? open3d() : (view3d.hidden = true));
    if (!view3d.hidden) open3d();
    review?.show(f);
  }

  // Saved candidates live in this browser's localStorage, as layer 3 has no backend
  //; each keeps its position and values, so it outlives a re-export without it.
  type Saved = Record<string, { lon: number; lat: number; max_h: number | null; area_m2: number | null }>;
  const SAVED_KEY = "boulder-scout-saved";
  let saved: Saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(SAVED_KEY) ?? "{}");
  } catch {
    // no storage (private window) or a damaged entry: start empty
  }
  for (const id of Object.keys(saved)) map.setFeatureState({ source: "boulders", id }, { saved: true });
  const exportButton = document.getElementById("export") as HTMLButtonElement;
  function storeSaved(): void {
    try {
      localStorage.setItem(SAVED_KEY, JSON.stringify(saved));
    } catch {
      // kept for this page only
    }
    const n = Object.keys(saved).length;
    exportButton.disabled = n === 0;
    exportButton.textContent = n ? `Export as KML (${n})` : "Export as KML";
  }
  storeSaved();
  // Google Maps lists cannot be written from a web page; Google My Maps imports KML, and the
  // Google Maps app shows a My Maps map under Saved → Maps.
  exportButton.onclick = () => {
    const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    const m = (v: number | null, digits: number) => (v === null ? "–" : v.toFixed(digits));
    const placemarks = Object.entries(saved).map(
      ([id, b]) => `  <Placemark>
    <name>${m(b.max_h, 1)} m high · ${m(b.area_m2, 0)} m²</name>
    <description>${esc(`Height ${m(b.max_h, 1)} m, area ${m(b.area_m2, 0)} m². Boulder Scout candidate ${id}`)}</description>
    <Point><coordinates>${b.lon.toFixed(7)},${b.lat.toFixed(7)}</coordinates></Point>
  </Placemark>`,
    );
    const kml = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <name>Boulder Scout</name>
${placemarks.join("\n")}
</Document></kml>
`;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([kml], { type: "application/vnd.google-earth.kml+xml" }));
    a.download = "boulder-scout.kml";
    a.click();
    document.getElementById("exporthint")!.textContent =
      "boulder-scout.kml downloaded: import it in Google My Maps (mymaps.google.com)";
  };
  // Dev server only: the built site has no labels to read or write.
  const review = import.meta.env.DEV && new URLSearchParams(location.search).has("review")
    ? await startReview(map, data, text, info, (b) => {
        band = b;
        applyFilter();
      })
    : null;
  // queryRenderedFeatures returns every pin whose image box holds the click, in no useful
  // order: keep those whose drawn shape holds it and open the top one (the highest score draws
  // on top, symbol-sort-key). A dot's hit is its circle already.
  map.on("click", (e) => {
    if (review?.click(e)) return;
    const dots = map.getZoom() < PIN_ZOOM;
    const ids = new Set(
      map.queryRenderedFeatures(e.point, { layers: [dots ? "dots" : "boulders"] }).map((q) => q.properties.candidate_id),
    );
    const [a, sa, b, sb] = PIN_SIZE;
    const hits = data.features.filter((f) => {
      if (!ids.has(f.properties.candidate_id)) return false;
      if (dots) return true;
      const t = Math.min(Math.max((Math.sqrt(f.properties.area_m2 ?? 4) - a) / (b - a), 0), 1);
      const s = sa + t * (sb - sa);
      const at = map.project(f.geometry.coordinates);
      return onPin((e.point.x - at.x) / s, (e.point.y - at.y) / s, true);
    });
    const f = hits.sort((p, q) => (q.properties.score ?? -Infinity) - (p.properties.score ?? -Infinity))[0];
    if (f) openInfo(f);
  });
  for (const layer of LAYERS) {
    map.on("mouseenter", layer, () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", layer, () => (map.getCanvas().style.cursor = ""));
  }
});
