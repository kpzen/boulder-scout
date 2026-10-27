import { type GeoJSONSource, Map, NavigationControl, ScaleControl, setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { toSweref, toWgs84 } from "./sweref";

// The area page (dev server only): draw one or more areas on a map of Sweden, see the 2.5 km
// tiles whose centre lies inside them (pipeline/area.py's polygon_tiles, which decides the run)
// beside the tiles already run, and save them as data/area.geojson for `run_area.py --area`.
setWorkerUrl(workerUrl);

const TILE = 2500;
// One worker's pace over the polygon area: 428 tiles in 15,161 s.
const SECONDS_PER_TILE = 35;
const SNAP_PX = 10; // a click this close to the first corner closes the area
type LonLat = [number, number];

let areas: LonLat[][] = []; // closed areas, each its corners without the repeated first
let open: LonLat[] = []; // the area being drawn
const done = ((await (await fetch("/units.json")).json()) as LonLat[]).map((c) => c.join());
const ran = new Set(done);
const saved = await fetch("/area.geojson");
if (saved.ok) {
  const { geometry } = (await saved.json()).features[0];
  const polygons: LonLat[][][] = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
  areas = polygons.map((p) => p[0].slice(0, -1));
}

// South-west corners ("E,N" in EPSG:3006) of the tiles whose centre lies inside one area (even-odd).
function tilesInside(corners: LonLat[]): string[] {
  const ring = corners.map(toSweref);
  const xs = ring.map((p) => p[0]);
  const ys = ring.map((p) => p[1]);
  const [x0, y0] = [Math.floor(Math.min(...xs) / TILE) * TILE, Math.floor(Math.min(...ys) / TILE) * TILE];
  const [x1, y1] = [Math.ceil(Math.max(...xs) / TILE) * TILE, Math.ceil(Math.max(...ys) / TILE) * TILE];
  const out: string[] = [];
  for (let y = y0; y < y1; y += TILE)
    for (let x = x0; x < x1; x += TILE) {
      const [px, py] = [x + TILE / 2, y + TILE / 2];
      let hit = false;
      ring.forEach(([ax, ay], i) => {
        const [cx, cy] = ring[(i + 1) % ring.length];
        if (ay !== cy && ay > py !== cy > py && px < ax + ((py - ay) * (cx - ax)) / (cy - ay)) hit = !hit;
      });
      if (hit) out.push(`${x},${y}`);
    }
  return out;
}

const map = new Map({
  container: "app",
  style: {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        maxzoom: 19,
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      },
      tiles: { type: "geojson", data: { type: "FeatureCollection", features: [] } },
      area: { type: "geojson", data: { type: "FeatureCollection", features: [] } },
    },
    layers: [
      { id: "osm", type: "raster", source: "osm" },
      {
        id: "tiles",
        type: "fill",
        source: "tiles",
        paint: { "fill-color": ["case", ["get", "done"], "#2f9e5f", "#e8833a"], "fill-opacity": 0.4 },
      },
      { id: "tile-edges", type: "line", source: "tiles", paint: { "line-color": "#fff", "line-opacity": 0.5 } },
      { id: "outline", type: "line", source: "area", paint: { "line-color": "#c0392b", "line-width": 2 } },
      {
        id: "corners",
        type: "circle",
        source: "area",
        filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-radius": 5, "circle-color": "#c0392b", "circle-stroke-color": "#fff", "circle-stroke-width": 1.5 },
      },
    ],
  },
  bounds: [10.9, 55.3, 24.2, 69.1], // Sweden
  doubleClickZoom: false, // a quick second click is another corner
  renderWorldCopies: false,
});
map.addControl(new NavigationControl({ showCompass: false }), "top-right");
map.addControl(new ScaleControl(), "bottom-right");

const summary = document.getElementById("summary")!;
const hint = document.getElementById("hint")!;
const status = document.getElementById("status")!;
const save = document.getElementById("save") as HTMLButtonElement;

function render(): void {
  const drawn = new Set([...areas, ...(open.length >= 3 ? [open] : [])].flatMap(tilesInside));
  const fresh = [...drawn].filter((t) => !ran.has(t));
  summary.textContent =
    `${fresh.length} new tiles · about ${((fresh.length * SECONDS_PER_TILE) / 3600).toFixed(1)} h · ` +
    `${ran.size} run already`;
  hint.textContent =
    open.length === 0
      ? "Click on the map to start an area. Drag a corner to move it."
      : open.length < 3
        ? "Keep clicking to add corners."
        : "Click the first corner or press Enter to close the area.";
  save.disabled = areas.length === 0 && open.length < 3;
  const square = (t: string) => {
    const [x, y] = t.split(",").map(Number);
    const ring: LonLat[] = [[x, y], [x + TILE, y], [x + TILE, y + TILE], [x, y + TILE], [x, y]];
    return ring.map(toWgs84);
  };
  (map.getSource("tiles") as GeoJSONSource).setData({
    type: "FeatureCollection",
    features: [...done.map((t) => [t, true] as const), ...fresh.map((t) => [t, false] as const)].map(([t, d]) => ({
      type: "Feature",
      properties: { done: d },
      geometry: { type: "Polygon", coordinates: [square(t)] },
    })),
  });
  const shapes = [...areas.map((a) => [...a, a[0]]), open];
  (map.getSource("area") as GeoJSONSource).setData({
    type: "FeatureCollection",
    features: [
      ...shapes
        .filter((s) => s.length >= 2)
        .map((s) => ({ type: "Feature" as const, properties: {}, geometry: { type: "LineString" as const, coordinates: s } })),
      ...[...areas, open]
        .flat()
        .map((c) => ({ type: "Feature" as const, properties: {}, geometry: { type: "Point" as const, coordinates: c } })),
    ],
  });
}

function edited(): void {
  status.textContent = "Not saved.";
  render();
}

function close(): void {
  if (open.length < 3) return;
  areas.push(open);
  open = [];
  edited();
}

// The corner within SNAP_PX of a screen point, from the corners themselves (not what is drawn,
// which lags a just-placed corner): area index (areas.length for the open one) and position.
function cornerAt(point: { x: number; y: number }): { ai: number; ci: number } | null {
  for (const [ai, a] of [...areas, open].entries())
    for (const [ci, c] of a.entries()) {
      const p = map.project(c);
      if (Math.hypot(p.x - point.x, p.y - point.y) <= SNAP_PX) return { ai, ci };
    }
  return null;
}

// Dragging a corner moves it; the click that ends a drag places nothing.
let dragging: { ai: number; ci: number } | null = null;
let dragged = false;
map.on("mousedown", (e) => {
  dragging = cornerAt(e.point);
  dragged = false;
  if (dragging) e.preventDefault(); // no map pan
});
map.on("mousemove", (e) => {
  if (!dragging) {
    map.getCanvas().style.cursor = cornerAt(e.point) ? "move" : "";
    return;
  }
  const corners = dragging.ai === areas.length ? open : areas[dragging.ai];
  corners[dragging.ci] = [e.lngLat.lng, e.lngLat.lat];
  dragged = true;
  edited();
});
map.on("mouseup", () => (dragging = null));

map.on("load", render);
map.on("click", (e) => {
  if (dragged) {
    dragged = false;
    return;
  }
  const first = open[0] && map.project(open[0]);
  if (open.length >= 3 && first && Math.hypot(first.x - e.point.x, first.y - e.point.y) <= SNAP_PX) return close();
  // A click on a corner (one that did not move) adds nothing.
  if (cornerAt(e.point)) return;
  open.push([e.lngLat.lng, e.lngLat.lat]);
  edited();
});
addEventListener("keydown", (e) => {
  if (e.key === "Enter") close();
});
document.getElementById("undo")!.onclick = () => {
  // The last corner of the open area, or else reopen the last closed area.
  if (open.length === 0) open = areas.pop() ?? [];
  else open.pop();
  edited();
};
document.getElementById("clear")!.onclick = () => {
  areas = [];
  open = [];
  edited();
};
save.onclick = async () => {
  close(); // an open area of three corners or more is saved closed
  const area = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: {},
        geometry: { type: "MultiPolygon", coordinates: areas.map((a) => [[...a, a[0]]]) },
      },
    ],
  };
  const r = await fetch("/area.geojson", { method: "PUT", body: JSON.stringify(area) });
  status.textContent = r.ok ? "Saved to data/area.geojson." : `Save failed: HTTP ${r.status}.`;
};
status.textContent = saved.ok ? "Opened the saved area, data/area.geojson." : "";
