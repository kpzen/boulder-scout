import {
  createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, statSync, writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// scripts/build_demo.py points EXPORT_DIR at the demo's chosen tiles.
const EXPORT = process.env.EXPORT_DIR ?? fileURLToPath(new URL("../data/public_export", import.meta.url));
// Review mode's labels; the e2e tests point LABELS_FILE elsewhere.
const LABELS = process.env.LABELS_FILE ?? fileURLToPath(new URL("../data/labels/labels.json", import.meta.url));
// The area page's polygon for run_area.py --area, and the unit tables of the tiles already run
//; the e2e tests point both elsewhere.
const AREA = process.env.AREA_FILE ?? fileURLToPath(new URL("../data/area.geojson", import.meta.url));
const UNITS = process.env.UNITS_DIR ?? fileURLToPath(new URL("../data/units", import.meta.url));

// A FeatureCollection holding one MultiPolygon of [lon, lat] rings, as the area page writes it.
function isArea(area: { features?: { geometry?: { type?: string; coordinates?: unknown } }[] }): boolean {
  const geometry = area.features?.[0]?.geometry;
  const polygons = geometry?.coordinates;
  const isRing = (ring: unknown) =>
    Array.isArray(ring) && ring.length >= 4 &&
    ring.every((p) => Array.isArray(p) && p.length === 2 && Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90);
  return (
    area.features?.length === 1 && geometry?.type === "MultiPolygon" && Array.isArray(polygons) &&
    polygons.length >= 1 && polygons.every((p) => Array.isArray(p) && p.length === 1 && isRing(p[0]))
  );
}

// The build copies the pipeline's static export (tiles.json and tiles/) to the site root.
// The dev server serves it straight from disk instead, unscanned and unwatched. A missing file
// is a 404, not Vite's index.html fallback, so a failed load shows as one. A range request (how the
// map reads a tile's head, patches and photo crops) gets that range.
export default defineConfig(({ command }) => ({
  publicDir: command === "build" ? EXPORT : false,
  // Review mode imports the label vocabulary it shares with the fit.
  server: { fs: { allow: [".", "../scoring/label_vocabulary.json"] } },
  plugins: [
    {
      name: "serve-export",
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const path = req.url?.split("?")[0] ?? "";
          // The one write layer 3 makes, and only in the dev server: review mode
          // replaces the labels file with a JSON array on every change.
          if (path === "/labels.json" && req.method === "PUT") {
            let body = "";
            req.on("data", (chunk) => (body += chunk));
            req.on("end", () => {
              try {
                if (!Array.isArray(JSON.parse(body))) throw new Error("not an array");
              } catch {
                res.statusCode = 400;
                return res.end();
              }
              mkdirSync(dirname(LABELS), { recursive: true });
              writeFileSync(`${LABELS}.tmp`, body); // a crash mid-write leaves the old file whole
              renameSync(`${LABELS}.tmp`, LABELS);
              res.end();
            });
            return;
          }
          if (path === "/labels.json") {
            res.setHeader("Content-Type", "application/json");
            return res.end(existsSync(LABELS) ? readFileSync(LABELS) : "[]");
          }
          // The area page's second write: replaces the area file with its polygon.
          if (path === "/area.geojson" && req.method === "PUT") {
            let body = "";
            req.on("data", (chunk) => (body += chunk));
            req.on("end", () => {
              try {
                if (!isArea(JSON.parse(body))) throw new Error("not one polygon");
              } catch {
                res.statusCode = 400;
                return res.end();
              }
              mkdirSync(dirname(AREA), { recursive: true });
              writeFileSync(`${AREA}.tmp`, body);
              renameSync(`${AREA}.tmp`, AREA);
              res.end();
            });
            return;
          }
          if (path === "/area.geojson") {
            if (!existsSync(AREA)) {
              res.statusCode = 404;
              return res.end();
            }
            res.setHeader("Content-Type", "application/json");
            return res.end(readFileSync(AREA));
          }
          // The south-west corners [E, N] of the tiles with a unit table; the stem gives the corner
          // in 100 m (y20e018-69175_4900_25 is E 490000, N 6917500).
          if (path === "/units.json") {
            const corners = (existsSync(UNITS) ? readdirSync(UNITS) : []).flatMap((name) => {
              const m = /-(\d+)_(\d+)_25\.parquet$/.exec(name);
              return m ? [[Number(m[2]) * 100, Number(m[1]) * 100]] : [];
            });
            res.setHeader("Content-Type", "application/json");
            return res.end(JSON.stringify(corners));
          }
          if (path !== "/tiles.json" && !/^\/tiles\/[\w.-]+\.(boulders|photos)\.bin$/.test(path)) return next();
          const file = `${EXPORT}${path}`;
          if (!existsSync(file)) {
            res.statusCode = 404;
            return res.end();
          }
          res.setHeader("Content-Type", path.endsWith(".bin") ? "application/octet-stream" : "application/json");
          const range = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range ?? "");
          if (!range) return createReadStream(file).pipe(res);
          const size = statSync(file).size;
          const [start, end] = [Number(range[1]), Math.min(Number(range[2]), size - 1)];
          res.statusCode = 206;
          res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
          createReadStream(file, { start, end }).pipe(res);
        });
      },
    },
  ],
}));
