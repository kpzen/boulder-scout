import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { defineConfig } from "@playwright/test";

// Review mode's labels go here in tests, not to data/labels; the dev server and the workers
// inherit it.
process.env.LABELS_FILE = resolve(tmpdir(), "boulder-scout-e2e-labels.json");
// The area page's polygon and the tiles it counts as run, likewise (e2e/area.spec.ts).
process.env.AREA_FILE = resolve(tmpdir(), "boulder-scout-e2e-area.geojson");
process.env.UNITS_DIR = resolve(tmpdir(), "boulder-scout-e2e-units");

// Cloud sessions ship a Chromium that doesn't match this Playwright's pinned revision.
const cloudChromium = "/opt/pw-browsers/chromium";

export default defineConfig({
  testDir: "e2e",
  globalSetup: "./e2e/synthetic-setup.ts",
  // The pins' symbol layer takes the map ~2 s longer to settle under software WebGL with
  // parallel workers.
  expect: { timeout: 10_000 },
  use: {
    baseURL: "http://localhost:5173",
    launchOptions: existsSync(cloudChromium) ? { executablePath: cloudChromium } : {},
  },
  webServer: {
    command: "pnpm dev --port 5173 --strictPort",
    url: "http://localhost:5173",
    // A stand-in ArcGIS key, so the imagery basemap is offered; the tests route Esri's tiles.
    env: { VITE_ARCGIS_KEY: "e2e" },
  },
});
