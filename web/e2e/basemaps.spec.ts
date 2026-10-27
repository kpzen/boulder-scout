import { expect, type Page, test } from "@playwright/test";
import { serve } from "./export";

const geojson = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [14.944, 62.457] },
      properties: { candidate_id: "a", max_h: 2, area_m2: 10, score: 2 },
    },
  ],
};

// Each host gets a solid tile of its own colour, so the screenshot shows which source is drawn.
const IMAGERY = [20, 40, 160];
const OSM = [230, 140, 30];

async function solidPng(page: Page, [r, g, b]: number[]): Promise<Buffer> {
  const url = await page.evaluate(
    (fill) => {
      const c = Object.assign(document.createElement("canvas"), { width: 256, height: 256 });
      const ctx = c.getContext("2d")!;
      ctx.fillStyle = fill;
      ctx.fillRect(0, 0, 256, 256);
      return c.toDataURL("image/png");
    },
    `rgb(${r},${g},${b})`,
  );
  return Buffer.from(url.split(",")[1], "base64");
}

// Share of map pixels within 10 of each basemap colour. The attribution bar is left out: its
// text is subpixel-antialiased on Windows, and the colour fringes land within that tolerance.
async function shares(page: Page): Promise<{ imagery: number; osm: number }> {
  const clip = await page.evaluate(() => {
    const { top } = document.querySelector(".maplibregl-ctrl-attrib")!.getBoundingClientRect();
    return { x: 0, y: 0, width: window.innerWidth, height: Math.floor(top) };
  });
  const png = await page.screenshot({ clip });
  return page.evaluate(
    async ([src, imagery, osm]) => {
      const img = new Image();
      img.src = src as string;
      await img.decode();
      const ctx = Object.assign(document.createElement("canvas"), {
        width: img.width,
        height: img.height,
      }).getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, img.width, img.height).data;
      const near = (i: number, c: number[]) => c.every((v, k) => Math.abs(d[i + k] - v) <= 10);
      const n = { imagery: 0, osm: 0 };
      for (let i = 0; i < d.length; i += 4) {
        if (near(i, imagery as number[])) n.imagery++;
        else if (near(i, osm as number[])) n.osm++;
      }
      const total = d.length / 4;
      return { imagery: n.imagery / total, osm: n.osm / total };
    },
    [`data:image/png;base64,${png.toString("base64")}`, IMAGERY, OSM] as const,
  );
}

test("the basemap toggle switches source and attribution", async ({ page }) => {
  await page.goto("about:blank");
  const imageryTile = await solidPng(page, IMAGERY);
  const osmTile = await solidPng(page, OSM);
  await serve(page, geojson.features);
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) =>
    r.fulfill({ contentType: "image/png", body: imageryTile }),
  );
  await page.route("https://tile.openstreetmap.org/**", (r) =>
    r.fulfill({ contentType: "image/png", body: osmTile }),
  );
  await page.goto("/");
  const attribution = page.locator(".maplibregl-ctrl-attrib-inner");
  await expect(page.locator("#count")).toHaveText("1 of 1 shown");

  // Imagery is the default.
  await expect(page.locator("input[value=imagery]")).toBeChecked();
  await expect(attribution).toContainText("Esri");
  await expect(attribution).toContainText("Lantmäteriet");
  await expect(attribution).not.toContainText("OpenStreetMap");
  await expect.poll(async () => (await shares(page)).imagery).toBeGreaterThan(0.5);
  expect((await shares(page)).osm).toBe(0);

  await page.locator("input[value=osm]").check();
  await expect(attribution).toContainText("OpenStreetMap");
  await expect(attribution).toContainText("Lantmäteriet");
  await expect(attribution).not.toContainText("Esri");
  await expect.poll(async () => (await shares(page)).osm).toBeGreaterThan(0.5);
  expect((await shares(page)).imagery).toBe(0);
  await page.screenshot({ path: "test-results/basemaps-osm.png" });

  await page.locator("input[value=imagery]").check();
  await expect(attribution).toContainText("Esri");
  await expect(attribution).not.toContainText("OpenStreetMap");
  await expect.poll(async () => (await shares(page)).imagery).toBeGreaterThan(0.5);
  expect((await shares(page)).osm).toBe(0);
});
