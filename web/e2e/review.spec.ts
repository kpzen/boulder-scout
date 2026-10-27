import { execFileSync } from "node:child_process";
import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import vocabulary from "../../scoring/label_vocabulary.json" with { type: "json" };
import { boulders, serve } from "./export";

// One fixture candidate. fit() puts it at the centre of the padded map (main.ts: left 260,
// right 80; index.html: a 48 px bar above the map, 8 px below), and its pin's head is about
// 25 px above it, where a click opens it.
const all = boulders(process.env.FIXTURE_EXPORT!);
const one = all.find((f) => f.properties.score >= 2)!;
const centre = { x: (260 + 1280 - 80) / 2, y: 48 + (720 - 48 - 8) / 2 - 25 };

test.beforeEach(async ({ page }) => {
  await serve(page, [one]);
  await page.route("https://tile.openstreetmap.org/**", (r) => r.abort()); // no network in tests
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) => r.abort());
});

test("review mode labels a candidate and a missed boulder into the labels file", async ({ page }) => {
  const file = process.env.LABELS_FILE!;
  rmSync(file, { force: true });
  await page.goto("/?review");
  const count = page.locator("#count");
  await expect(count).toHaveText("1 of 1 shown");
  const bar = page.locator("#review");
  const status = bar.locator(".status");
  await page.mouse.click(centre.x, centre.y);

  // A class first, then one of its tags and a note; each change saves the file.
  const info = page.locator("#info");
  const id = one.properties.candidate_id;
  await expect(info.locator("b")).toHaveText(id);
  const [x, y] = id.split("_").slice(-2).map(Number);
  // The panel offers the label vocabulary's classes, the ones the fit reads.
  const offered = await info.locator("input[name=class]").evaluateAll((els) => els.map((e) => (e as HTMLInputElement).value));
  expect(offered).toEqual(vocabulary.classes);
  await info.getByLabel("vegetation", { exact: true }).check();
  await info.getByLabel("jagged top").check();
  await info.locator("textarea").fill("starfish arms");
  await info.locator("textarea").blur();

  // The candidate's band counts it as a negative; a band click shows only that band's
  // candidates, in place of the score slider, and a second click shows all again.
  const band = Math.min(5, Math.floor(one.properties.score) - 2); // bands 2–3, 3–4, …, ≥ 7
  const rows = bar.locator("tr[data-band]");
  await expect(rows.nth(band).locator("td")).toHaveText([/./, "0/10", "1/10"]);
  await rows.nth(band === 0 ? 1 : 0).click();
  await expect(count).toHaveText("0 of 1 shown");
  await rows.nth(band).click();
  await expect(count).toHaveText("1 of 1 shown");
  await page.screenshot({ path: "test-results/review.png" });
  await rows.nth(band).click();

  // A missed boulder goes where the next map click lands.
  await bar.getByRole("button", { name: "Mark missed boulder" }).click();
  await page.mouse.click(640, 550);
  await expect(info.locator("b")).toHaveText("Missed boulder");
  await expect(status).toHaveText("saved 2 labels");

  const [candidate, missed] = JSON.parse(readFileSync(file, "utf8"));
  expect(candidate).toMatchObject({
    candidate_id: id,
    class: "vegetation",
    tags: ["jagged top"],
    note: "starfish arms",
    score: one.properties.score,
  });
  expect([candidate.E, candidate.N]).toEqual([x, y]);
  expect(candidate.run).toMatch(/^[0-9a-f]{12}$/);
  expect(missed).toMatchObject({ candidate_id: null, class: "boulder", run: candidate.run });

  // The client's own WGS84 → SWEREF 99 TM agrees with pyproj to the millimetre.
  const [E, N] = execFileSync(
    "uv",
    ["run", "python", "-c",
      "import sys; from pyproj import Transformer; t = Transformer.from_crs(4326, 3006, always_xy=True); print(*t.transform(float(sys.argv[1]), float(sys.argv[2])))",
      String(missed.lon), String(missed.lat)],
    { cwd: resolve(import.meta.dirname, "../..") },
  ).toString().trim().split(" ").map(Number);
  expect(Math.abs(missed.E - E)).toBeLessThan(0.001);
  expect(Math.abs(missed.N - N)).toBeLessThan(0.001);
});

// A 1 x 1 PNG standing in for the exported colour crop.
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

test("without ?review there is no review panel; the aerial photo is the exported crop", async ({ page }) => {
  // The viewer asks no photo service; the crop comes from the export, its tile's photos
  // file.
  const external: string[] = [];
  await page.route("https://minkarta.lantmateriet.se/**", (r) => (external.push(r.request().url()), r.abort()));
  const crops: string[] = [];
  await page.route("**/tiles/*.photos.bin", (r) => (crops.push(r.request().url()), r.fulfill({ body: PNG })));
  await serve(page, [{ ...one, properties: { ...one.properties, photo: [0, PNG.length] } }]);
  await page.goto("/");
  await expect(page.locator("#count")).toHaveText("1 of 1 shown");
  await page.mouse.click(centre.x, centre.y);
  const info = page.locator("#info");
  await expect(info).toBeVisible();
  await expect(page.locator("#review-panel")).toHaveCount(0);
  await expect(page.locator("#review")).toHaveCount(0);

  // The photo loads only once asked for, with the Min karta link beside it.
  const id = one.properties.candidate_id;
  const [x, y] = id.split("_").slice(-2).map(Number);
  expect(crops).toEqual([]);
  await info.getByRole("button", { name: "Aerial photos" }).click();
  const photos = page.locator("#photos");
  await expect(photos.locator("img")).toHaveCount(1);
  await expect.poll(() => crops).toEqual([expect.stringContaining("/tiles/e2e.photos.bin")]);
  await expect(photos).toContainText("© Lantmäteriet");
  await expect(photos.getByRole("link", { name: "Min karta" })).toHaveAttribute(
    "href",
    new RegExp(`\\?e=${Math.round(x)}&n=${Math.round(y)}&.*mapprofile=flygbild`),
  );
  expect(external).toEqual([]);
  await photos.getByRole("button", { name: "Close" }).click();
  await expect(photos).toBeHidden();
});

test("a candidate without an exported crop gets the Min karta link only", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#count")).toHaveText("1 of 1 shown");
  await page.mouse.click(centre.x, centre.y);
  await page.locator("#info").getByRole("button", { name: "Aerial photos" }).click();
  const photos = page.locator("#photos");
  await expect(photos.getByRole("link", { name: "Min karta" })).toBeVisible();
  await expect(photos.locator("img")).toHaveCount(0);
});

test("review mode steps through the labelled candidates with their photos open", async ({ page }) => {
  const [lon, lat] = one.geometry.coordinates;
  const label = (id: string, cls: string, E: number, N: number) => ({
    candidate_id: id, E, N, lon, lat, class: cls, tags: [], note: "", run: "", time: "",
  });
  writeFileSync(
    process.env.LABELS_FILE!,
    JSON.stringify([
      label("t_495000.0_6925000.0", "vegetation", 495000, 6925000),
      label(null as unknown as string, "boulder", 495100, 6925100),
      label("t_495200.0_6925200.0", "boulder", 495200, 6925200),
    ]),
  );
  await page.goto("/?review");
  const step = page.locator("#review .step");
  await expect(step).toContainText("2 labelled");
  // Boulders first; each step opens the label in the info panel with its photos.
  await step.getByRole("button", { name: "›" }).click();
  const info = page.locator("#info");
  await expect(info.locator("b")).toHaveText("t_495200.0_6925200.0");
  await expect(info).toContainText("boulder");
  await expect(page.locator("#photos")).toBeVisible();
  await expect(step).toContainText("1 / 2");
  await page.keyboard.press("ArrowRight");
  await expect(info.locator("b")).toHaveText("t_495000.0_6925000.0");
  await info.getByRole("button", { name: "‹" }).click();
  await expect(info.locator("b")).toHaveText("t_495200.0_6925200.0");
  await expect(page.locator("#photos")).toBeVisible();
});
