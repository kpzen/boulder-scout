import type { GeoJSONSource, Map, MapMouseEvent } from "maplibre-gl";
import vocabulary from "../../scoring/label_vocabulary.json";
import type { Boulders } from "./main";
import { photoButton } from "./photos";

// Review mode (?review): label candidates the author picks in each score band, and
// mark boulders that have no candidate. Every change rewrites data/labels/labels.json through the
// dev server's /labels.json endpoint, the one write layer 3 makes.

type Feature = Boulders["features"][number];

// The classes and tags are the fit's too. Boulder's tags (its height) are one choice,
// vegetation's any number. More only once a cue keeps recurring in the notes.
const { classes: CLASSES, tags: TAGS }: { classes: string[]; tags: Record<string, { multi: boolean; tags: string[] }> } =
  vocabulary;
// Each score band's target, agreed with the author: TARGET positives (boulder) and TARGET
// negatives (any other class but unsure). The exported range only (score >= 2.0),
// finer where most candidates are; counts are this run's labels, so earlier runs' do not fill them.
const BANDS = [2, 3, 4, 5, 6, 7, Infinity];
const TARGET = 10;

// A candidate's label also keeps its exported values at labelling time (score, area_m2, the
// filter features), so it can be matched by position and compared after a re-run.
type Label = {
  candidate_id: string | null; // null for a missed boulder (class boulder)
  E: number; // EPSG:3006
  N: number;
  lon: number;
  lat: number;
  class: string;
  tags: string[];
  note: string;
  run: string; // the first 12 hex digits of the SHA-256 of the tiles' heads, in tiles.json's order
  time: string;
  [value: string]: unknown;
};

export async function startReview(
  map: Map,
  data: Boulders,
  text: string,
  info: HTMLElement,
  setBand: (band: [number, number] | null) => void,
) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  const run = [...digest.slice(0, 6)].map((b) => b.toString(16).padStart(2, "0")).join("");
  const res = await fetch("/labels.json");
  const labels: Label[] = res.ok ? await res.json() : [];

  // The band table and the missed-boulder button sit under the sliders.
  const bar = document.createElement("div");
  bar.id = "review";
  const rows = BANDS.slice(1).map((hi, i) => {
    const lo = BANDS[i];
    const name = hi === Infinity ? `≥ ${lo}` : `${lo} – ${hi}`;
    return `<tr data-band="${i}"><td>${name}</td><td></td><td></td></tr>`;
  });
  bar.innerHTML = `<table><tr><th>score</th><th>boulder</th><th>not</th></tr>${rows.join("")}</table><button>Mark missed boulder</button> <span class="status">${res.ok ? "" : "no labels endpoint: nothing will be saved"}</span><div class="step"><button>‹</button> <span></span> <button>›</button></div>`;
  document.getElementById("filters")!.append(bar);
  const mark = bar.querySelector("button")!;

  // Step through the labelled candidates, a class at a time (also with the arrow keys), each
  // with its aerial photo open. A label from an earlier run may have no candidate in this
  // one, so the label itself is shown.
  const byId = Object.fromEntries(data.features.map((f) => [f.properties.candidate_id, f.properties]));
  const stepped = labels
    .filter((l) => l.candidate_id !== null)
    .sort((a, b) => CLASSES.indexOf(a.class) - CLASSES.indexOf(b.class));
  const [back, next] = bar.querySelectorAll<HTMLButtonElement>(".step button");
  const where = bar.querySelector(".step span")!;
  where.textContent = `${stepped.length} labelled`;
  let at = -1;
  function step(by: number): void {
    if (!stepped.length) return;
    at = (at + by + stepped.length) % stepped.length;
    const l = stepped[at];
    where.textContent = `${at + 1} / ${stepped.length}`;
    map.jumpTo({ center: [l.lon, l.lat], zoom: Math.max(map.getZoom(), 17) });
    // The photo panel can cover the bar's buttons, so the info panel gets its own.
    info.innerHTML = `<b>${l.candidate_id}</b><div>${l.class}${l.tags.length ? ` · ${l.tags.join(", ")}` : ""}</div><div>${l.note}</div><div class="buttons"><button>‹</button><button>›</button></div>`;
    info.hidden = false;
    const [infoBack, infoNext] = info.querySelectorAll<HTMLButtonElement>(".buttons button");
    infoBack.onclick = () => step(-1);
    infoNext.onclick = () => step(1);
    const button = photoButton(l.E, l.N, byId[l.candidate_id!]);
    info.querySelector(".buttons")!.append(button);
    if (document.getElementById("photos")!.hidden) button.click();
  }
  back.onclick = () => step(-1);
  next.onclick = () => step(1);
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLTextAreaElement) return;
    if (e.key === "ArrowRight") step(1);
    if (e.key === "ArrowLeft") step(-1);
  });
  const status = bar.querySelector(".status")!;
  const bandRows = [...bar.querySelectorAll<HTMLTableRowElement>("tr[data-band]")];
  function showCounts(): void {
    bandRows.forEach((row, i) => {
      const inBand = labels.filter(
        (l) =>
          l.candidate_id !== null &&
          l.run === run &&
          (l.score as number) >= BANDS[i] &&
          (l.score as number) < BANDS[i + 1],
      );
      const pos = inBand.filter((l) => l.class === "boulder").length;
      const neg = inBand.filter((l) => l.class !== "boulder" && l.class !== "unsure").length;
      row.cells[1].textContent = `${pos}/${TARGET}`;
      row.cells[2].textContent = `${neg}/${TARGET}`;
    });
  }
  showCounts();
  // A click on a band shows only its candidates; a second click shows all again.
  bandRows.forEach((row, i) => {
    row.onclick = () => {
      const on = !row.classList.contains("on");
      for (const r of bandRows) r.classList.remove("on");
      row.classList.toggle("on", on);
      setBand(on ? [BANDS[i], BANDS[i + 1]] : null);
    };
  });

  const missed = () => ({
    type: "FeatureCollection" as const,
    features: labels.flatMap((l, i) =>
      l.candidate_id === null
        ? [{ type: "Feature" as const, geometry: { type: "Point" as const, coordinates: [l.lon, l.lat] }, properties: { i } }]
        : [],
    ),
  });
  map.addSource("missed", { type: "geojson", data: missed() });
  map.addLayer({
    id: "missed",
    type: "circle",
    source: "missed",
    paint: { "circle-radius": 8, "circle-color": "#2166ac", "circle-stroke-color": "#fff", "circle-stroke-width": 2 },
  });

  // Saves run one after another, so the file always ends with the latest labels.
  let saving = Promise.resolve();
  function save(): void {
    (map.getSource("missed") as GeoJSONSource).setData(missed());
    showCounts();
    const [body, n] = [JSON.stringify(labels, null, 1), labels.length];
    saving = saving.then(async () => {
      try {
        const r = await fetch("/labels.json", { method: "PUT", body });
        status.textContent = r.ok ? `saved ${n} labels` : `save failed: HTTP ${r.status}`;
      } catch (err) {
        status.textContent = `save failed: ${err}`;
      }
    });
  }

  // The review panel goes under the candidate info: a class, then its tags and a note.
  function show(f: Feature): void {
    const id = f.properties.candidate_id;
    const [E, N] = id.split("_").slice(-2).map(Number); // {tile_id}_{x:.1f}_{y:.1f}
    let label = labels.find((l) => l.candidate_id === id);
    const panel = document.createElement("div");
    panel.id = "review-panel";
    panel.innerHTML =
      CLASSES.map(
        (c) => `<label><input type="radio" name="class" value="${c}"${label?.class === c ? " checked" : ""}> ${c}</label>`,
      ).join("") +
      `<div class="tags"></div><textarea name="note" rows="3" placeholder="note"></textarea>`;
    info.append(panel);
    const tags = panel.querySelector(".tags")!;
    const note = panel.querySelector("textarea")!;
    function render(): void {
      const t = label && TAGS[label.class];
      tags.innerHTML = t
        ? t.tags
            .map(
              (tag) =>
                `<label><input type="${t.multi ? "checkbox" : "radio"}" name="tag" value="${tag}"${label!.tags.includes(tag) ? " checked" : ""}> ${tag}</label>`,
            )
            .join("")
        : "";
      note.value = label?.note ?? "";
      note.disabled = !label; // a label starts with its class
    }
    render();
    panel.addEventListener("change", (e) => {
      const input = e.target as HTMLInputElement;
      if (input.name === "class") {
        if (!label) {
          const [lon, lat] = f.geometry.coordinates;
          label = { ...f.properties, E, N, lon, lat, class: "", tags: [], note: "", run, time: "" };
          for (const k of ["tile", "patch", "photo"]) delete label[k]; // where its files are
          labels.push(label);
        }
        label.class = input.value;
        label.tags = [];
        render();
      } else if (input.name === "tag") {
        label!.tags = [...tags.querySelectorAll<HTMLInputElement>("input:checked")].map((i) => i.value);
      } else {
        label!.note = note.value;
      }
      label!.time = new Date().toISOString();
      save();
    });
  }

  function showMissed(label: Label): void {
    info.innerHTML = `<b>Missed boulder</b><div>E ${label.E.toFixed(1)} · N ${label.N.toFixed(1)}</div><textarea name="note" rows="3" placeholder="note"></textarea><button>Delete</button> <button>Close</button>`;
    info.hidden = false;
    const note = info.querySelector("textarea")!;
    const [del, close] = info.querySelectorAll("button");
    info.querySelector("div")!.append(" ", photoButton(label.E, label.N));
    note.value = label.note;
    note.onchange = () => {
      label.note = note.value;
      label.time = new Date().toISOString();
      save();
    };
    del.onclick = () => {
      labels.splice(labels.indexOf(label), 1);
      info.hidden = true;
      save();
    };
    close.onclick = () => (info.hidden = true);
  }

  let placing = false;
  mark.onclick = () => {
    placing = true;
    map.getCanvas().style.cursor = "crosshair";
  };

  // Placing a missed boulder and clicking one come before the candidate click; true if handled.
  function click(e: MapMouseEvent): boolean {
    if (placing) {
      placing = false;
      map.getCanvas().style.cursor = "";
      const { lng, lat } = e.lngLat;
      // WGS84 to SWEREF 99 TM (EPSG:3006) by Lantmäteriet's Gauss–Krüger formulas: GRS80,
      // central meridian 15°, scale 0.9996, false easting 500 km. The e2e test checks it with pyproj.
      const fl = 1 / 298.257222101;
      const [e2, n] = [fl * (2 - fl), fl / (2 - fl)];
      const k = 0.9996 * (6378137 / (1 + n)) * (1 + n ** 2 / 4 + n ** 4 / 64);
      const [A, B, C, D] = [e2, (5 * e2 ** 2 - e2 ** 3) / 6, (104 * e2 ** 3 - 45 * e2 ** 4) / 120, (1237 * e2 ** 4) / 1260];
      const phi = (lat * Math.PI) / 180;
      const s = Math.sin(phi) ** 2;
      const phiStar = phi - Math.sin(phi) * Math.cos(phi) * (A + B * s + C * s ** 2 + D * s ** 3);
      const dl = ((lng - 15) * Math.PI) / 180;
      const xi = Math.atan(Math.tan(phiStar) / Math.cos(dl));
      const eta = Math.atanh(Math.cos(phiStar) * Math.sin(dl));
      const beta = [
        n / 2 - (2 * n ** 2) / 3 + (5 * n ** 3) / 16 + (41 * n ** 4) / 180,
        (13 * n ** 2) / 48 - (3 * n ** 3) / 5 + (557 * n ** 4) / 1440,
        (61 * n ** 3) / 240 - (103 * n ** 4) / 140,
        (49561 * n ** 4) / 161280,
      ];
      const N = k * (xi + beta.reduce((sum, b, j) => sum + b * Math.sin(2 * (j + 1) * xi) * Math.cosh(2 * (j + 1) * eta), 0));
      const E = 500000 + k * (eta + beta.reduce((sum, b, j) => sum + b * Math.cos(2 * (j + 1) * xi) * Math.sinh(2 * (j + 1) * eta), 0));
      const label: Label = {
        candidate_id: null,
        E,
        N,
        lon: lng,
        lat,
        class: "boulder",
        tags: [],
        note: "",
        run,
        time: new Date().toISOString(),
      };
      labels.push(label);
      save();
      showMissed(label);
      return true;
    }
    const hit = map.queryRenderedFeatures(e.point, { layers: ["missed"] })[0];
    if (hit) showMissed(labels[hit.properties.i]);
    return hit !== undefined;
  }

  return { show, click };
}
