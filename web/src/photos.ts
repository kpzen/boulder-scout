import { type Range, readRange } from "./bundle";

// The aerial photo of a candidate: the colour crop the export keeps in its tile's
// photos file, a WebP of a 40 m square around its centroid (scripts/measure_photos.py),
// and a link to Min karta's photo map; small ticks at the edges mark the centre. The viewer asks
// no photo service itself, and a point without an exported crop, or an export without photo
// files, gets the link only.
const PHOTO_M = 40;

// The photo panel sits left of the info panel; while open it follows the shown point, so the
// photo loads only when asked for.
const photos = document.createElement("div");
photos.id = "photos";
photos.hidden = true;
document.getElementById("main")!.append(photos);
type Crop = { tile: string; photo?: Range };
let photosAt: [number, number, Crop?] = [0, 0];
function showPhotos(): void {
  const [E, N, crop] = photosAt;
  const img = photos.querySelector("img");
  if (img) URL.revokeObjectURL(img.src);
  photos.innerHTML =
    (crop?.photo
      ? `<figure><div class="photo"><img alt="aerial photo" width="250" height="250"></div><figcaption>${PHOTO_M} m · Datakälla: Lantmäteriets Min karta, © Lantmäteriet, bearbetad, CC BY 4.0</figcaption></figure>`
      : "") +
    `<a href="https://minkarta.lantmateriet.se/plats/3006/v2.0/?e=${Math.round(E)}&n=${Math.round(N)}&z=13&mapprofile=flygbild&layers=%5B%5B%227%22%5D%5D" target="_blank" rel="noopener">Min karta</a> <button>Close</button>`;
  photos.querySelector("button")!.onclick = () => (photos.hidden = true);
  const figure = photos.querySelector("figure");
  if (!crop?.photo || !figure) return;
  readRange(`${import.meta.env.BASE_URL}tiles/${crop.tile}.photos.bin`, crop.photo).then(
    (buf) => (figure.querySelector("img")!.src = URL.createObjectURL(new Blob([buf], { type: "image/webp" }))),
    () => figure.remove(),
  );
}

// The button that opens the photo panel at E, N (EPSG:3006), with the candidate's crop if the
// export has one; an open panel moves there at once.
export function photoButton(E: number, N: number, crop?: Crop): HTMLButtonElement {
  photosAt = [E, N, crop];
  if (!photos.hidden) showPhotos();
  const button = document.createElement("button");
  button.textContent = "Aerial photos";
  // A second click closes the panel.
  button.onclick = () => {
    photos.hidden = !photos.hidden;
    if (!photos.hidden) showPhotos();
  };
  return button;
}
