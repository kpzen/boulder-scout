import Delaunator from "delaunator";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { type Range, readRange } from "./bundle";

// 3D view of one candidate's patch. Ground points become a green
// Delaunay mesh. Rock points become a grey surface of their own (never joined to the ground, so
// no false ramps), with triangles no longer than MAX_EDGE so separate pieces stay apart, and
// walls dropped from its outer edges to the ground. Rock points in no triangle stay grey dots.
// Vegetation (point_type 2) is faint dots: canopy around and over a candidate is what
// tells a floating crown from a solid rock.

const MAX_EDGE = 2.0; // m, horizontal: longer rock triangles bridge gaps rather than rock

const panel = document.getElementById("view3d")!;
const caption = document.getElementById("caption3d")!;
const renderer = new THREE.WebGLRenderer({ antialias: true });
panel.append(renderer.domElement);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
let scene = new THREE.Scene();
const render = () => renderer.render(scene, camera);
controls.addEventListener("change", render);
document.getElementById("close3d")!.addEventListener("click", () => (panel.hidden = true));
// The panel can be resized from its corner (CSS resize); the scene follows.
new ResizeObserver(() => {
  if (panel.hidden || renderer.domElement.style.display === "none") return;
  renderer.setSize(panel.clientWidth, panel.clientHeight);
  camera.aspect = panel.clientWidth / panel.clientHeight;
  camera.updateProjectionMatrix();
  render();
}).observe(panel);

const groundMaterial = new THREE.MeshLambertMaterial({
  color: 0x7f9f5f,
  transparent: true,
  opacity: 0.6,
  side: THREE.DoubleSide,
});
const rockMaterial = new THREE.MeshLambertMaterial({
  color: 0x8e8b86,
  flatShading: true,
  side: THREE.DoubleSide,
});
// Walls are inferred (airborne LiDAR sees no faces under an edge), so they are see-through:
// ground or foliage under a floating crown stays visible behind them.
const wallMaterial = new THREE.MeshLambertMaterial({
  color: 0x8e8b86,
  flatShading: true,
  side: THREE.DoubleSide,
  transparent: true,
  opacity: 0.45,
  depthWrite: false,
});
const looseRockMaterial = new THREE.PointsMaterial({ color: 0x6e6b66, size: 6, sizeAttenuation: false });
const vegetationMaterial = new THREE.PointsMaterial({
  color: 0x14532d,
  size: 5,
  sizeAttenuation: false,
  transparent: true,
  opacity: 0.7,
});

export async function openPatch(tile: string, patch: Range | undefined, title: string): Promise<void> {
  // A patch that fails to load opens the panel with the error and no (stale) scene.
  let buf: ArrayBuffer;
  try {
    if (!patch) throw new Error("not in the export");
    buf = await readRange(`${import.meta.env.BASE_URL}tiles/${tile}.boulders.bin`, patch);
  } catch (e) {
    caption.textContent = `${title} · could not load patch: ${e instanceof Error ? e.message : String(e)}`;
    renderer.domElement.style.display = "none";
    panel.hidden = false;
    return;
  }
  renderer.domElement.style.display = "block";

  // 8 bytes per point, little-endian dx, dy, dz int16 cm, intensity, point_type.
  // Three.js is y-up: east -> x, up -> y, north -> -z.
  const view = new DataView(buf);
  const ground: THREE.Vector3[] = [];
  const rock: THREE.Vector3[] = [];
  const vegetation: THREE.Vector3[] = [];
  for (let o = 0; o < buf.byteLength; o += 8) {
    const p = new THREE.Vector3(
      view.getInt16(o, true) / 100,
      view.getInt16(o + 4, true) / 100,
      -view.getInt16(o + 2, true) / 100,
    );
    const type = view.getUint8(o + 7);
    if (type === 0) ground.push(p);
    else if (type === 1) rock.push(p);
    else vegetation.push(p);
  }

  scene.traverse((o) => {
    if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments || o instanceof THREE.Points) {
      o.geometry.dispose();
    }
  });
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f4f0);
  scene.add(new THREE.AmbientLight(0xffffff, 1.5));
  const sun = new THREE.DirectionalLight(0xffffff, 2);
  sun.position.set(1, 2, 0.5);
  scene.add(sun);

  const tri = Delaunator.from(ground, (p) => p.x, (p) => p.z);
  const groundGeometry = new THREE.BufferGeometry().setFromPoints(ground);
  groundGeometry.setIndex(Array.from(tri.triangles));
  groundGeometry.computeVertexNormals();
  const groundMesh = new THREE.Mesh(groundGeometry, groundMaterial);
  scene.add(groundMesh);

  scene.add(new THREE.Points(new THREE.BufferGeometry().setFromPoints(vegetation), vegetationMaterial));

  // Ground height under each rock point (null if it falls outside the ground mesh).
  const ray = new THREE.Raycaster();
  const down = new THREE.Vector3(0, -1, 0);
  const floor = rock.map((p) => {
    ray.set(new THREE.Vector3(p.x, 1000, p.z), down);
    return ray.intersectObject(groundMesh)[0]?.point.y ?? null;
  });

  // Rock surface: plan-view Delaunay over the rock points alone, short triangles only.
  const rt = Delaunator.from(rock, (p) => p.x, (p) => p.z);
  const short = (a: number, b: number) =>
    Math.hypot(rock[a].x - rock[b].x, rock[a].z - rock[b].z) <= MAX_EDGE;
  const index: number[] = [];
  const edgeUse = new Map<string, [number, number, number]>();
  for (let t = 0; t < rt.triangles.length; t += 3) {
    const [a, b, c] = [rt.triangles[t], rt.triangles[t + 1], rt.triangles[t + 2]];
    if (!(short(a, b) && short(b, c) && short(c, a))) continue;
    index.push(a, b, c);
    for (const [u, w] of [[a, b], [b, c], [c, a]]) {
      const key = u < w ? `${u},${w}` : `${w},${u}`;
      const use = edgeUse.get(key);
      edgeUse.set(key, use ? [use[0], use[1], use[2] + 1] : [u, w, 1]);
    }
  }
  const triangles = index.length / 3;
  // Walls: each outer edge (in one triangle only) drops to the ground under its two ends.
  // Vertex i + n is rock point i lowered to the ground.
  const n = rock.length;
  let walls = 0;
  for (const [u, w, use] of edgeUse.values()) {
    if (use !== 1 || floor[u] === null || floor[w] === null) continue;
    index.push(u, w, w + n, u, w + n, u + n);
    walls++;
  }
  const lowered = rock.map((p, i) => new THREE.Vector3(p.x, floor[i] ?? p.y, p.z));
  const rockGeometry = new THREE.BufferGeometry().setFromPoints([...rock, ...lowered]);
  rockGeometry.setIndex(index);
  rockGeometry.addGroup(0, triangles * 3, 0);
  rockGeometry.addGroup(triangles * 3, walls * 6, 1);
  rockGeometry.computeVertexNormals();
  scene.add(new THREE.Mesh(rockGeometry, [rockMaterial, wallMaterial]));
  const inMesh = new Set(index.slice(0, triangles * 3));
  const loose = rock.filter((_, i) => !inMesh.has(i));
  scene.add(new THREE.Points(new THREE.BufferGeometry().setFromPoints(loose), looseRockMaterial));

  caption.textContent = `${title} · ${ground.length} ground points · ${rock.length} rock points · ${triangles} rock triangles · ${walls} wall quads · ${loose.length} loose rock points · ${vegetation.length} vegetation points`;
  panel.hidden = false;
  renderer.setSize(panel.clientWidth, panel.clientHeight);
  camera.aspect = panel.clientWidth / panel.clientHeight;
  camera.updateProjectionMatrix();
  // Frame ground and rock together, so a tall rock's top stays in view: the camera sits on a
  // fixed diagonal. Their bounding sphere is loose (the ground disc's corners are out
  // of shot from this angle), so it is fitted at 0.65 of the distance that would hold it all.
  const bounds = new THREE.Box3().setFromPoints([...ground, ...rock]);
  const sphereFit = bounds.getBoundingSphere(new THREE.Sphere());
  const fov = THREE.MathUtils.degToRad(Math.min(camera.fov, camera.fov * camera.aspect)) / 2;
  const dist = (0.65 * sphereFit.radius) / Math.sin(fov);
  camera.position
    .copy(sphereFit.center)
    .addScaledVector(new THREE.Vector3(1, 0.8, 1).normalize(), dist);
  controls.target.copy(sphereFit.center);
  controls.update();
  render();
}
