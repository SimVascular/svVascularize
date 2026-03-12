const VTK_JS_URLS = [
  "https://unpkg.com/vtk.js@34.16.3",
  "https://cdn.jsdelivr.net/npm/vtk.js@34.16.3",
  "https://unpkg.com/vtk.js@34.16.3/dist/vtk.js",
  "https://cdn.jsdelivr.net/npm/vtk.js@34.16.3/dist/vtk.js",
  "https://unpkg.com/vtk.js@34.16.3/vtk.js",
  "https://cdn.jsdelivr.net/npm/vtk.js@34.16.3/vtk.js",
];

const THEBE_JS_URLS = [
  "https://unpkg.com/thebe@0.9.2/lib/index.js",
  "https://cdn.jsdelivr.net/npm/thebe@0.9.2/lib/index.js",
];

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === "true") {
        resolve();
        return;
      }
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)), { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolve();
    });
    script.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)));
    document.head.appendChild(script);
  });
}

let vtkPromise = null;
async function getVtk() {
  if (vtkPromise) return vtkPromise;

  vtkPromise = (async () => {
    if (globalThis.vtk) return globalThis.vtk;

    let lastErr = null;
    for (const url of VTK_JS_URLS) {
      try {
        await loadScript(url);
      } catch (err) {
        lastErr = err;
        continue;
      }
      if (globalThis.vtk) return globalThis.vtk;
    }
    throw lastErr || new Error("vtk.js loaded but `window.vtk` is missing");
  })();

  return vtkPromise;
}

let thebePromise = null;
async function getThebe() {
  if (thebePromise) return thebePromise;

  thebePromise = (async () => {
    if (globalThis.thebe || globalThis.thebelab) return globalThis.thebe || globalThis.thebelab;

    let lastErr = null;
    for (const url of THEBE_JS_URLS) {
      try {
        await loadScript(url);
      } catch (err) {
        lastErr = err;
        continue;
      }
      if (globalThis.thebe || globalThis.thebelab) return globalThis.thebe || globalThis.thebelab;
    }

    throw lastErr || new Error("Thebe loaded but global `thebe`/`thebelab` is missing");
  })();

  return thebePromise;
}

function toRgb01(hex) {
  const n = Number(hex) >>> 0;
  const r = ((n >> 16) & 0xff) / 255;
  const g = ((n >> 8) & 0xff) / 255;
  const b = (n & 0xff) / 255;
  return [r, g, b];
}

function buildThebeConfig() {
  // NOTE: Update these if your default branch or org changes.
  const repo = "SimVascular/svVascularize";
  const ref = "main";

  const config = {
    requestKernel: true,
    predefinedOutput: true,
    binderOptions: {
      provider: "github",
      repo,
      ref,
      binderUrl: "https://mybinder.org",
    },
    kernelOptions: {
      name: "python3",
      // Keep kernels in the docs folder so relative paths match the examples.
      path: "docs",
    },
    selector: "#svv-playground .thebe",
    // Thebe has used both underscore and camelCase in different integration examples.
    selector_input: "pre[data-executable]",
    selectorInput: "pre[data-executable]",
    selector_output: ".thebe-output",
    selectorOutput: ".thebe-output",
    codeMirrorConfig: {
      lineNumbers: true,
      mode: "python",
      indentUnit: 4,
      tabSize: 4,
    },
  };

  // Common global names used by integrations (e.g., Jupyter Book).
  globalThis.thebe_config = config;
  globalThis.thebeConfig = config;

  return config;
}

async function bootstrapThebe() {
  const config = buildThebeConfig();
  await getThebe();

  const thebeBootstrap = globalThis.thebe?.bootstrap;
  const thebelabBootstrap = globalThis.thebelab?.bootstrap;
  const bootstrap = typeof thebeBootstrap === "function" ? thebeBootstrap : thebelabBootstrap;
  if (typeof bootstrap !== "function") {
    throw new Error("Thebe bootstrap function not found");
  }

  let result;
  try {
    // Extra args are ignored if the implementation doesn't accept them.
    result = bootstrap(config);
  } catch {
    result = bootstrap();
  }
  if (result && typeof result.then === "function") {
    await result;
  }
}

async function createVtkViewer(container) {
  const vtk = await getVtk();
  if (!container) throw new Error("Missing viewer container");

  container.replaceChildren();

  const full = vtk.Rendering.Misc.vtkFullScreenRenderWindow.newInstance({
    rootContainer: container,
    containerStyle: {
      height: "100%",
      width: "100%",
      position: "relative",
      overflow: "hidden",
    },
    background: [1, 1, 1],
  });
  full.setControllerVisibility?.(false);

  const renderer = full.getRenderer();
  const renderWindow = full.getRenderWindow();

  let domainSize = 1.0;

  function deleteVtk(obj) {
    try {
      obj?.delete?.();
    } catch {
      // ignore
    }
  }

  let domainActor = null;
  let domainMapper = null;
  let domainSource = null;

  function setDomain(kind, size) {
    const domainKind = String(kind) === "box" ? "box" : "sphere";
    domainSize = Number(size) || 1.0;

    if (domainActor) {
      renderer.removeActor(domainActor);
      deleteVtk(domainActor);
      deleteVtk(domainMapper);
      deleteVtk(domainSource);
      domainActor = null;
      domainMapper = null;
      domainSource = null;
    }

    if (domainKind === "box") {
      domainSource = vtk.Filters.Sources.vtkCubeSource.newInstance({
        xLength: domainSize * 2,
        yLength: domainSize * 2,
        zLength: domainSize * 2,
      });
    } else {
      domainSource = vtk.Filters.Sources.vtkSphereSource.newInstance({
        radius: domainSize,
        thetaResolution: 36,
        phiResolution: 18,
      });
    }

    domainMapper = vtk.Rendering.Core.vtkMapper.newInstance();
    domainMapper.setInputConnection(domainSource.getOutputPort());

    domainActor = vtk.Rendering.Core.vtkActor.newInstance();
    domainActor.setMapper(domainMapper);
    const prop = domainActor.getProperty?.();
    prop?.setRepresentationToWireframe?.();
    prop?.setOpacity?.(0.25);
    const [dr, dg, db] = toRgb01(0x1aa3ff);
    prop?.setColor?.(dr, dg, db);

    renderer.addActor(domainActor);
    renderer.resetCamera?.();
    renderWindow.render?.();
  }

  let treeActor = null;
  let treeMapper = null;
  let treeTube = null;
  let treePolyData = null;
  let treeLines = null;

  function clear() {
    if (!treeActor) return;
    renderer.removeActor(treeActor);
    deleteVtk(treeActor);
    deleteVtk(treeMapper);
    deleteVtk(treeTube);
    deleteVtk(treePolyData);
    deleteVtk(treeLines);
    treeActor = null;
    treeMapper = null;
    treeTube = null;
    treePolyData = null;
    treeLines = null;
    renderWindow.render?.();
  }

  function setLines(points, edges) {
    if (!Array.isArray(points) || !Array.isArray(edges)) return;

    const nPoints = points.length;
    if (nPoints === 0 || edges.length === 0) {
      clear();
      return;
    }

    const coords = new Float32Array(nPoints * 3);
    for (let i = 0; i < nPoints; i += 1) {
      const p = points[i] || [0, 0, 0];
      coords[i * 3 + 0] = Number(p[0]) || 0;
      coords[i * 3 + 1] = Number(p[1]) || 0;
      coords[i * 3 + 2] = Number(p[2]) || 0;
    }

    const valid = [];
    for (const e of edges) {
      const a = Number(e?.[0]);
      const b = Number(e?.[1]);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      const ia = a | 0;
      const ib = b | 0;
      if (ia < 0 || ib < 0 || ia >= nPoints || ib >= nPoints) continue;
      if (ia === ib) continue;
      valid.push([ia, ib]);
    }

    if (valid.length === 0) {
      clear();
      return;
    }

    // Cell array for line segments: [2, a, b, 2, a, b, ...]
    const cells = new Uint32Array(valid.length * 3);
    for (let i = 0; i < valid.length; i += 1) {
      const [a, b] = valid[i];
      const off = i * 3;
      cells[off + 0] = 2;
      cells[off + 1] = a;
      cells[off + 2] = b;
    }

    clear();

    treePolyData = vtk.Common.DataModel.vtkPolyData.newInstance();
    treePolyData.getPoints().setData(coords, 3);

    treeLines = vtk.Common.Core.vtkCellArray.newInstance();
    if (typeof treeLines.setData === "function") {
      treeLines.setData(cells);
    } else {
      treeLines = vtk.Common.Core.vtkCellArray.newInstance({ values: cells });
    }
    treePolyData.setLines(treeLines);

    // Render as tubes (more vascular than thin aliased lines).
    treeTube = vtk.Filters.General.vtkTubeFilter.newInstance();
    treeTube.setInputData(treePolyData);
    treeTube.setCapping?.(false);
    treeTube.setNumberOfSides?.(14);
    treeTube.setRadius?.(Math.max(1e-4, domainSize * 0.012));

    treeMapper = vtk.Rendering.Core.vtkMapper.newInstance();
    treeMapper.setInputConnection(treeTube.getOutputPort());

    treeActor = vtk.Rendering.Core.vtkActor.newInstance();
    treeActor.setMapper(treeMapper);
    const prop = treeActor.getProperty?.();
    const [tr, tg, tb] = toRgb01(0x003d6b);
    prop?.setColor?.(tr, tg, tb);

    renderer.addActor(treeActor);
    renderer.resetCamera?.();
    renderWindow.render?.();
  }

  const ro =
    "ResizeObserver" in window
      ? new ResizeObserver(() => {
          full.resize?.();
        })
      : null;
  ro?.observe?.(container);

  // Default wireframe domain.
  setDomain("box", 0.5);

  return { setLines, setDomain, clear };
}

async function createThreeViewer(container) {
  if (!container) throw new Error("Missing viewer container");

  const THREE = await import("https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js");
  const { OrbitControls } = await import(
    "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js"
  );

  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  container.replaceChildren(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
  camera.position.set(2.2, 1.8, 2.2);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dir = new THREE.DirectionalLight(0xffffff, 0.6);
  dir.position.set(2.0, 3.0, 4.0);
  scene.add(dir);

  let domainMesh = null;
  let lines = null;

  function disposeObject(obj) {
    if (!obj) return;
    if (obj.geometry) obj.geometry.dispose?.();
    if (obj.material) obj.material.dispose?.();
  }

  function setDomain(kind, size) {
    if (domainMesh) {
      scene.remove(domainMesh);
      disposeObject(domainMesh);
      domainMesh = null;
    }

    const s = Number(size) || 1.0;
    const geo =
      String(kind) === "box"
        ? new THREE.BoxGeometry(s * 2, s * 2, s * 2)
        : new THREE.SphereGeometry(s, 36, 18);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x1aa3ff,
      wireframe: true,
      transparent: true,
      opacity: 0.18,
    });
    domainMesh = new THREE.Mesh(geo, mat);
    scene.add(domainMesh);
  }

  function fitCameraToObject(obj) {
    if (!obj) return;
    const box = new THREE.Box3().setFromObject(obj);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
    const fov = (camera.fov * Math.PI) / 180.0;
    let distance = (maxDim / 2) / Math.tan(fov / 2);
    distance *= 1.8;

    const v = new THREE.Vector3(1, 0.85, 1).normalize();
    camera.position.copy(center.clone().add(v.multiplyScalar(distance)));
    camera.near = Math.max(0.01, distance / 100);
    camera.far = distance * 100;
    camera.updateProjectionMatrix();

    controls.target.copy(center);
    controls.update();
  }

  function setLines(points, edges) {
    if (!Array.isArray(points) || !Array.isArray(edges)) return;

    if (lines) {
      scene.remove(lines);
      disposeObject(lines);
      lines = null;
    }
    if (edges.length === 0) return;

    const segments = [];
    for (const e of edges) {
      const a = Number(e?.[0]);
      const b = Number(e?.[1]);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      const pa = points[a | 0];
      const pb = points[b | 0];
      if (!pa || !pb) continue;
      segments.push([pa, pb]);
    }
    if (segments.length === 0) return;

    const positions = new Float32Array(segments.length * 2 * 3);
    for (let i = 0; i < segments.length; i += 1) {
      const [pa, pb] = segments[i];
      const off = i * 6;
      positions[off + 0] = Number(pa[0]);
      positions[off + 1] = Number(pa[1]);
      positions[off + 2] = Number(pa[2]);
      positions[off + 3] = Number(pb[0]);
      positions[off + 4] = Number(pb[1]);
      positions[off + 5] = Number(pb[2]);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingSphere();

    const material = new THREE.LineBasicMaterial({ color: 0x003d6b, transparent: true, opacity: 0.95 });
    lines = new THREE.LineSegments(geometry, material);
    scene.add(lines);
    fitCameraToObject(lines);
  }

  function clear() {
    if (!lines) return;
    scene.remove(lines);
    disposeObject(lines);
    lines = null;
  }

  function resize() {
    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();

  const ro = "ResizeObserver" in window ? new ResizeObserver(resize) : null;
  ro?.observe?.(container);

  (function animate() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  })();

  setDomain("box", 0.5);
  return { setLines, setDomain, clear };
}

async function createViewer(container) {
  try {
    return await createVtkViewer(container);
  } catch (err) {
    console.warn("VTK.js viewer failed; falling back to three.js.", err);
    return await createThreeViewer(container);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const playground = document.getElementById("svv-playground");
  if (!playground) return;

  const loadBtn = document.getElementById("svv-play-load");
  const statusEl = document.getElementById("svv-play-status");
  const viewerEl = document.getElementById("svv-play-canvas");
  const outputEl = playground.querySelector(".thebe-output");

  function setStatus(text) {
    if (statusEl) statusEl.textContent = String(text ?? "");
  }

  // Expose status setter for Python via `display(Javascript(...))`.
  globalThis.svvPlaygroundSetStatus = setStatus;

  let viewer = null;
  let started = false;

  function updateOutputVisibility() {
    if (!outputEl) return;
    const text = (outputEl.textContent || "").trim();
    const hasMedia = outputEl.querySelector("img, svg, canvas, pre, code, table") !== null;
    outputEl.classList.toggle("has-content", text.length > 0 || hasMedia);
  }

  if (outputEl && "MutationObserver" in window) {
    updateOutputVisibility();
    const mo = new MutationObserver(updateOutputVisibility);
    mo.observe(outputEl, { childList: true, subtree: true, characterData: true });
  }

  async function ensureViewer() {
    if (viewer) return viewer;
    viewer = await createViewer(viewerEl);

    // Expose render hooks for Python-side JS output.
    globalThis.render_lines = (points, edges) => viewer?.setLines(points, edges);
    globalThis.render_domain = (kind, size) => viewer?.setDomain(kind, size);
    globalThis.playground_clear = () => viewer?.clear();

    return viewer;
  }

  async function startSession() {
    if (started) return;
    started = true;

    loadBtn.disabled = true;
    setStatus("Launching Binder… (can take 1–3 min)");

    try {
      await ensureViewer();
      await bootstrapThebe();
      setStatus("Session ready — run the cell");
    } catch (err) {
      console.error(err);
      setStatus("Failed to start (see DevTools)");
      started = false;
      loadBtn.disabled = false;
    }
  }

  loadBtn?.addEventListener("click", startSession);
});
