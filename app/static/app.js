/* GeoPulse frontend.
   Every number rendered here comes from the API, which calls the same tools the
   assistant does — so the cards and the chat can never disagree. */

const $ = (id) => document.getElementById(id);
const state = {
  grid: "h3-8",
  model: "lightgbm_final",
  horizon: 4,
  when: null,
  place: null,
  history: [],
  config: null,
  maxPickups: 1,
  busy: false,
};

/* ── map ─────────────────────────────────────────────────────────── */
/* Vector basemaps, not raster. The reason is concrete: with a vector style the
   choropleth can be inserted *beneath* the label layers, so street and place names
   stay legible on top of the data. A raster basemap is one flat image, so anything
   drawn over it necessarily buries the labels. */

const prefersDark = matchMedia("(prefers-color-scheme: dark)").matches;

/** Palette derived from the active basemap's own theme, not the OS, so a dark
 *  basemap under a light OS still gets readable overlay colours. */
let theme = "light";        // Liberty is a light style; the switcher updates this
const palette = () => (theme === "dark"
  ? { cell0: "#1b2b26", cell1: "#22513f", accent: "#54e3b3", route: "#7ef0c8",
      focus: "#54e3b3" }
  : { cell0: "#dbe7e2", cell1: "#b9dccd", accent: "#0b7a5a", route: "#0b7a5a",
      focus: "#0b7a5a" });

const map = new maplibregl.Map({
  container: "map",
  // Always the full-detail street map. Keying this off the OS theme sent dark-mode
  // users to a minimal basemap with no buildings, shops or POI names, which reads
  // as a broken map rather than a styling choice.
  style: "https://tiles.openfreemap.org/styles/liberty",
  center: [-73.978, 40.745],
  zoom: 11.7,
  attributionControl: { compact: true },
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

let marker = null;
let geometry = { type: "FeatureCollection", features: [] };
let routeData = null;
let blinkRegion = null;
let stationsFC = { type: "FeatureCollection", features: [] };

/** The id of the first symbol layer, so overlays slot in underneath the labels. */
function firstSymbolLayer() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const symbol = layers.find((l) => l.type === "symbol");
  return symbol ? symbol.id : undefined;
}

/** Every source and layer this app owns. Re-runnable: setStyle() wipes the style,
 *  taking custom layers with it, so this is called again after each switch. */
function addOverlays() {
  const c = palette();
  const under = firstSymbolLayer();
  if (map.getSource("cells")) return;

  map.addSource("cells", { type: "geojson", data: geometry });
  map.addLayer({
    id: "cells-fill", type: "fill", source: "cells",
    // `pickups` is a 0-4 log-scaled index, not a raw count (see refresh()): a few
    // Midtown cells carry most of the demand, so a linear ramp would render the
    // rest of the city as one flat colour
    paint: {
      "fill-color": [
        "interpolate", ["linear"], ["coalesce", ["get", "pickups"], 0],
        0, c.cell0, 0.8, c.cell1, 1.8, "#86c9ae",
        2.6, "#2fbc8d", 3.3, "#f2a65a", 4, "#e8590c",
      ],
      // starts invisible: the heatmap is a lens you turn on, not wallpaper. It
      // otherwise covers the buildings, shop names and street labels that make the
      // map worth looking at.
      "fill-opacity": 0,
      "fill-opacity-transition": { duration: 450 },
    },
  }, under);
  map.addLayer({
    id: "cells-line", type: "line", source: "cells",
    paint: {
      "line-color": c.accent, "line-width": 0.4, "line-opacity": 0,
      "line-opacity-transition": { duration: 450 },
    },
  }, under);
  // the focused cell pulses translucently rather than filling solid, so the
  // streets underneath stay readable while the eye is drawn to it
  map.addLayer({
    id: "cells-blink", type: "fill", source: "cells",
    filter: ["==", ["get", "region_id"], "__none__"],
    paint: { "fill-color": c.focus, "fill-opacity": 0.2 },
  }, under);
  map.addLayer({
    id: "cells-focus", type: "line", source: "cells",
    filter: ["==", ["get", "region_id"], "__none__"],
    paint: { "line-color": c.focus, "line-width": 2.6, "line-opacity": 0.9 },
  }, under);

  map.addSource("route", { type: "geojson", data: emptyFC() });
  map.addLayer({
    id: "route-glow", type: "line", source: "route",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": c.route, "line-width": 16, "line-opacity": 0.2, "line-blur": 11 },
  }, under);
  map.addLayer({
    id: "route-line", type: "line", source: "route",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": c.route, "line-width": 5.5, "line-opacity": 0.95 },
  }, under);
  map.addLayer({
    id: "route-dash", type: "line", source: "route",
    layout: { "line-cap": "butt", "line-join": "round" },
    paint: {
      "line-color": theme === "dark" ? "#06231a" : "#ffffff",
      "line-width": 2.4, "line-opacity": 0.9, "line-dasharray": [0, 4, 3],
    },
  }, under);
  map.addSource("route-points", { type: "geojson", data: emptyFC() });
  map.addLayer({
    id: "route-points", type: "circle", source: "route-points",
    paint: {
      "circle-radius": ["case", ["==", ["get", "kind"], "target"], 8, 5],
      "circle-color": ["case", ["==", ["get", "kind"], "target"], c.route, "#ffffff"],
      "circle-stroke-width": 2.5,
      "circle-stroke-color": ["case", ["==", ["get", "kind"], "target"],
        theme === "dark" ? "#06231a" : "#ffffff", c.route],
    },
  }, under);

  // Real Citi Bike docks, with their real names - 2,459 of them from the station
  // registry. Drawn above the labels so a dock is never hidden by a street name.
  map.addSource("stations", { type: "geojson", data: stationsFC });
  map.addLayer({
    id: "stations-dot", type: "circle", source: "stations",
    minzoom: 12.5,
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 12.5, 2.2, 15, 4.5, 17, 7],
      "circle-color": c.accent,
      "circle-opacity": 0.9,
      "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 12.5, 0.5, 15, 1.5],
      "circle-stroke-color": theme === "dark" ? "#06231a" : "#ffffff",
    },
  });
  map.addLayer({
    id: "stations-label", type: "symbol", source: "stations",
    minzoom: 14.6,
    layout: {
      "text-field": ["get", "name"],
      "text-font": ["Noto Sans Regular"],
      "text-size": 11,
      "text-offset": [0, 1.1],
      "text-anchor": "top",
      "text-max-width": 9,
      "text-allow-overlap": false,
      "text-optional": true,
    },
    paint: {
      "text-color": theme === "dark" ? "#a9f3d6" : "#0b5f47",
      "text-halo-color": theme === "dark" ? "#0e1215" : "#ffffff",
      "text-halo-width": 1.6,
    },
  });

  map.on("click", "stations-dot", (e) => {
    const f = e.features[0];
    dropPin(f.geometry.coordinates[1], f.geometry.coordinates[0], f.properties.name);
    loadStations(f.geometry.coordinates[1], f.geometry.coordinates[0]);
  });
  map.on("mouseenter", "stations-dot", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "stations-dot", () => (map.getCanvas().style.cursor = ""));

  map.on("click", "cells-fill", (e) => {
    const f = e.features[0];
    dropPin(e.lngLat.lat, e.lngLat.lng, `Cell ${f.properties.region_id.slice(0, 8)}…`);
    loadStations(e.lngLat.lat, e.lngLat.lng);
  });
  map.on("mouseenter", "cells-fill", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "cells-fill", () => (map.getCanvas().style.cursor = ""));
}

/** Switch basemap without losing what is drawn on top of it. */
function setBasemap(style) {
  theme = style.theme;
  document.documentElement.setAttribute("data-theme", style.theme);
  map.setStyle(style.url);
  map.once("styledata", () => {
    addOverlays();
    if (map.getSource("cells")) map.getSource("cells").setData(geometry);
    if (map.getSource("stations")) map.getSource("stations").setData(stationsFC);
    setGridVisible(gridVisible);
    if (routeData) drawRoute(routeData, false);
    if (blinkRegion) startBlink(blinkRegion);
  });
}

map.on("load", async () => {
  addOverlays();
  animateDash();
  await boot();
});

function emptyFC() { return { type: "FeatureCollection", features: [] }; }

/* ── map effects ─────────────────────────────────────────────────── */

/** Marching-ants along the route. Cycling dasharray is the only way to animate a
 *  line in MapLibre; the sequence is pre-baked because setting it every frame at
 *  fractional values causes visible stutter. */
const DASHES = [
  [0, 4, 3], [0.5, 4, 2.5], [1, 4, 2], [1.5, 4, 1.5],
  [2, 4, 1], [2.5, 4, 0.5], [3, 4, 0], [0, 0.5, 3, 3.5],
];
let dashStep = 0;
function animateDash() {
  setInterval(() => {
    if (!map.getLayer("route-dash")) return;
    dashStep = (dashStep + 1) % DASHES.length;
    map.setPaintProperty("route-dash", "line-dasharray", DASHES[dashStep]);
  }, 90);
}

/** Pulse the focused cell's fill opacity. Kept translucent on purpose so the
 *  streets underneath stay visible — a solid blink hides the thing you're
 *  looking for. */
let blinkFrame = null;
function startBlink(regionId) {
  stopBlink();
  blinkRegion = regionId;
  if (!regionId || !map.getLayer("cells-blink")) return;
  map.setFilter("cells-blink", ["==", ["get", "region_id"], regionId]);
  map.setFilter("cells-focus", ["==", ["get", "region_id"], regionId]);
  const started = performance.now();
  const tick = (now) => {
    const t = (now - started) / 1000;
    // sine keeps the pulse smooth at both ends rather than snapping
    const o = 0.16 + 0.34 * (0.5 + 0.5 * Math.sin(t * 2.2));
    map.setPaintProperty("cells-blink", "fill-opacity", o);
    map.setPaintProperty("cells-focus", "line-opacity", 0.55 + 0.45 * (o / 0.5));
    blinkFrame = requestAnimationFrame(tick);
  };
  blinkFrame = requestAnimationFrame(tick);
}
/** Fade the choropleth in or out. Hidden by default so the real map shows through;
 *  turned on when the user is actually looking at demand. */
let gridVisible = false;
function setGridVisible(on) {
  gridVisible = on;
  if (!map.getLayer("cells-fill")) return;
  map.setPaintProperty("cells-fill", "fill-opacity", on ? 0.5 : 0);
  map.setPaintProperty("cells-line", "line-opacity", on ? 0.4 : 0);
  const toggle = $("grid-toggle");
  if (toggle) toggle.setAttribute("aria-pressed", String(on));
  const legend = $("legend");
  if (legend) legend.classList.toggle("hidden", !on);
}

function stopBlink() {
  if (blinkFrame) cancelAnimationFrame(blinkFrame);
  blinkFrame = null;
}

function clearFocus() {
  stopBlink();
  blinkRegion = null;
  if (map.getLayer("cells-blink")) {
    map.setFilter("cells-blink", ["==", ["get", "region_id"], "__none__"]);
    map.setFilter("cells-focus", ["==", ["get", "region_id"], "__none__"]);
  }
}

/** Google-style approach: pull back, travel, then settle in.
 *  MapLibre's flyTo already arcs when the jump is long; over a short hop it barely
 *  moves, so the pull-back is forced to keep the motion legible. */
function flyToPlace(lat, lng, zoom = 15.2) {
  const from = map.getCenter();
  const km = haversineKm(from.lat, from.lng, lat, lng);
  const pullback = Math.max(10.2, Math.min(map.getZoom(), zoom) - (km > 3 ? 2.6 : 1.8));

  map.easeTo({ zoom: pullback, duration: 520, easing: (t) => t * (2 - t) });
  setTimeout(() => {
    map.flyTo({
      center: [lng, lat], zoom, curve: 1.5, speed: 0.75,
      essential: true, padding: { bottom: sheetHeight() },
    });
  }, 480);
}

function haversineKm(aLat, aLng, bLat, bLng) {
  const R = 6371, r = Math.PI / 180;
  const dLat = (bLat - aLat) * r, dLng = (bLng - aLng) * r;
  const h = Math.sin(dLat / 2) ** 2
    + Math.cos(aLat * r) * Math.cos(bLat * r) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

function sheetHeight() {
  return $("sheet").classList.contains("expanded") ? Math.min(innerHeight * 0.5, 380) : 190;
}

/** Draw the A* path and frame it. */
function drawRoute(route, frame = true) {
  if (!route || !route.ok || !map.getSource("route")) return;
  routeData = route;
  const coords = route.geometry.coordinates;
  map.getSource("route").setData({
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: {}, geometry: route.geometry }],
  });
  map.getSource("route-points").setData({
    type: "FeatureCollection",
    features: coords.map((c, i) => ({
      type: "Feature",
      properties: { kind: i === coords.length - 1 ? "target" : i === 0 ? "origin" : "hop" },
      geometry: { type: "Point", coordinates: c },
    })),
  });

  const lngs = coords.map((c) => c[0]);
  const lats = coords.map((c) => c[1]);
  if (!frame) return;
  const bounds = [[Math.min(...lngs), Math.min(...lats)],
    [Math.max(...lngs), Math.max(...lats)]];
  // pull back first so the whole path enters frame from a wider view
  map.easeTo({ zoom: Math.max(map.getZoom() - 1.6, 11), duration: 460 });
  setTimeout(() => map.fitBounds(bounds, {
    padding: { top: 110, left: 70, right: 70, bottom: sheetHeight() + 40 },
    maxZoom: 16.4, duration: 1250, curve: 1.5,
  }), 420);
}

function clearRoute() {
  routeData = null;
  if (!map.getSource("route")) return;
  map.getSource("route").setData(emptyFC());
  map.getSource("route-points").setData(emptyFC());
}

/* ── boot ────────────────────────────────────────────────────────── */
async function boot() {
  const cfg = await fetch("/api/config").then((r) => r.json());
  state.config = cfg;
  // the datetime-local field is naive and the backend reads a naive value as NYC
  // local time, so seed it from the local default - seeding from a UTC ISO string
  // would show a time five hours off and silently shift on first edit
  state.when = cfg.window.default_local;

  renderChips("model-chips", cfg.models.map((m) => ({ id: m.id, label: m.label })),
    () => state.model, (v) => { state.model = v; refresh(); });
  renderChips("grid-chips", cfg.grids.map((g) => ({ id: g.id, label: g.label })),
    () => state.grid, (v) => { state.grid = v; loadGeometry().then(refresh); });
  renderChips("horizon-chips", cfg.horizons.map((h) => ({ id: String(h.value), label: h.label })),
    () => String(state.horizon), (v) => { state.horizon = Number(v); refresh(); });

  const when = $("when-input");
  when.min = cfg.window.start_local.slice(0, 16);
  when.max = cfg.window.end_local.slice(0, 16);
  when.value = cfg.window.default_local.slice(0, 16);
  when.addEventListener("change", () => {
    if (!when.value) return;
    state.when = when.value;
    refresh();
  });

  renderBasemaps(cfg.basemaps || []);
  $("grid-toggle").addEventListener("click", () => setGridVisible(!gridVisible));
  setGridVisible(false);

  $("mode-text").textContent = cfg.assistant_mode;
  $("mode-badge").title = cfg.assistant_model
    ? `Assistant: ${cfg.assistant_mode} (${cfg.assistant_model})`
    : "Assistant: built-in router (no API key set)";
  showNotice(cfg.notice);

  await loadGeometry();
  await loadStationLayer();
  await refresh();
  addBot("Ask me where you are and I'll tell you how the bikes are moving. "
    + "I forecast **trip demand**, not live dock counts — there's no historical "
    + "dock data for 2023–24, so I report flow pressure instead of a bike count.");
}

function renderChips(containerId, items, get, set) {
  const box = $(containerId);
  box.innerHTML = "";
  items.forEach((item) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.textContent = item.label;
    b.setAttribute("aria-pressed", String(get() === item.id));
    b.addEventListener("click", () => {
      set(item.id);
      [...box.children].forEach((c) => c.setAttribute("aria-pressed", String(c === b)));
    });
    box.appendChild(b);
  });
}

function renderBasemaps(styles) {
  if (!styles.length) return;
  const list = $("mapstyle-list");
  // start on whichever style matches the OS preference, which is what the map
  // was constructed with
  let current = styles.find((s) => s.theme === theme) || styles[0];
  list.innerHTML = "";
  styles.forEach((style) => {
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = `<span class="ms-name"></span><span class="ms-detail"></span>`;
    b.querySelector(".ms-name").textContent = style.label;
    b.querySelector(".ms-detail").textContent = style.detail;
    b.setAttribute("aria-pressed", String(style.id === current.id));
    b.addEventListener("click", () => {
      current = style;
      setBasemap(style);
      [...list.children].forEach((c) => c.setAttribute("aria-pressed", String(c === b)));
      list.hidden = true;
      $("mapstyle-toggle").setAttribute("aria-expanded", "false");
    });
    list.appendChild(b);
  });
  $("mapstyle-toggle").addEventListener("click", (e) => {
    e.stopPropagation();
    const open = list.hidden;
    list.hidden = !open;
    $("mapstyle-toggle").setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".mapstyle")) {
      list.hidden = true;
      $("mapstyle-toggle").setAttribute("aria-expanded", "false");
    }
  });
}

function showNotice(text) {
  const el = $("notice");
  el.textContent = text || "";
  el.classList.toggle("show", Boolean(text));
}

/* ── data ────────────────────────────────────────────────────────── */
async function loadStationLayer() {
  const data = await fetch(`/api/stations?grid=${state.grid}`)
    .then((r) => r.json()).catch(() => null);
  if (!data || !data.stations) return;
  stationsFC = {
    type: "FeatureCollection",
    features: data.stations.map((st) => ({
      type: "Feature",
      properties: { name: st.station_name, region_id: st.region_id },
      geometry: { type: "Point", coordinates: [st.lng, st.lat] },
    })),
  };
  if (map.getSource("stations")) map.getSource("stations").setData(stationsFC);
}

async function loadGeometry() {
  geometry = await fetch(`/api/geojson?grid=${state.grid}`).then((r) => r.json());
}

async function refresh() {
  const url = `/api/overview?grid=${state.grid}&model=${state.model}`
    + `&horizon=${state.horizon}&when=${encodeURIComponent(state.when || "")}`;
  const data = await fetch(url).then((r) => r.json());
  if (!data.ok) { showNotice(data.error || "Could not load the forecast."); return; }
  showNotice(state.config?.notice);

  const byRegion = new Map(data.regions.map((r) => [r.region_id, r]));
  const max = Math.max(...data.regions.map((r) => r.cumulative_pickups), 1);
  state.maxPickups = max;
  $("legend-max").textContent = Math.round(max);

  // log scale: a handful of Midtown cells carry most of the demand, so a linear
  // ramp would render the rest of the city as one flat colour
  geometry.features.forEach((f) => {
    const row = byRegion.get(f.properties.region_id);
    const v = row ? row.cumulative_pickups : 0;
    f.properties.pickups = v > 0 ? Math.log1p(v) / Math.log1p(max) * 4 : 0;
    f.properties.raw = v;
  });
  if (map.getSource("cells")) map.getSource("cells").setData(geometry);

  if (!state.place) {
    $("sheet-sub").textContent =
      `${data.model_label} · ${Math.round(data.total_predicted_pickups).toLocaleString()} `
      + `pickups citywide in the next ${data.window_minutes} min · ${data.time.local_pretty}`;
  }
  if (state.place) await loadStations(state.place.lat, state.place.lng, false);
}

async function loadStations(lat, lng, moveMap = true) {
  const url = `/api/station_forecast?lat=${lat}&lng=${lng}&grid=${state.grid}`
    + `&model=${state.model}&horizon=${state.horizon}`
    + `&when=${encodeURIComponent(state.when || "")}&k=4`;
  const data = await fetch(url).then((r) => r.json());
  if (!data.ok) { showNotice(data.error); return; }
  renderStationCards(data);
  // glow the cell you asked about; the full heatmap stays off unless requested,
  // so the streets, shops and buildings underneath remain visible
  if (data.stations.length) startBlink(data.stations[0].station.region_id);
  if (moveMap) flyToPlace(lat, lng, 15.2);
  expand(true);
  loadRoute(lat, lng, moveMap);
}

async function loadRoute(lat, lng, frame = true) {
  const url = `/api/route?lat=${lat}&lng=${lng}&grid=${state.grid}`
    + `&model=${state.model}&horizon=${state.horizon}`
    + `&when=${encodeURIComponent(state.when || "")}`;
  const route = await fetch(url).then((r) => r.json()).catch(() => null);
  if (!route || !route.ok) { clearRoute(); return; }
  state.route = route;
  renderRouteCard(route);
  startBlink(route.target.station.region_id);
  // the pin already flew to the origin; framing the path too would fight it
  if (frame) setTimeout(() => drawRoute(route), 1100);
  else drawRoute(route);
}

/* ── rendering ───────────────────────────────────────────────────── */
function renderStationCards(data) {
  const box = $("cards");
  box.innerHTML = "";
  $("sheet-sub").textContent =
    `${data.model_label} · next ${data.window_minutes} min · ${data.time.local_pretty}`;

  data.stations.forEach((s) => {
    const st = s.station;
    const v = s.availability;
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-top">
        <span class="card-name"></span>
        <span class="card-dist">${distance(st.distance_km)}</span>
      </div>
      <div class="flow">
        <div class="out"><div class="n">${s.predicted_pickups.toFixed(0)}</div>
          <div class="k">picked up</div></div>
        <div class="in"><div class="n">${s.predicted_dropoffs.toFixed(0)}</div>
          <div class="k">returned</div></div>
      </div>
      <span class="pill ${v.verdict}"></span>
      <p class="card-note"></p>
      <div class="card-approx"></div>`;
    card.querySelector(".card-name").textContent = st.station_name;
    card.querySelector(".pill").textContent = v.label;
    card.querySelector(".card-note").textContent = v.detail;
    card.querySelector(".card-approx").textContent =
      `Derived: ${(st.pickup_share * 100).toFixed(1)}% share of its cell's `
      + `${s.region_pickups.toFixed(0)} predicted pickups · ${v.measures}`;
    card.addEventListener("click", () => dropPin(st.lat, st.lng, st.station_name));
    box.appendChild(card);
  });
}

function renderRouteCard(route) {
  const t = route.target;
  const st = t.station;
  const card = document.createElement("div");
  card.className = `card route-card${route.walkable ? "" : " unreachable"}`;
  const smarter = !route.walkable
    ? `<div class="route-smart warn">${escapeHtml(route.reachability)}</div>`
    : route.chose_nearest ? ""
      : `<div class="route-smart">Skipped a closer dock — that one is draining.</div>`;
  card.innerHTML = `
    <div class="route-head">
      <div class="route-dest">
        <div class="route-eyebrow">Walk to</div>
        <div class="card-name"></div>
      </div>
      <div class="route-eta"><div class="n">${route.walkable
        ? route.walk_minutes : (route.walk_metres / 1000).toFixed(1)}</div>
        <div class="k">${route.walkable ? "min" : "km"}</div></div>
    </div>
    ${smarter}
    <div class="route-meta">
      <span>${route.walk_metres} m</span><span>·</span>
      <span>${t.availability.label}</span>
      ${route.graph_hops ? `<span>·</span><span>${route.graph_hops} hops</span>` : ""}
    </div>
    <div class="route-steps"></div>
    <div class="card-approx"></div>`;
  card.querySelector(".card-name").textContent = st.station_name;
  card.querySelector(".route-steps").innerHTML = route.legs
    .map((l) => `<div class="step"><span class="dotline"></span>
        <span class="step-txt"></span><span class="step-m">${l.metres} m</span></div>`)
    .join("");
  [...card.querySelectorAll(".step-txt")].forEach((el, i) => {
    el.textContent = route.legs[i].to;
  });
  card.querySelector(".card-approx").textContent = route.caveat;
  $("cards").prepend(card);
}

function renderComparison(data) {
  const box = $("cards");
  const max = Math.max(
    ...data.models.map((m) => m.region_pickups),
    data.actual_region_pickups || 0, 1);
  const rows = data.models
    .map((m) => barRow(m.label, m.region_pickups, max, false))
    .join("");
  const actual = data.actual_region_pickups != null
    ? barRow("Actual", data.actual_region_pickups, max, true) : "";
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `
    <div class="card-top"><span class="card-name">Model comparison</span></div>
    <p class="card-note" style="margin:6px 0 10px">
      Predicted pickups in this cell for the ${data.bin_minutes}-minute bin
      ${data.horizon * data.bin_minutes} min ahead.</p>
    <div class="bars">${rows}${actual}</div>
    <div class="card-approx">Cell-level figures, directly comparable to the actual.
      On the TEST set these rank LightGBM &gt; ST-GNN &gt; TFT on MAE.</div>`;
  box.prepend(card);
}

function barRow(name, value, max, isActual) {
  const pct = Math.max(2, (value / max) * 100);
  return `<div class="bar-row">
      <span class="bar-name">${escapeHtml(name)}</span>
      <span class="bar-track"><span class="bar-fill${isActual ? " actual" : ""}"
        style="width:${pct}%"></span></span>
      <span class="bar-val">${value.toFixed(0)}</span>
    </div>`;
}

function distance(km) {
  if (km < 0.02) return "here";
  return km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`;
}

function dropPin(lat, lng, label) {
  state.place = { lat, lng, name: label };
  $("sheet-place").textContent = label;
  if (marker) marker.remove();
  const el = document.createElement("div");
  el.style.cssText = "width:16px;height:16px;border-radius:50%;background:#0b7a5a;"
    + "border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.35)";
  marker = new maplibregl.Marker({ element: el }).setLngLat([lng, lat]).addTo(map);
}

/* ── search ──────────────────────────────────────────────────────── */
const input = $("place-input");
const suggestions = $("suggestions");
let searchTimer = null;

input.addEventListener("input", () => {
  $("place-clear").hidden = !input.value;
  clearTimeout(searchTimer);
  const q = input.value.trim();
  if (q.length < 2) { suggestions.hidden = true; return; }
  searchTimer = setTimeout(async () => {
    const res = await fetch(`/api/place?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    if (!data.ok) { suggestions.hidden = true; return; }
    suggestions.innerHTML = "";
    data.matches.forEach((m) => {
      const b = document.createElement("button");
      b.type = "button";
      b.innerHTML = `<span class="s-name"></span><span class="s-kind">${m.source}</span>`;
      b.querySelector(".s-name").textContent = m.name;
      b.addEventListener("click", () => {
        input.value = m.name;
        suggestions.hidden = true;
        dropPin(m.lat, m.lng, m.name);
        loadStations(m.lat, m.lng);
      });
      suggestions.appendChild(b);
    });
    suggestions.hidden = false;
  }, 180);
});

$("place-clear").addEventListener("click", () => {
  input.value = ""; suggestions.hidden = true; $("place-clear").hidden = true;
  state.place = null;
  $("sheet-place").textContent = "NYC overview";
  $("cards").innerHTML = "";
  clearFocus();
  clearRoute();
  if (marker) { marker.remove(); marker = null; }
  refresh();
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".topbar")) suggestions.hidden = true;
});

/* ── sheet ───────────────────────────────────────────────────────── */
function expand(on) {
  $("sheet").classList.toggle("expanded", on);
  document.body.classList.toggle("expanded", on);
}
$("grabber").addEventListener("click", () =>
  expand(!$("sheet").classList.contains("expanded")));

/* ── chat ────────────────────────────────────────────────────────── */
function addMsg(text, cls, { raw = false } = {}) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  // `raw` is only ever used for markup this file authored (the typing dots);
  // anything from the server or the user goes through escaping
  el.innerHTML = raw ? text : cls === "user" ? escapeHtml(text) : markdown(text);
  $("chat").appendChild(el);
  $("sheet-body").scrollTop = $("sheet-body").scrollHeight;
  return el;
}
const addBot = (t) => addMsg(t, "bot");

function markdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("chat-input").value.trim();
  if (!text || state.busy) return;
  $("chat-input").value = "";
  send(text);
});

$("quick").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b && !state.busy) send(b.dataset.q);
});

async function send(text) {
  state.busy = true;
  $("send").disabled = true;
  expand(true);
  addMsg(text, "user");
  const thinking = addMsg('<span class="typing"><i></i><i></i><i></i></span>',
                          "bot", { raw: true });

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text, grid: state.grid, model: state.model,
        horizon: state.horizon, when: state.when,
        history: state.history, last_place: state.place,
      }),
    });
    const data = await res.json();
    thinking.remove();
    addBot(data.text || "No answer.");
    if (data.degraded) addMsg(data.degraded, "bot warn");
    state.history = data.history || [];

    if (data.place && data.place.lat) {
      dropPin(data.place.lat, data.place.lng, data.place.name || "Selected place");
      map.easeTo({ center: [data.place.lng, data.place.lat],
        zoom: Math.max(map.getZoom(), 13.4) });
    }
    // render cards from the same payloads the assistant saw
    (data.data || []).forEach((payload) => {
      if (!payload || !payload.ok) return;
      if (payload.stations) renderStationCards(payload);
      if (payload.models) renderComparison(payload);
      if (payload.geometry && payload.target) {
        drawRoute(payload);
        renderRouteCard(payload);
        startBlink(payload.target.station.region_id);
      }
    });
  } catch (err) {
    thinking.remove();
    addMsg(`Something went wrong: ${err.message}`, "bot warn");
  } finally {
    state.busy = false;
    $("send").disabled = false;
  }
}
