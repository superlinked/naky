import {createFrameCallbackLifecycle, indexStateBlocks, parseEventStream, selectTimeGroup, stateAt} from "./replay.mjs";

const DURATION_MS = 13299;
const WIDTH = 304;
const HEIGHT = 640;
const EXPECTED = Object.freeze({
  schema: "naky.public-demo.v1",
  revision: "7901fa3f0b9e36afa87d24ff9ecd48037c7fd6a0",
  mediaSha256: "9df9a40972ff72ae9dc407a44e02a2bda3606ded63117157f89af44d00cf0a44",
  eventsSha256: "deced573621b8b530d66eaf1d00acd8faaf23b898c3da94c7239510744b8a1a7",
  stateSha256: "5ad0419dd4ebe355ab72ad153c0a509115715c82b07422d42db74c3788e1478a",
});
const NS = "http://www.w3.org/2000/svg";

const elements = Object.freeze({
  comparison: document.querySelector(".comparison"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
  modeExplanation: document.querySelector("#mode-explanation"),
  overlayStatus: document.querySelector("#overlay-status"),
  video: document.querySelector("#demo-video"),
  overlay: document.querySelector("#overlay"),
  moment: document.querySelector("#moment-label"),
  play: document.querySelector("#play-button"),
  playLabel: document.querySelector("#play-label"),
  timeline: document.querySelector("#timeline"),
  time: document.querySelector("#time-output"),
  markerRail: document.querySelector("#marker-rail"),
  videoError: document.querySelector("#video-error"),
  stateViewport: document.querySelector("#state-viewport"),
  stateStatus: document.querySelector("#state-status"),
  follow: document.querySelector("#follow-button"),
});

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let events = [];
let stateBlocks = [];
let structuralSamples = [];
let blockNodes = new Map();
let selectedBlock = null;
let followPlayhead = true;
let programmaticScrollUntil = 0;
const frameCallbacks = createFrameCallbackLifecycle();

function manifestIsValid(value) {
  return value?.schema_version === EXPECTED.schema &&
    value.recording?.revision === EXPECTED.revision &&
    value.recording?.member === "IOS/330.mp4" &&
    value.presentation?.width === WIDTH && value.presentation?.height === HEIGHT &&
    value.presentation?.duration_ms === DURATION_MS &&
    value.presentation?.media?.path === "ios-330.mp4" &&
    value.presentation?.media?.sha256 === EXPECTED.mediaSha256 &&
    value.output?.events?.path === "ios-330.events.ndjson" &&
    value.output?.events?.sha256 === EXPECTED.eventsSha256 &&
    value.output?.state?.path === "ios-330.screen.txt" &&
    value.output?.state?.sha256 === EXPECTED.stateSha256 &&
    value.output?.event_count === 243 &&
    Array.isArray(value.output?.structural_sample_times_ms);
}

async function fetchAsset(path, type = "text") {
  const response = await fetch(path, {credentials: "same-origin"});
  if (!response.ok) throw new Error(`could not load ${path}`);
  return type === "json" ? response.json() : response.text();
}

function formatTime(milliseconds) {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function svg(name, attributes = {}) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  return node;
}

function operationName(kind) {
  return {appeared: "add", content: "change", moved: "move", disappeared: "remove"}[kind] ?? kind;
}

function drawLabel(change, index) {
  const box = change.box;
  const text = `${operationName(change.kind)} · ${change.elementId.replace(/^e0*/, "e")}`;
  const width = Math.min(84, Math.max(31, text.length * 4.5 + 8));
  const x = Math.max(1, Math.min(WIDTH - width - 1, box.x));
  const y = Math.max(1, box.y - 11 - index * 2);
  const group = svg("g", {class: "overlay-label"});
  group.append(svg("rect", {x, y, width, height: 10, rx: 2}));
  const label = svg("text", {x: x + 4, y: y + 7.2});
  label.textContent = text;
  group.append(label);
  return group;
}

function renderOverlay(snapshot) {
  const changed = new Map(snapshot.changes.map((item) => [item.elementId, item]));
  const fragment = document.createDocumentFragment();
  for (const item of snapshot.elements) {
    const isChanged = changed.has(item.id);
    fragment.append(svg("rect", {
      class: `element-box${isChanged ? " is-changed" : ""}`,
      x: item.box.x, y: item.box.y, width: item.box.width, height: item.box.height,
    }));
  }
  for (const activity of snapshot.activities) {
    fragment.append(svg("rect", {
      class: "activity-box",
      x: activity.box.x, y: activity.box.y, width: activity.box.width, height: activity.box.height,
    }));
  }
  snapshot.changes.slice(-4).forEach((change, index) => fragment.append(drawLabel(change, index)));
  elements.overlay.replaceChildren(fragment);
}

function setCurrentBlock(block) {
  for (const [candidate, node] of blockNodes) {
    node.classList.toggle("is-current", candidate === block);
    node.classList.toggle("is-past", candidate.frametime < (block?.frametime ?? -1));
  }
  if (block === selectedBlock) return;
  selectedBlock = block;
  if (block && followPlayhead) {
    programmaticScrollUntil = performance.now() + (reduceMotion ? 80 : 700);
    blockNodes.get(block)?.scrollIntoView({block: "center", behavior: reduceMotion ? "auto" : "smooth"});
  }
}

function renderAt(timeMs) {
  const bounded = Math.max(0, Math.min(DURATION_MS, Math.round(timeMs)));
  elements.timeline.value = String(bounded);
  elements.time.value = `${formatTime(bounded)} / ${formatTime(DURATION_MS)}`;
  if (!events.length) return;

  const snapshot = stateAt(events, bounded, structuralSamples);
  renderOverlay(snapshot);
  const current = selectTimeGroup(stateBlocks, bounded);
  setCurrentBlock(current);
  const changeCount = snapshot.changes.length;
  const activityCount = snapshot.activities.length;
  const details = [];
  if (changeCount) details.push(`${changeCount} recent ${changeCount === 1 ? "change" : "changes"}`);
  if (activityCount) details.push(`${activityCount} active ${activityCount === 1 ? "region" : "regions"}`);
  elements.moment.textContent = `${snapshot.elements.length} retained text regions${details.length ? ` · ${details.join(" · ")}` : ""}`;
  elements.stateStatus.textContent = current ? `State at ${current.frametime.toLocaleString()} ms` : "Before the first observation";
}

function currentMediaMs() {
  return Number.isFinite(elements.video.currentTime) ? Math.round(elements.video.currentTime * 1000) : Number(elements.timeline.value);
}

function requestFrameLoop() {
  if (elements.video.paused) return;
  if ("requestVideoFrameCallback" in elements.video) {
    frameCallbacks.request(
      (callback) => elements.video.requestVideoFrameCallback(callback),
      (identifier) => elements.video.cancelVideoFrameCallback?.(identifier),
      (_now, metadata) => {
        renderAt(Math.round(metadata.mediaTime * 1000));
        requestFrameLoop();
      },
    );
  } else {
    frameCallbacks.request(
      (callback) => requestAnimationFrame(callback),
      (identifier) => cancelAnimationFrame(identifier),
      () => {
        renderAt(currentMediaMs());
        requestFrameLoop();
      },
    );
  }
}

function invalidateFrameLoop() {
  frameCallbacks.invalidate();
}

function updatePlayState() {
  const playing = !elements.video.paused && !elements.video.ended;
  elements.play.classList.toggle("is-playing", playing);
  elements.play.setAttribute("aria-label", playing ? "Pause recording" : "Play recording");
  elements.playLabel.textContent = playing ? "Pause" : "Play";
  if (playing) requestFrameLoop();
}

function renderStateBlocks() {
  const fragment = document.createDocumentFragment();
  blockNodes = new Map();
  stateBlocks.forEach((block) => {
    const node = document.createElement("pre");
    node.className = "state-block";
    node.dataset.time = String(block.frametime);
    node.textContent = block.text;
    fragment.append(node);
    blockNodes.set(block, node);
  });
  elements.stateViewport.replaceChildren(fragment);
}

function renderMarkers() {
  const fragment = document.createDocumentFragment();
  structuralSamples.forEach((timeMs) => {
    const marker = document.createElement("i");
    marker.style.left = `${(timeMs / DURATION_MS) * 100}%`;
    fragment.append(marker);
  });
  elements.markerRail.replaceChildren(fragment);
}

function setStateViewportAvailability(available) {
  if (!available) {
    const active = document.activeElement;
    if (active === elements.stateViewport || elements.stateViewport.contains(active)) {
      elements.modeButtons.find((button) => button.dataset.mode === "pixels")?.focus();
    }
    elements.stateViewport.setAttribute("aria-hidden", "true");
    elements.stateViewport.tabIndex = -1;
    return;
  }
  elements.stateViewport.removeAttribute("aria-hidden");
  elements.stateViewport.tabIndex = 0;
}

function setMode(mode) {
  if (mode !== "state" && mode !== "pixels") return;
  elements.comparison.dataset.mode = mode;
  elements.modeButtons.forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (mode === "state") {
    setStateViewportAvailability(true);
    elements.modeExplanation.textContent = "A text model receives the exact state stream on the right. The recording remains visible here only as a reference.";
    elements.overlayStatus.textContent = "Näky overlay";
  } else {
    setStateViewportAvailability(false);
    elements.modeExplanation.textContent = "In the released comparison, a multimodal reader receives 20 sampled pixel frames. The state stream is not sent.";
    elements.overlayStatus.textContent = "Pixels only";
  }
  renderAt(currentMediaMs());
}

function bindInteractions() {
  elements.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  elements.play.addEventListener("click", async () => {
    if (!elements.video.paused) {
      invalidateFrameLoop();
      elements.video.pause();
      return;
    }
    if (elements.video.ended || elements.video.currentTime * 1000 >= DURATION_MS - 20) elements.video.currentTime = 0;
    try { await elements.video.play(); } catch { updatePlayState(); }
  });
  elements.timeline.addEventListener("input", () => {
    const timeMs = Number(elements.timeline.value);
    invalidateFrameLoop();
    if (Number.isFinite(elements.video.duration)) elements.video.currentTime = timeMs / 1000;
    renderAt(timeMs);
    requestFrameLoop();
  });
  elements.video.addEventListener("play", updatePlayState);
  elements.video.addEventListener("pause", () => { invalidateFrameLoop(); updatePlayState(); renderAt(currentMediaMs()); });
  elements.video.addEventListener("seeking", invalidateFrameLoop);
  elements.video.addEventListener("seeked", () => { renderAt(currentMediaMs()); requestFrameLoop(); });
  elements.video.addEventListener("timeupdate", () => renderAt(currentMediaMs()));
  elements.video.addEventListener("ended", () => { invalidateFrameLoop(); elements.video.currentTime = 0; renderAt(0); updatePlayState(); });
  elements.video.addEventListener("error", () => { invalidateFrameLoop(); elements.videoError.hidden = false; updatePlayState(); });
  elements.stateViewport.addEventListener("scroll", () => {
    if (!followPlayhead || performance.now() < programmaticScrollUntil) return;
    followPlayhead = false;
    elements.follow.hidden = false;
    elements.stateStatus.textContent = "State follow paused";
  }, {passive: true});
  elements.follow.addEventListener("click", () => {
    followPlayhead = true;
    elements.follow.hidden = true;
    const current = selectedBlock;
    selectedBlock = null;
    setCurrentBlock(current);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && !elements.video.paused) {
      invalidateFrameLoop();
      elements.video.pause();
    }
  });
}

async function initialize() {
  elements.play.disabled = true;
  bindInteractions();
  try {
    const manifest = await fetchAsset("demo/ios-330.manifest.json", "json");
    if (!manifestIsValid(manifest)) throw new Error("demo manifest identity differs");
    const [eventText, stateText] = await Promise.all([
      fetchAsset("demo/ios-330.events.ndjson"),
      fetchAsset("demo/ios-330.screen.txt"),
    ]);
    events = parseEventStream(eventText);
    stateBlocks = indexStateBlocks(stateText);
    structuralSamples = [...manifest.output.structural_sample_times_ms];
    if (events.length !== manifest.output.event_count) throw new Error("event count differs from manifest");
    renderStateBlocks();
    renderMarkers();
    renderAt(0);
    elements.play.disabled = false;
    elements.video.loop = true;
    if (!reduceMotion) {
      try { await elements.video.play(); } catch { updatePlayState(); }
    }
  } catch (error) {
    elements.moment.textContent = "Demo output could not be loaded";
    elements.stateStatus.textContent = error instanceof Error ? error.message : "Demo output unavailable";
    elements.videoError.hidden = false;
  }
}

initialize();
