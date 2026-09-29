import {
  TYPES,
  createFrameCallbackLifecycle,
  indexStateBlocks,
  overlayPrimitives,
  parseEventStream,
  selectTimeGroup,
  stateAt,
} from "./replay.mjs";

const CATALOG_SCHEMA = "naky.public-demo-gallery.v1";
const MANIFEST_SCHEMA = "naky.public-demo.v1";
const RELEASE = "v0.1.0";
const RELEASE_BINARY_SHA256 = "c791889fa5bc3f85a8398dddc0901ff3cf4f9bb84777fe48bf7e1f3c58c33b4a";
const NS = "http://www.w3.org/2000/svg";
const SHA256 = /^[0-9a-f]{64}$/;
const SLUG = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const EVENT_FAMILY_NAMES = new Map([
  [TYPES.SNAPSHOT, "snapshot"],
  [TYPES.APPEARED, "element_appeared"],
  [TYPES.MOVED, "element_moved"],
  [TYPES.CONTENT, "element_content_changed"],
  [TYPES.DISAPPEARED, "element_disappeared"],
  [TYPES.RESIZED, "screen_resized"],
  [TYPES.ACTIVITY, "activity_observed"],
]);

const elements = Object.freeze({
  demo: document.querySelector(".demo-shell"),
  demoKicker: document.querySelector("#demo-kicker"),
  recordingSelect: document.querySelector("#recording-select"),
  recordingSummary: document.querySelector("#recording-summary"),
  comparison: document.querySelector(".comparison"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
  modeExplanation: document.querySelector("#mode-explanation"),
  overlayStatus: document.querySelector("#overlay-status"),
  stage: document.querySelector("#phone-stage"),
  video: document.querySelector("#demo-video"),
  videoFallbackDownload: document.querySelector("#video-fallback-download"),
  videoDownload: document.querySelector("#video-download"),
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
  rawState: document.querySelector("#raw-state-link"),
  rawEvents: document.querySelector("#raw-events-link"),
  follow: document.querySelector("#follow-button"),
  sourceSummary: document.querySelector("#source-summary"),
});

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const frameCallbacks = createFrameCallbackLifecycle();
let catalog = [];
let catalogBySlug = new Map();
let events = [];
let stateBlocks = [];
let structuralSamples = [];
let blockNodes = new Map();
let selectedBlock = null;
let followPlayhead = true;
let programmaticScrollUntil = 0;
let durationMs = 0;
let presentationFrame = null;
let selectionGeneration = 0;
let fetchController = null;

function exactKeys(value, expected) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const actual = Object.keys(value);
  return actual.length === expected.length && actual.every((key) => expected.includes(key));
}

function nonemptyString(value, maximum = 500) {
  return typeof value === "string" && value.length > 0 && value.length <= maximum;
}

function positiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function unsignedInteger(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function httpsUrl(value) {
  if (!nonemptyString(value, 2000)) return false;
  try { return new URL(value).protocol === "https:"; } catch { return false; }
}

function identityIsValid(value) {
  if (!exactKeys(value, ["bytes", "sha256"])) return false;
  if (!positiveInteger(value.bytes) || !SHA256.test(value.sha256)) return false;
  return true;
}

function fileIdentityIsValid(value, expectedPath) {
  return exactKeys(value, ["path", "bytes", "sha256"]) && value.path === expectedPath &&
    positiveInteger(value.bytes) && SHA256.test(value.sha256);
}

function recordingIsValid(recording) {
  const common = [
    "dataset", "dataset_url", "revision", "split", "member", "license",
    "license_url", "source_kind", "canonical_av1_input",
  ];
  if (!recording || !nonemptyString(recording.dataset, 200) || !httpsUrl(recording.dataset_url) ||
      !nonemptyString(recording.revision, 500) || !nonemptyString(recording.split, 200) ||
      !nonemptyString(recording.member, 1000) || !nonemptyString(recording.license, 200) ||
      !httpsUrl(recording.license_url) || !identityIsValid(recording.canonical_av1_input)) return false;
  if (recording.source_kind === "continuous_recording") {
    return exactKeys(recording, [...common, "original"]) && identityIsValid(recording.original);
  }
  if (recording.source_kind !== "screenshot_sequence" ||
      !exactKeys(recording, [...common, "source_sequence", "ffv1_bridge"]) ||
      !identityIsValid(recording.ffv1_bridge)) return false;
  const sequence = recording.source_sequence;
  return exactKeys(sequence, ["frames", "width", "height", "pixel_format", "packing", "bytes", "sha256"]) &&
    positiveInteger(sequence.frames) && positiveInteger(sequence.width) && positiveInteger(sequence.height) &&
    sequence.pixel_format === "rgb24" && nonemptyString(sequence.packing, 500) &&
    positiveInteger(sequence.bytes) && SHA256.test(sequence.sha256);
}

function catalogIsValid(value) {
  if (!exactKeys(value, ["schema_version", "recordings"]) || value.schema_version !== CATALOG_SCHEMA ||
      !Array.isArray(value.recordings) || value.recordings.length === 0) return false;
  const slugs = new Set();
  return value.recordings.every((entry) => {
    if (!exactKeys(entry, ["slug", "title", "subtitle", "manifest"]) ||
        !SLUG.test(entry.slug) || slugs.has(entry.slug) ||
        !nonemptyString(entry.title, 100) || !nonemptyString(entry.subtitle, 140) ||
        entry.manifest !== `${entry.slug}.manifest.json`) return false;
    slugs.add(entry.slug);
    return true;
  });
}

function manifestIsValid(value, entry) {
  if (!value || typeof value !== "object" || value.schema_version !== MANIFEST_SCHEMA) return false;
  const recording = value.recording;
  const presentation = value.presentation;
  const output = value.output;
  if (!recordingIsValid(recording) || !presentation || !output) return false;
  if (!exactKeys(presentation, ["width", "height", "duration_ms", "media", "note"]) ||
      !positiveInteger(presentation.width) || !positiveInteger(presentation.height) ||
      !positiveInteger(presentation.duration_ms) || !nonemptyString(presentation.note, 2000) ||
      !fileIdentityIsValid(presentation.media, `${entry.slug}.mp4`)) return false;
  if (!exactKeys(output, [
    "release", "release_binary_sha256", "stream_id", "events", "state", "event_count",
    "event_family_counts", "structural_sample_times_ms", "first_event_ms", "last_event_ms",
    "generation_note",
  ]) || output.release !== RELEASE || output.release_binary_sha256 !== RELEASE_BINARY_SHA256 ||
      !nonemptyString(output.stream_id, 300) ||
      !fileIdentityIsValid(output.events, `${entry.slug}.events.ndjson`) ||
      !fileIdentityIsValid(output.state, `${entry.slug}.screen.txt`) ||
      !positiveInteger(output.event_count) || !Array.isArray(output.structural_sample_times_ms) ||
      !output.event_family_counts || typeof output.event_family_counts !== "object" ||
      Array.isArray(output.event_family_counts) || !unsignedInteger(output.first_event_ms) ||
      !unsignedInteger(output.last_event_ms) || output.first_event_ms > output.last_event_ms ||
      output.last_event_ms > presentation.duration_ms || !nonemptyString(output.generation_note, 2000)) return false;
  let previous = -1;
  for (const timeMs of output.structural_sample_times_ms) {
    if (!unsignedInteger(timeMs) || timeMs <= previous || timeMs > presentation.duration_ms) return false;
    previous = timeMs;
  }
  return Object.values(output.event_family_counts).every(unsignedInteger);
}

async function fetchAsset(path, type = "text", signal = undefined) {
  const response = await fetch(path, {credentials: "same-origin", signal});
  if (!response.ok) throw new Error(`could not load ${path}`);
  return type === "json" ? response.json() : response.text();
}

function demoPath(name) {
  return `demo/${name}`;
}

function formatTime(milliseconds) {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function durationLabel(milliseconds) {
  const seconds = Math.max(1, Math.round(milliseconds / 1000));
  return seconds < 60 ? `${seconds} seconds` : formatTime(milliseconds);
}

function svg(name, attributes = {}) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  return node;
}

function appendBox(parent, className, box) {
  parent.append(svg("rect", {
    class: className,
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
  }));
}

function boxCenter(box) {
  return {x: box.x + box.width / 2, y: box.y + box.height / 2};
}

function appendArrow(parent, className, from, to, marker) {
  parent.append(svg("line", {
    class: className,
    x1: from.x,
    y1: from.y,
    x2: to.x,
    y2: to.y,
    "marker-end": `url(#${marker})`,
  }));
}

function arrowMarker(id, className) {
  const marker = svg("marker", {
    id,
    viewBox: "0 0 8 8",
    refX: 7,
    refY: 4,
    markerWidth: 5,
    markerHeight: 5,
    orient: "auto-start-reverse",
  });
  marker.append(svg("path", {class: className, d: "M 0 0 L 8 4 L 0 8 z"}));
  return marker;
}

function renderOverlay(snapshot) {
  const projection = overlayPrimitives(snapshot);
  elements.overlay.setAttribute("viewBox", `0 0 ${projection.frame.width} ${projection.frame.height}`);
  const fragment = document.createDocumentFragment();
  const definitions = svg("defs");
  definitions.append(arrowMarker("change-arrow", "change-arrow-head"));
  definitions.append(arrowMarker("activity-arrow", "activity-arrow-head"));
  fragment.append(definitions);

  const retainedLayer = svg("g", {class: "retained-layer"});
  projection.retained.forEach((item) => appendBox(retainedLayer, "element-box", item.box));
  fragment.append(retainedLayer);

  const changeLayer = svg("g", {class: "change-layer"});
  projection.changes.forEach((change) => {
    appendBox(changeLayer, `change-box is-${change.kind}`, change.box);
    if (change.kind === "moved") {
      appendBox(changeLayer, "move-origin", change.from);
      appendArrow(changeLayer, "change-path", boxCenter(change.from), boxCenter(change.to), "change-arrow");
    }
  });
  fragment.append(changeLayer);

  const activityLayer = svg("g", {class: "activity-layer"});
  projection.activities.forEach((activity) => {
    appendBox(activityLayer, `activity-box is-${activity.kind}`, activity.box);
    if (activity.kind === "region_translated") {
      const start = boxCenter(activity.box);
      appendArrow(activityLayer, "activity-path", start, {x: start.x + activity.dx, y: start.y + activity.dy}, "activity-arrow");
    }
  });
  fragment.append(activityLayer);
  elements.overlay.replaceChildren(fragment);
  return projection;
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

function plural(count, singular, pluralForm = `${singular}s`) {
  return `${count} ${count === 1 ? singular : pluralForm}`;
}

function renderAt(timeMs) {
  const bounded = Math.max(0, Math.min(durationMs, Math.round(timeMs)));
  elements.timeline.value = String(bounded);
  elements.time.value = `${formatTime(bounded)} / ${formatTime(durationMs)}`;
  if (!events.length) return;

  const replaySnapshot = stateAt(events, bounded, structuralSamples);
  const snapshot = replaySnapshot.frame ? replaySnapshot : {...replaySnapshot, frame: presentationFrame};
  const projection = renderOverlay(snapshot);
  const current = selectTimeGroup(stateBlocks, bounded);
  setCurrentBlock(current);
  const details = [plural(projection.retained.length, "retained region")];
  if (projection.changes.length) details.push(plural(projection.changes.length, "recent change"));
  if (projection.activities.length) details.push(plural(projection.activities.length, "recent activity region"));
  elements.moment.textContent = details.join(" · ");
  elements.stateStatus.textContent = current ? `State at ${current.frametime.toLocaleString()} ms` : "Before the first observation";
}

function currentMediaMs() {
  return Number.isFinite(elements.video.currentTime) ? Math.round(elements.video.currentTime * 1000) : Number(elements.timeline.value);
}

function requestFrameLoop() {
  if (elements.video.paused || !events.length) return;
  const generation = selectionGeneration;
  if ("requestVideoFrameCallback" in elements.video) {
    frameCallbacks.request(
      (callback) => elements.video.requestVideoFrameCallback(callback),
      (identifier) => elements.video.cancelVideoFrameCallback?.(identifier),
      (_now, metadata) => {
        if (generation !== selectionGeneration) return;
        renderAt(Math.round(metadata.mediaTime * 1000));
        requestFrameLoop();
      },
    );
  } else {
    frameCallbacks.request(
      (callback) => requestAnimationFrame(callback),
      (identifier) => cancelAnimationFrame(identifier),
      () => {
        if (generation !== selectionGeneration) return;
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
  const playing = !elements.video.paused && !elements.video.ended && events.length > 0;
  elements.play.classList.toggle("is-playing", playing);
  elements.play.setAttribute("aria-label", playing ? "Pause recording" : "Play recording");
  elements.playLabel.textContent = playing ? "Pause" : "Play";
  if (playing) requestFrameLoop();
}

function renderStateBlocks() {
  const fragment = document.createDocumentFragment();
  blockNodes = new Map();
  selectedBlock = null;
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
    marker.style.left = `${(timeMs / durationMs) * 100}%`;
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
    elements.overlayStatus.textContent = "Complete Näky overlay";
  } else {
    setStateViewportAvailability(false);
    elements.modeExplanation.textContent = "In the released comparison, a multimodal reader receives 20 sampled pixel frames. The state stream is not sent.";
    elements.overlayStatus.textContent = "Pixels only";
  }
  renderAt(currentMediaMs());
}

function setLink(link, href) {
  if (href) {
    link.href = href;
    link.removeAttribute("aria-disabled");
  } else {
    link.removeAttribute("href");
    link.setAttribute("aria-disabled", "true");
  }
}

function setSourceSummary(entry, manifest) {
  const dataset = document.createElement("a");
  dataset.href = manifest.recording.dataset_url;
  dataset.textContent = manifest.recording.dataset;
  const license = document.createElement("a");
  license.href = manifest.recording.license_url;
  license.textContent = manifest.recording.license;
  const member = document.createElement("code");
  member.textContent = manifest.recording.member;
  elements.sourceSummary.replaceChildren(
    document.createTextNode(`${entry.title} uses `),
    member,
    document.createTextNode(" from "),
    dataset,
    document.createTextNode(` (${manifest.recording.split}), under `),
    license,
    document.createTextNode(". Näky-generated overlay and text are derived from this source. "),
  );
  const full = document.createElement("a");
  full.href = "demo/ATTRIBUTION.md";
  full.textContent = "Full attribution and transformation notes.";
  elements.sourceSummary.append(full);
}

function clearRecording(entry, generation) {
  invalidateFrameLoop();
  elements.video.pause();
  elements.video.removeAttribute("src");
  elements.video.dataset.selectionGeneration = String(generation);
  elements.video.dataset.expectedSource = "";
  elements.video.load();
  events = [];
  stateBlocks = [];
  structuralSamples = [];
  durationMs = 0;
  presentationFrame = null;
  blockNodes = new Map();
  selectedBlock = null;
  followPlayhead = true;
  programmaticScrollUntil = 0;
  elements.follow.hidden = true;
  elements.overlay.replaceChildren();
  elements.markerRail.replaceChildren();
  elements.stateViewport.replaceChildren();
  elements.timeline.min = "0";
  elements.timeline.max = "0";
  elements.timeline.value = "0";
  elements.time.value = "0:00 / 0:00";
  elements.play.disabled = true;
  elements.timeline.disabled = true;
  elements.videoError.hidden = true;
  elements.videoDownload.hidden = true;
  setLink(elements.videoFallbackDownload, null);
  setLink(elements.videoDownload, null);
  setLink(elements.rawState, null);
  setLink(elements.rawEvents, null);
  elements.recordingSummary.textContent = `${entry.title} · Loading authentic output…`;
  elements.moment.textContent = "Loading authentic output…";
  elements.stateStatus.textContent = "Loading state stream…";
  elements.demoKicker.textContent = "Loading recording, one shared playhead";
  elements.demo.setAttribute("aria-busy", "true");
  updatePlayState();
}

function validateLoadedRecording(manifest, parsedEvents, parsedBlocks) {
  if (parsedEvents.length !== manifest.output.event_count) throw new Error("event count differs from manifest");
  const expectedSource = `urn:naky:stream:${manifest.output.stream_id}`;
  if (parsedEvents.some((event) => event.source !== expectedSource)) throw new Error("event stream identity differs from manifest");
  const first = parsedEvents[0];
  if (first.type !== TYPES.SNAPSHOT || first.data.frame.width !== manifest.presentation.width ||
      first.data.frame.height !== manifest.presentation.height) throw new Error("event frame differs from manifest");
  if (parsedEvents.at(-1).frametime > manifest.presentation.duration_ms ||
      parsedBlocks.at(-1).frametime > manifest.presentation.duration_ms) throw new Error("output extends beyond recording duration");
  if (parsedEvents[0].frametime !== manifest.output.first_event_ms ||
      parsedEvents.at(-1).frametime !== manifest.output.last_event_ms) throw new Error("event time bounds differ from manifest");
  const familyCounts = {};
  parsedEvents.forEach((event) => {
    const family = EVENT_FAMILY_NAMES.get(event.type);
    familyCounts[family] = (familyCounts[family] ?? 0) + 1;
  });
  const actualFamilies = Object.entries(familyCounts).sort(([left], [right]) => left.localeCompare(right));
  const expectedFamilies = Object.entries(manifest.output.event_family_counts).sort(([left], [right]) => left.localeCompare(right));
  if (JSON.stringify(actualFamilies) !== JSON.stringify(expectedFamilies)) throw new Error("event family counts differ from manifest");
  const actualStructural = [...new Set(parsedEvents
    .filter((event) => event.type !== TYPES.ACTIVITY)
    .map((event) => event.frametime))];
  if (actualStructural.length !== manifest.output.structural_sample_times_ms.length ||
      actualStructural.some((timeMs, index) => timeMs !== manifest.output.structural_sample_times_ms[index])) {
    throw new Error("structural sample times differ from manifest");
  }
}

function configureRecording(entry, manifest) {
  durationMs = manifest.presentation.duration_ms;
  const width = manifest.presentation.width;
  const height = manifest.presentation.height;
  presentationFrame = {width, height};
  elements.timeline.max = String(durationMs);
  elements.stage.style.setProperty("--frame-ratio", `${width} / ${height}`);
  elements.stage.style.setProperty("--stage-width", `${width <= height ? Math.min(width, 360) : Math.min(width, 720)}px`);
  elements.overlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
  elements.video.setAttribute("aria-label", `${entry.title} screen recording used to demonstrate Näky`);
  elements.demoKicker.textContent = `${durationLabel(durationMs)}, one shared playhead`;
  elements.recordingSummary.textContent = `${entry.title} · ${entry.subtitle}`;
  setLink(elements.rawState, demoPath(manifest.output.state.path));
  setLink(elements.rawEvents, demoPath(manifest.output.events.path));
  setLink(elements.videoDownload, demoPath(manifest.presentation.media.path));
  setLink(elements.videoFallbackDownload, demoPath(manifest.presentation.media.path));
  elements.videoDownload.hidden = false;
  setSourceSummary(entry, manifest);
}

function waitForMediaMetadata(generation) {
  if (elements.video.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve(true);
  return new Promise((resolve) => {
    let complete = false;
    const finish = (ready) => {
      if (complete) return;
      complete = true;
      clearTimeout(timeout);
      elements.video.removeEventListener("loadedmetadata", loaded);
      elements.video.removeEventListener("error", failed);
      resolve(ready && generation === selectionGeneration);
    };
    const loaded = () => finish(true);
    const failed = () => finish(false);
    const timeout = setTimeout(() => finish(false), 15000);
    elements.video.addEventListener("loadedmetadata", loaded);
    elements.video.addEventListener("error", failed);
  });
}

async function loadRecording(entry, autoplay = true) {
  selectionGeneration += 1;
  const generation = selectionGeneration;
  fetchController?.abort();
  fetchController = new AbortController();
  clearRecording(entry, generation);
  try {
    const manifest = await fetchAsset(demoPath(entry.manifest), "json", fetchController.signal);
    if (generation !== selectionGeneration) return;
    if (!manifestIsValid(manifest, entry)) throw new Error("demo manifest differs from the public gallery contract");
    const [eventText, stateText] = await Promise.all([
      fetchAsset(demoPath(manifest.output.events.path), "text", fetchController.signal),
      fetchAsset(demoPath(manifest.output.state.path), "text", fetchController.signal),
    ]);
    if (generation !== selectionGeneration) return;
    const parsedEvents = parseEventStream(eventText);
    const parsedBlocks = indexStateBlocks(stateText);
    validateLoadedRecording(manifest, parsedEvents, parsedBlocks);

    events = parsedEvents;
    stateBlocks = parsedBlocks;
    structuralSamples = [...manifest.output.structural_sample_times_ms];
    configureRecording(entry, manifest);
    renderStateBlocks();
    renderMarkers();
    renderAt(0);
    elements.video.loop = true;
    const mediaUrl = new URL(demoPath(manifest.presentation.media.path), document.baseURI).href;
    elements.video.dataset.expectedSource = mediaUrl;
    elements.video.src = mediaUrl;
    elements.video.load();
    elements.demo.removeAttribute("aria-busy");
    const mediaReady = await waitForMediaMetadata(generation);
    if (generation !== selectionGeneration) return;
    elements.timeline.disabled = false;
    elements.play.disabled = !mediaReady;
    if (!mediaReady) elements.videoError.hidden = false;
    if (mediaReady && autoplay && !reduceMotion) {
      try { await elements.video.play(); } catch { updatePlayState(); }
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return;
    if (generation !== selectionGeneration) return;
    elements.moment.textContent = "Demo output could not be loaded";
    elements.stateStatus.textContent = error instanceof Error ? error.message : "Demo output unavailable";
    elements.videoError.hidden = false;
    elements.demo.removeAttribute("aria-busy");
    updatePlayState();
  }
}

function bindInteractions() {
  elements.recordingSelect.addEventListener("change", () => {
    const entry = catalogBySlug.get(elements.recordingSelect.value);
    if (entry) loadRecording(entry);
  });
  elements.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  elements.play.addEventListener("click", async () => {
    if (!elements.video.paused) {
      invalidateFrameLoop();
      elements.video.pause();
      return;
    }
    if (elements.video.ended || elements.video.currentTime * 1000 >= durationMs - 20) elements.video.currentTime = 0;
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
  elements.video.addEventListener("error", () => {
    if (!elements.video.currentSrc || elements.video.currentSrc !== elements.video.dataset.expectedSource) return;
    invalidateFrameLoop();
    elements.play.disabled = true;
    elements.timeline.disabled = false;
    elements.videoError.hidden = false;
    updatePlayState();
  });
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

function populateSelector(recordings) {
  const options = document.createDocumentFragment();
  recordings.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.slug;
    option.textContent = `${entry.title} — ${entry.subtitle}`;
    options.append(option);
  });
  elements.recordingSelect.replaceChildren(options);
  elements.recordingSelect.disabled = false;
}

async function initialize() {
  elements.play.disabled = true;
  elements.timeline.disabled = true;
  elements.recordingSelect.disabled = true;
  bindInteractions();
  try {
    const value = await fetchAsset("demo/gallery.json", "json");
    if (!catalogIsValid(value)) throw new Error("demo gallery contract differs");
    catalog = value.recordings;
    catalogBySlug = new Map(catalog.map((entry) => [entry.slug, entry]));
    populateSelector(catalog);
    await loadRecording(catalog[0]);
  } catch (error) {
    elements.recordingSummary.textContent = "Recording gallery could not be loaded";
    elements.moment.textContent = "Demo output could not be loaded";
    elements.stateStatus.textContent = error instanceof Error ? error.message : "Demo output unavailable";
    elements.videoError.hidden = false;
    elements.demo.removeAttribute("aria-busy");
  }
}

initialize();
