const TYPES = Object.freeze({
  SNAPSHOT: "dev.naky.screenevents.snapshot.v0",
  APPEARED: "dev.naky.screenevents.element.appeared.v0",
  MOVED: "dev.naky.screenevents.element.moved.v0",
  CONTENT: "dev.naky.screenevents.element.content_changed.v0",
  DISAPPEARED: "dev.naky.screenevents.element.disappeared.v0",
  RESIZED: "dev.naky.screenevents.screen.resized.v0",
  ACTIVITY: "dev.naky.screenevents.activity.observed.v0",
});

const TYPE_SET = new Set(Object.values(TYPES));
export const RECENT_CHANGE_WINDOW_MS = 1000;
const ENVELOPE_KEYS = new Set([
  "specversion", "id", "source", "sequence", "frametime",
  "datacontenttype", "type", "data",
]);

function fail(message) {
  throw new Error(`invalid ScreenEvents stream: ${message}`);
}

function exactKeys(value, expected, context) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${context} must be an object`);
  const actual = Object.keys(value);
  if (actual.length !== expected.size || actual.some((key) => !expected.has(key))) {
    fail(`${context} fields differ`);
  }
}

function uint(value, context, positive = false) {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0)) fail(`${context} must be an unsigned integer`);
  return value;
}

function rect(value, frame, context) {
  exactKeys(value, new Set(["x", "y", "width", "height"]), context);
  const result = {
    x: uint(value.x, `${context}.x`),
    y: uint(value.y, `${context}.y`),
    width: uint(value.width, `${context}.width`, true),
    height: uint(value.height, `${context}.height`, true),
  };
  if (frame && (result.x + result.width > frame.width || result.y + result.height > frame.height)) {
    fail(`${context} falls outside the frame`);
  }
  return result;
}

function frameSize(value, context) {
  exactKeys(value, new Set(["width", "height"]), context);
  return {width: uint(value.width, `${context}.width`, true), height: uint(value.height, `${context}.height`, true)};
}

function element(value, frame, context) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${context} must be an object`);
  const allowed = new Set(["id", "box", "text", "role", "state", "confidence"]);
  const required = new Set(["id", "box", "confidence"]);
  if (Object.keys(value).some((key) => !allowed.has(key)) || [...required].some((key) => !(key in value))) {
    fail(`${context} fields differ`);
  }
  if (!/^e[0-9]{6,}$/.test(value.id) || value.id === "e000000") fail(`${context}.id is not canonical`);
  if ("text" in value && typeof value.text !== "string") fail(`${context}.text must be a string`);
  if ("role" in value && typeof value.role !== "string") fail(`${context}.role must be a string`);
  if ("state" in value) {
    if (!value.state || typeof value.state !== "object" || Array.isArray(value.state) ||
        Object.values(value.state).some((item) => typeof item !== "string")) fail(`${context}.state is invalid`);
  }
  if (typeof value.confidence !== "number" || !Number.isFinite(value.confidence) || value.confidence < 0 || value.confidence > 1) {
    fail(`${context}.confidence is invalid`);
  }
  return {...value, box: rect(value.box, frame, `${context}.box`), state: value.state ? {...value.state} : {}};
}

function sameRect(left, right) {
  return left.x === right.x && left.y === right.y && left.width === right.width && left.height === right.height;
}

function cloneElement(value) {
  return {...value, box: {...value.box}, state: {...value.state}};
}

function sameElementObservation(left, right) {
  if (!left || !right || left.id !== right.id || left.text !== right.text ||
      left.role !== right.role ||
      !sameRect(left.box, right.box)) return false;
  const leftState = Object.entries(left.state);
  const rightState = Object.entries(right.state);
  return leftState.length === rightState.length &&
    leftState.every(([key, value]) => right.state[key] === value);
}

function activitySnapshots(events) {
  let elements = new Map();
  const snapshots = [];
  for (const event of events) {
    if (event.type === TYPES.SNAPSHOT) {
      elements = new Map(event.data.elements.map((item) => [item.id, cloneElement(item)]));
    } else if (event.type === TYPES.APPEARED || event.type === TYPES.CONTENT) {
      const item = cloneElement(event.data.element);
      elements.set(item.id, item);
    } else if (event.type === TYPES.MOVED) {
      elements.get(event.data.id).box = {...event.data.to};
    } else if (event.type === TYPES.DISAPPEARED) {
      elements.delete(event.data.id);
    } else if (event.type === TYPES.ACTIVITY) {
      if (event.data.kind === "element_region_changed") {
        const referenced = elements.get(event.data.element_id);
        snapshots.push({...event.data, frametime: event.frametime, box: {...referenced.box}, element: cloneElement(referenced)});
      } else {
        snapshots.push({...event.data, frametime: event.frametime, box: {...event.data.box}});
      }
    }
  }
  return snapshots;
}

function validateAndApply(events) {
  let source = null;
  let previousTime = -1;
  let frame = null;
  let elements = new Map();
  for (let index = 0; index < events.length; index += 1) {
    const event = events[index];
    exactKeys(event, ENVELOPE_KEYS, `event ${index}`);
    if (event.specversion !== "1.0" || event.datacontenttype !== "application/json") fail(`event ${index} envelope differs`);
    if (typeof event.id !== "string" || typeof event.source !== "string" || !event.source.startsWith("urn:naky:stream:")) fail(`event ${index} identity is invalid`);
    if (event.sequence !== index) fail(`event sequence must be contiguous at ${index}`);
    uint(event.frametime, `event ${index}.frametime`);
    if (event.frametime < previousTime) fail(`event time decreases at ${index}`);
    if (source === null) source = event.source;
    if (event.source !== source) fail("event sources differ");
    if (!TYPE_SET.has(event.type)) fail(`unknown event family ${event.type}`);
    previousTime = event.frametime;

    if (event.type === TYPES.SNAPSHOT) {
      exactKeys(event.data, new Set(["frame", "elements"]), `event ${index}.data`);
      frame = frameSize(event.data.frame, `event ${index}.data.frame`);
      if (!Array.isArray(event.data.elements)) fail(`event ${index}.data.elements must be an array`);
      elements = new Map();
      for (const raw of event.data.elements) {
        const item = element(raw, frame, `event ${index}.element`);
        if (elements.has(item.id)) fail(`duplicate element ${item.id}`);
        elements.set(item.id, item);
      }
      continue;
    }
    if (!frame) fail(`delta before snapshot at ${index}`);
    if (event.type === TYPES.APPEARED || event.type === TYPES.CONTENT) {
      exactKeys(event.data, new Set(["element"]), `event ${index}.data`);
      const item = element(event.data.element, frame, `event ${index}.data.element`);
      if (event.type === TYPES.APPEARED && elements.has(item.id)) fail(`duplicate element ${item.id}`);
      if (event.type === TYPES.CONTENT && !elements.has(item.id)) fail(`unknown element ${item.id}`);
      elements.set(item.id, item);
    } else if (event.type === TYPES.MOVED) {
      exactKeys(event.data, new Set(["id", "from", "to"]), `event ${index}.data`);
      const current = elements.get(event.data.id);
      if (!current) fail(`unknown element ${event.data.id}`);
      const from = rect(event.data.from, frame, `event ${index}.data.from`);
      const to = rect(event.data.to, frame, `event ${index}.data.to`);
      if (!sameRect(current.box, from)) fail(`move origin differs for ${event.data.id}`);
      current.box = to;
    } else if (event.type === TYPES.DISAPPEARED) {
      exactKeys(event.data, new Set(["id"]), `event ${index}.data`);
      if (!elements.delete(event.data.id)) fail(`unknown element ${event.data.id}`);
    } else if (event.type === TYPES.RESIZED) {
      exactKeys(event.data, new Set(["from", "to"]), `event ${index}.data`);
      const from = frameSize(event.data.from, `event ${index}.data.from`);
      const to = frameSize(event.data.to, `event ${index}.data.to`);
      if (from.width !== frame.width || from.height !== frame.height) fail("resize origin differs");
      frame = to;
    } else if (event.type === TYPES.ACTIVITY) {
      if (!event.data || typeof event.data !== "object") fail(`event ${index}.data must be an object`);
      uint(event.data.from_frametime, `event ${index}.data.from_frametime`);
      if (event.data.from_frametime >= event.frametime) fail(`invalid activity interval at ${index}`);
      if (event.data.kind === "element_region_changed") {
        exactKeys(event.data, new Set(["from_frametime", "kind", "element_id"]), `event ${index}.data`);
        if (!elements.has(event.data.element_id)) fail(`unknown element ${event.data.element_id}`);
      } else if (event.data.kind === "region_translated") {
        exactKeys(event.data, new Set(["from_frametime", "kind", "box", "dx", "dy"]), `event ${index}.data`);
        rect(event.data.box, frame, `event ${index}.data.box`);
        if (!Number.isSafeInteger(event.data.dx) || !Number.isSafeInteger(event.data.dy) || (event.data.dx === 0 && event.data.dy === 0)) fail(`invalid translation at ${index}`);
      } else fail(`unknown activity kind at ${index}`);
    }
  }
  if (!events.length || events[0].type !== TYPES.SNAPSHOT) fail("stream must start with a snapshot");
  return events;
}

export function parseEventStream(text) {
  if (typeof text !== "string" || !text.endsWith("\n")) fail("NDJSON must end with LF");
  const lines = text.slice(0, -1).split("\n");
  if (lines.some((line) => !line || line.includes("\r"))) fail("NDJSON rows must be nonempty LF UTF-8 text");
  const events = lines.map((line, index) => {
    try { return JSON.parse(line); } catch { fail(`row ${index + 1} is not JSON`); }
  });
  return validateAndApply(events);
}

export function secondsToEventMs(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
    throw new Error("media time must be a finite nonnegative number");
  }
  return Math.round(seconds * 1000);
}

export function createPresentationClock(presentedFramesSupported) {
  if (typeof presentedFramesSupported !== "boolean") throw new Error("presented-frame support must be boolean");
  let selectedTimeMs = 0;
  const selectCurrent = (seconds) => {
    selectedTimeMs = secondsToEventMs(seconds);
    return selectedTimeMs;
  };
  return Object.freeze({
    reset() {
      selectedTimeMs = 0;
      return selectedTimeMs;
    },
    presented: selectCurrent,
    current: selectCurrent,
    passive(seconds) {
      return presentedFramesSupported ? selectedTimeMs : selectCurrent(seconds);
    },
  });
}

export function createFrameCallbackLifecycle() {
  let generation = 0;
  let pending = null;

  return Object.freeze({
    request(schedule, cancel, callback) {
      if (pending !== null) return false;
      if (typeof schedule !== "function" || typeof cancel !== "function" || typeof callback !== "function") {
        throw new Error("frame callback lifecycle requires schedule, cancel, and callback functions");
      }
      const requestedGeneration = generation;
      let fired = false;
      const identifier = schedule((...args) => {
        fired = true;
        if (requestedGeneration !== generation) return;
        pending = null;
        callback(...args);
      });
      if (!fired) pending = {identifier, cancel};
      return true;
    },
    invalidate() {
      generation += 1;
      const request = pending;
      pending = null;
      if (request !== null) request.cancel(request.identifier);
    },
  });
}

function samplePosition(sampleBoundaries, timeMs) {
  if (!Array.isArray(sampleBoundaries)) throw new Error("structural samples must be an array");
  let latest = null;
  let next = null;
  let previous = -1;
  for (const value of sampleBoundaries) {
    uint(value, "structural sample");
    if (value <= previous) throw new Error("structural samples must be strictly increasing");
    previous = value;
    if (value <= timeMs) latest = value;
    else if (next === null) next = value;
  }
  return {latest, next};
}

export function stateAt(events, timeMs, sampleBoundaries = null) {
  uint(timeMs, "playhead time");
  let frame = null;
  let elements = new Map();
  let trails = [];
  const changes = [];
  for (const event of events) {
    if (event.frametime > timeMs) continue;
    if (event.type === TYPES.SNAPSHOT) {
      frame = {...event.data.frame};
      elements = new Map(event.data.elements.map((item) => [item.id, cloneElement(item)]));
      trails = [];
    } else if (event.type === TYPES.APPEARED || event.type === TYPES.CONTENT) {
      const item = cloneElement(event.data.element);
      elements.set(item.id, item);
      changes.push({kind: event.type === TYPES.APPEARED ? "appeared" : "content", frametime: event.frametime, elementId: item.id, box: {...item.box}, element: cloneElement(item)});
    } else if (event.type === TYPES.MOVED) {
      const current = elements.get(event.data.id);
      current.box = {...event.data.to};
      trails.push({elementId: event.data.id, from: {...event.data.from}, to: {...event.data.to}, frametime: event.frametime});
      changes.push({kind: "moved", frametime: event.frametime, elementId: event.data.id, box: {...event.data.to}, element: cloneElement(current)});
    } else if (event.type === TYPES.DISAPPEARED) {
      const former = elements.get(event.data.id);
      elements.delete(event.data.id);
      changes.push({kind: "disappeared", frametime: event.frametime, elementId: event.data.id, box: {...former.box}, element: cloneElement(former)});
    } else if (event.type === TYPES.RESIZED) frame = {...event.data.to};
  }
  const allActivities = activitySnapshots(events);
  const activities = allActivities.filter((activity) => {
    const duration = activity.frametime - activity.from_frametime;
    return activity.frametime <= timeMs && timeMs < activity.frametime + duration;
  });
  const samples = sampleBoundaries ?? [...new Set(events.filter((event) => event.type !== TYPES.ACTIVITY).map((event) => event.frametime))];
  const {latest, next} = samplePosition(samples, timeMs);
  const staleIds = new Set(allActivities.filter((activity) =>
    activity.kind === "element_region_changed" && activity.element.text !== undefined &&
    activity.frametime > (latest ?? -1) && activity.frametime < timeMs &&
    sameElementObservation(elements.get(activity.element.id), activity.element)
  ).map((activity) => activity.element.id));
  return {
    frame,
    elements: [...elements.values()].map(cloneElement),
    staleIds: [...staleIds],
    latestStructuralSampleMs: latest,
    nextStructuralSampleMs: next,
    awaitingStructuralRefresh: staleIds.size > 0,
    activities,
    trails,
    changes: changes.filter((item) => timeMs - item.frametime <= RECENT_CHANGE_WINDOW_MS),
  };
}

export function indexEventRows(text, events = parseEventStream(text)) {
  const lines = text.slice(0, -1).split("\n");
  return events.map((event, index) => ({frametime: event.frametime, sequence: event.sequence, text: `${lines[index]}\n`}));
}

export function indexStateBlocks(text) {
  if (typeof text !== "string" || !text.endsWith("\n") || text.includes("\r")) fail("state stream must be LF text ending in LF");
  const lines = text.slice(0, -1).split("\n");
  const header = lines.shift();
  if (header !== "screenevents-state") fail("state stream header differs");
  const blocks = [];
  let current = null;
  for (const line of lines) {
    const match = /^@([0-9]+)(?: .*)?$/.exec(line);
    if (match) {
      const frametime = Number(match[1]);
      uint(frametime, "state stream frametime");
      if (current && frametime < current.frametime) fail("state stream time decreases");
      current = {frametime, text: `${line}\n`};
      blocks.push(current);
    } else {
      if (!current || !line) fail("state stream fact is outside a time block");
      current.text += `${line}\n`;
    }
  }
  if (!blocks.length) fail("state stream has no blocks");
  blocks[0].text = `${header}\n${blocks[0].text}`;
  return blocks;
}

export function selectTimeGroup(groups, timeMs) {
  uint(timeMs, "playhead time");
  let selected = null;
  for (const group of groups) {
    if (group.frametime > timeMs) break;
    selected = group;
  }
  return selected;
}

export {TYPES};
