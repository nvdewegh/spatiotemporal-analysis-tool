'use strict';
// Local A2 viewer. Everything happens in this browser tab: the folder is read with the browser's folder picker,
// data.js is parsed as JSON text (never executed), and camera frame sheets are decoded only for the selected
// encounter, as they are needed. No file contents, frames or paths are sent anywhere; the only messages to
// Streamlit are the component handshake and the frame height.

const byId = id => document.getElementById(id);
const folder = byId('folder'), status = byId('status'), viewer = byId('viewer');
const encounter = byId('encounter'), interval = byId('interval'), timeline = byId('timeline');
const play = byId('play'), camera = byId('camera'), trajectory = byId('trajectory');
const mediaError = byId('media-error'), sampleOut = byId('sample'), sourceOut = byId('source');

const SPRITE_FRAMES = 4;          // 2 x 2 camera frames per sheet (as in the standalone viewer)
const CACHE_LIMIT = 10;           // decoded sheets kept, for the selected encounter only
let data = null, files = null, frame = 0, id = 0, missing = new Map();
let cache = new Map(), selectionVersion = 0, drawVersion = 0;
let playing = false, nextDue = 0;
// Test and diagnostic hooks (local only): what is on screen and when each sample was drawn
const state = window.__a2viewer = {drawn: [], shownSample: null, decodedSheets: 0};

// ---------------------------------------------------------------------------------------------- data
function parseA2(text) {
  // data.js is a JS assignment around JSON. Parse the JSON part only; the file is never executed.
  const payload = /^\s*window\.ENCOUNTERS\s*=\s*(\{[\s\S]*\})\s*;?\s*$/.exec(text);
  if (!payload) throw new Error('data.js in this folder is not the A2 trajectory file. ' + RECOVER_FOLDER);
  const d = JSON.parse(payload[1]);
  const b = d.bounds;
  if (d.frameCount !== 137 || d.fps !== 24 || d.referenceFrame !== 74 ||
      d.analysisStart !== 2 || d.analysisEnd !== 122 || !b ||
      ![b.xmin, b.xmax, b.ymin, b.ymax].every(Number.isFinite) ||
      b.xmin >= b.xmax || b.ymin >= b.ymax || !Array.isArray(d.encounters) ||
      d.encounters.length !== 26 || !d.encounters.every((e, i) => e && e.id === i &&
        Array.isArray(e.points) && e.points.length === 2 && e.points.every(track =>
          Array.isArray(track) && track.length === d.frameCount && track.every(p =>
            Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))))) {
    throw new Error('The A2 trajectory data are incomplete or use a different format. ' + RECOVER_ZIP);
  }
  return d;
}

const RECOVER_FOLDER = 'Extract LocalComparisonViewer_v01-1.zip first, then choose the extracted folder that ' +
  'contains index.html, data.js and assets (not the ZIP file itself).';
const RECOVER_ZIP = 'Download the viewer ZIP again, extract all of it, and open the extracted folder.';

const pad = (n, w) => String(n).padStart(w, '0');
const sheetPath = (kind, enc, sheet) => `assets/${kind}/${pad(enc, 2)}/${pad(sheet, 3)}.webp`;
const sheetCount = () => Math.ceil(data.frameCount / SPRITE_FRAMES);
function sheetFile(enc, sheet) {
  return files.get(sheetPath('detail', enc, sheet)) || files.get(sheetPath('overview', enc, sheet));
}
function missingSheets(enc) {
  const out = [];
  for (let s = 0; s < sheetCount(); s++) if (!sheetFile(enc, s)) out.push(sheetPath('detail', enc, s));
  return out;
}

// --------------------------------------------------------------------------------------------- timing
const range = () => interval.value === 'full' ? [0, data.frameCount - 1] : [data.analysisStart, data.analysisEnd];
const rel = s => (s - data.referenceFrame) / data.fps;
const signed = t => (t < 0 ? '−' : t > 0 ? '+' : '') + Math.abs(t).toFixed(2);

// ------------------------------------------------------------------------------------- frame sheets
function sheet(enc, index) {
  // Decoded sheet as an ImageBitmap (no object URL). Only the selected encounter is ever cached.
  const key = `${enc}/${index}`;
  if (cache.has(key)) {
    const hit = cache.get(key); cache.delete(key); cache.set(key, hit);       // most recently used last
    return hit;
  }
  const file = sheetFile(enc, index);
  const promise = file
    ? createImageBitmap(file).then(bitmap => { state.decodedSheets++; return bitmap; })
    : Promise.reject(Object.assign(new Error('missing'), {path: sheetPath('detail', enc, index)}));
  cache.set(key, promise);
  promise.catch(() => cache.delete(key));
  while (cache.size > CACHE_LIMIT) {
    const [oldKey, old] = cache.entries().next().value;
    cache.delete(oldKey);
    old.then(b => b.close(), () => {});
  }
  return promise;
}

function dropCache() {
  for (const p of cache.values()) p.then(b => b.close(), () => {});
  cache = new Map();
}

// ------------------------------------------------------------------------------------------- drawing
function paintTrajectories(enc, sample) {
  const ctx = trajectory.getContext('2d');
  const {width, height} = trajectory;
  const b = data.bounds, m = {l: 56, r: 18, t: 22, b: 44};
  const scale = Math.min((width - m.l - m.r) / (b.xmax - b.xmin), (height - m.t - m.b) / (b.ymax - b.ymin));
  const x0 = m.l + ((width - m.l - m.r) - (b.xmax - b.xmin) * scale) / 2;
  const y0 = m.t + ((height - m.t - m.b) - (b.ymax - b.ymin) * scale) / 2;
  const xy = ([x, y]) => [x0 + (x - b.xmin) * scale, y0 + (b.ymax - y) * scale];
  ctx.fillStyle = '#f7f9fb';
  ctx.fillRect(0, 0, width, height);
  ctx.font = '20px system-ui, sans-serif';             // canvas is 960 px wide and shown at about half size
  ctx.fillStyle = '#64747e';
  ctx.strokeStyle = '#e3e8ec';
  ctx.lineWidth = 1;
  for (let x = Math.ceil(b.xmin / 10) * 10; x <= b.xmax; x += 10) {
    const [px] = xy([x, b.ymin]);
    ctx.beginPath(); ctx.moveTo(px, y0); ctx.lineTo(px, y0 + (b.ymax - b.ymin) * scale); ctx.stroke();
    ctx.textAlign = 'center'; ctx.fillText(String(x), px, height - 14);
  }
  for (let y = Math.ceil(b.ymin / 5) * 5; y <= b.ymax; y += 5) {
    const [, py] = xy([b.xmin, y]);
    ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x0 + (b.xmax - b.xmin) * scale, py); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(String(y), x0 - 8, py + 7);
  }
  ctx.textAlign = 'left';
  ctx.fillText('x (m)', width - 62, height - 14);
  ctx.fillText('y (m)', 6, 20);
  data.encounters[enc].points.forEach((points, obj) => {
    const colour = obj === 0 ? '#126b9c' : '#c76a14';
    ctx.strokeStyle = colour;
    ctx.globalAlpha = 0.2; ctx.lineWidth = 1.5;                         // whole path, faint
    ctx.beginPath(); ctx.moveTo(...xy(points[0]));
    for (let i = 1; i < points.length; i++) ctx.lineTo(...xy(points[i]));
    ctx.stroke();
    ctx.globalAlpha = 0.85; ctx.lineWidth = 2.5;                        // trail up to this sample
    ctx.beginPath(); ctx.moveTo(...xy(points[0]));
    for (let i = 1; i <= sample; i++) ctx.lineTo(...xy(points[i]));
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.fillStyle = colour;
    ctx.beginPath(); ctx.arc(...xy(points[sample]), 6, 0, 2 * Math.PI); ctx.fill();
  });
  trajectory.dataset.sample = sample;
  trajectory.dataset.encounter = enc;
  trajectory.setAttribute('aria-label', `Config_${enc} point animation, sample ${sample}`);
}

function paintCamera(bitmap, enc, sample) {
  const w = bitmap.width / 2, h = bitmap.height / 2;
  if (camera.width !== Math.round(w) || camera.height !== Math.round(h)) {
    camera.width = Math.round(w); camera.height = Math.round(h);
  }
  const cell = sample % SPRITE_FRAMES;
  camera.getContext('2d').drawImage(bitmap, (cell % 2) * w, Math.floor(cell / 2) * h, w, h, 0, 0, camera.width, camera.height);
  camera.dataset.sample = sample;
  camera.dataset.encounter = enc;
  camera.setAttribute('aria-label', `Config_${enc} camera frame, sample ${sample}`);
}

function blankCamera(enc, sample, message) {
  const ctx = camera.getContext('2d');
  ctx.fillStyle = '#eef1f4'; ctx.fillRect(0, 0, camera.width, camera.height);
  ctx.fillStyle = '#6b7780'; ctx.font = '16px system-ui, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText(message, camera.width / 2, camera.height / 2);
  camera.dataset.sample = 'none';
  camera.dataset.encounter = enc;
  camera.setAttribute('aria-label', `Config_${enc} camera frame unavailable at sample ${sample}`);
}

function label(sample) {
  sampleOut.textContent = `Sample ${sample} / ${data.frameCount - 1} · t = ${signed(rel(sample))} s`;
  timeline.value = sample;
  timeline.setAttribute('aria-valuetext', `Sample ${sample}, ${rel(sample).toFixed(2)} seconds from the reference event`);
}

async function show(sample) {
  // Both views change in the same synchronous step, after the camera frame is ready: never out of step.
  const enc = id, token = ++drawVersion;
  let bitmap = null, error = null;
  try { bitmap = await sheet(enc, Math.floor(sample / SPRITE_FRAMES)); } catch (e) { error = e; }
  if (token !== drawVersion) return false;                             // a newer request (scrub, switch) won
  if (bitmap) {
    paintCamera(bitmap, enc, sample);
    mediaError.hidden = true;
  } else {
    stop();
    blankCamera(enc, sample, 'Camera frame not available');
    mediaError.textContent = error && error.path
      ? `Config_${enc}: camera frames are missing (${error.path}). The point animation still works. To recover, ` +
        'extract the complete viewer ZIP again without moving files out of the assets folder, then click ' +
        '“Open downloaded A2 viewer folder” and select the new folder.'
      : `Config_${enc}: this camera frame could not be read. Extract the viewer ZIP again and reopen the folder.`;
    mediaError.hidden = false;
  }
  paintTrajectories(enc, sample);
  label(sample);
  state.shownSample = sample;
  state.drawn.push([performance.now(), enc, sample, bitmap ? sample : null]);
  if (state.drawn.length > 2000) state.drawn.splice(0, 1000);
  // Prefetch the next sheet of this encounter so playback does not wait for decoding
  const next = Math.floor(sample / SPRITE_FRAMES) + 1;
  if (next < sheetCount() && sheetFile(enc, next)) sheet(enc, next).catch(() => {});
  return true;
}

function go(sample) {
  const [min, max] = range();
  frame = Math.max(min, Math.min(max, Math.round(sample)));
  return show(frame);
}

// ------------------------------------------------------------------------------------------ playback
function stop() {
  playing = false;
  play.textContent = 'Play';
}

async function start() {
  const [min, max] = range();
  if (frame >= max) await go(min);
  playing = true;
  play.textContent = 'Pause';
  nextDue = performance.now() + 1000 / data.fps;
  requestAnimationFrame(tick);
}

let rendering = false;
async function tick(now) {
  if (!playing) return;
  if (!rendering && now >= nextDue) {
    const [, max] = range();
    if (frame >= max) { stop(); return; }
    // Fixed 24 Hz schedule; if the browser fell far behind (e.g. tab in background) restart the schedule
    nextDue = now - nextDue > 250 ? now + 1000 / data.fps : nextDue + 1000 / data.fps;
    rendering = true;
    try { await go(frame + 1); } finally { rendering = false; }
  }
  if (playing) requestAnimationFrame(tick);
}

// --------------------------------------------------------------------------------------------- folder
folder.addEventListener('change', async () => {
  if (!folder.files.length) return;
  const selection = ++selectionVersion;
  ++drawVersion;
  stop();
  viewer.hidden = true;
  dropCache();
  status.textContent = 'Opening your local viewer folder…';
  try {
    const paths = Array.from(folder.files, f => [f.webkitRelativePath || f.name, f]);
    const entry = paths.find(([path, f]) => (path === 'data.js' || path.endsWith('/data.js')) && f.size < 2000000 &&
      paths.some(([p]) => p === path.replace(/data\.js$/, 'index.html')));
    if (!entry) throw new Error('This folder does not contain index.html and data.js. ' + RECOVER_FOLDER);
    const prefix = entry[0].slice(0, -'data.js'.length);
    const localFiles = new Map(paths.filter(([path]) => path.startsWith(prefix))
      .map(([path, file]) => [path.slice(prefix.length), file]));
    const loaded = parseA2(await entry[1].text());
    if (selection !== selectionVersion) return;
    data = loaded;
    files = localFiles;
    missing = new Map(data.encounters.map(e => [e.id, missingSheets(e.id)]));
    const incomplete = [...missing].filter(([, m]) => m.length).map(([e]) => `Config_${e}`);
    if (incomplete.length === data.encounters.length) {
      throw new Error('The selected folder has no A2 camera frames (the assets folder is missing or empty). ' +
                      RECOVER_ZIP);
    }
    encounter.replaceChildren(...data.encounters.map(e => new Option(
      `Config_${e.id}${missing.get(e.id).length ? ' (camera frames missing)' : ''}`, e.id)));
    interval.value = 'analysis';
    [timeline.min, timeline.max] = range();
    status.textContent = incomplete.length
      ? `A2 viewer opened, but camera frames are missing for ${incomplete.join(', ')}. Their point animations work. ` +
        'To restore the camera clips, extract the complete ZIP again and reopen the folder.'
      : `A2 viewer ready: ${data.encounters.length} encounters, all camera frames present. The files stay on this device.`;
    viewer.hidden = false;
    id = 0;
    encounter.value = '0';
    await selectEncounter(0, data.analysisStart);
  } catch (error) {
    if (selection === selectionVersion) status.textContent = error.message;
  } finally {
    folder.value = '';
  }
});

async function selectEncounter(enc, sample) {
  const wasPlaying = playing;  // switching during playback keeps playing from the same sample
  stop();
  ++drawVersion;               // any frame still loading for the previous encounter is discarded
  dropCache();                 // decoded frames are kept for one encounter at a time
  id = enc;
  const e = data.encounters[enc];
  sourceOut.textContent = `Config_${enc} · source clip ${e.source}`;
  await go(sample);
  if (wasPlaying && id === enc) start();
}

encounter.addEventListener('change', () => selectEncounter(Number(encounter.value), frame));
interval.addEventListener('change', () => { stop(); [timeline.min, timeline.max] = range(); go(frame); });
timeline.addEventListener('input', () => { stop(); go(Number(timeline.value)); });
byId('reference').addEventListener('click', () => { stop(); go(data.referenceFrame); });
byId('previous').addEventListener('click', () => { stop(); go(frame - 1); });
byId('next').addEventListener('click', () => { stop(); go(frame + 1); });
play.addEventListener('click', () => (playing ? stop() : start()));
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });

// ------------------------------------------------------------------------------ Streamlit integration
// Only the component handshake and the frame height are sent; never files, frames, blob URLs or paths.
function send(type, fields) {
  if (window.parent !== window) window.parent.postMessage({isStreamlitMessage: true, type, ...fields}, '*');
}
function fitFrame() {
  send('streamlit:setFrameHeight', {height: Math.ceil(document.querySelector('main').getBoundingClientRect().height) + 8});
}
window.addEventListener('message', event => {
  if (event.source === window.parent && event.data?.type === 'streamlit:render') fitFrame();
});
new ResizeObserver(fitFrame).observe(document.querySelector('main'));
send('streamlit:componentReady', {apiVersion: 1});
fitFrame();
