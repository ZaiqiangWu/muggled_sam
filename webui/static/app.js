/* Muggled SAM - Web UI client
 *
 * Mirrors the interactive workflow of save_prompts_run_video.py:
 *  - Open (file browser on the server) replaces --input_video
 *  - Save button saves ./tracking_states/<video_stem>.pt
 *  - Tools: Hover / Box / FG Point / BG Point / Clear (+ Crop when enabled)
 *  - Playback: play/pause (track), seek, step, reverse
 *  - Buffers, recording, text prompts (SAM3), preview/invert toggles
 */
"use strict";

// ------------------------------------------------------------------ state

const state = {
  open: false,
  view: "authoring",   // "authoring" | "segmentation" (both views stay mounted)
  cfg: null,
  model: { name: "-", version: null, device: "-", dtype: "-", tokens: "-" },
  video: { path: null, fps: 30, total_frames: 0, hw: null, use_webcam: false },
  frameIdx: 0,
  isPaused: true,
  isReversed: false,
  endReached: false,
  buffer: 0,
  maskIdx: 1,
  numBuffers: 4,
  score: 0,
  playing: false,
  playTimer: null,
  busy: false,

  // prompt editing (client-side mirror; server is authoritative)
  tool: "hover",
  prompts: { boxes: [], fg: [], bg: [] },
  promptHistory: [],  // placement order of working prompts: {kind: 'box'|'fg'|'bg'}
  boxDrag: null,      // {x0, y0, x1, y1} normalized, in progress
  cropRect: null,     // committed crop rect (normalized) while crop tool active
  cropDrag: null,
  hoverPos: null,

  zoom: 1.0,
  status: null,
  seenLog: [],
  dirSelected: null,  // absolute path of selected entry (file or folder) in the Open dialog
  dirSelectedIsDir: null,
  segJobs: [],        // last segmentation queue snapshot (for button handlers)
  segRepairJobId: null, // job whose repair picker / repair preview is open
  // { jobId, a, b, clips: [{s, e, direction}] }: A/B draft points +
  // committed clips set on the preview bar
  segRepairPick: null,
  segRepairSelectedClipIdx: null, // committed clip selected for timeline looping
  // currentTime changes asynchronously.  Keep the requested frame separately
  // so held ArrowRight advances from the last request, not a stale decoded frame.
  segRepairTargetFrame: null,
  segRepairClipIdx: null,  // clip index shown in the repair preview dialog
  segRepairAccepting: false, // accept request in flight (client-side guard)
};

// ------------------------------------------------------------------ helpers

const $ = (id) => document.getElementById(id);

function api(path, { method = "GET", body = null } = {}) {
  const opts = { method };
  if (body !== null) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  return fetch(path, opts).then(async (res) => {
    const data = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
    if (data && data.error) {
      throw new Error(data.error);
    }
    return data;
  });
}

function showLoading(text) {
  $("loading-text").textContent = text || "Loading...";
  $("loading").style.display = "flex";
}

function hideLoading() {
  $("loading").style.display = "none";
}

let toastTimer = null;
function toast(msg, isErr = false) {
  const el = $("toast");
  el.textContent = msg;
  el.style.borderColor = isErr ? "var(--danger)" : "var(--accent2)";
  el.style.display = "block";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.style.display = "none"), 3500);
}

function appendLog(lines) {
  const logEl = $("log");
  if (!lines) return;
  const fresh = lines.filter((l) => !state.seenLog.includes(l));
  if (!fresh.length) return;
  state.seenLog = [...state.seenLog, ...fresh].slice(-60);
  logEl.textContent = state.seenLog.join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}

function applyHideInfo(hide) {
  const onSeg = state.view === "segmentation";
  const hideBars = onSeg || !!hide;
  $("header").style.display = hideBars ? "none" : "flex";
  $("footer").style.display = hideBars ? "none" : "flex";
}

// ------------------------------------------------------------------ rendering

const frameCanvas = $("frame-canvas");
const overlayCanvas = $("overlay-canvas");
const fctx = frameCanvas.getContext("2d");
const octx = overlayCanvas.getContext("2d");

function drawImageB64(canvas, b64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = () => resolve();
    img.src = "data:image/jpeg;base64," + b64;
  });
}

function drawImageB64Png(canvas, b64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      // size the canvas to the image so the mask keeps its aspect ratio
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = () => resolve();
    img.src = "data:image/png;base64," + b64;
  });
}

function drawOverlay() {
  const w = overlayCanvas.width;
  const h = overlayCanvas.height;
  octx.clearRect(0, 0, w, h);
  if (!state.open) return;

  const p = state.prompts;

  // boxes
  octx.lineWidth = 2;
  octx.strokeStyle = "#ffe14f";
  for (const [tl, br] of p.boxes) {
    octx.strokeRect(tl[0] * w, tl[1] * h, (br[0] - tl[0]) * w, (br[1] - tl[1]) * h);
  }
  // in-progress box
  if (state.boxDrag) {
    const { x0, y0, x1, y1 } = state.boxDrag;
    octx.setLineDash([6, 4]);
    octx.strokeRect(
      Math.min(x0, x1) * w, Math.min(y0, y1) * h,
      Math.abs(x1 - x0) * w, Math.abs(y1 - y0) * h
    );
    octx.setLineDash([]);
  }

  // crop rect
  if (state.cropRect) {
    const [[x1, y1], [x2, y2]] = state.cropRect;
    octx.strokeStyle = "#39c26d";
    octx.lineWidth = 2;
    octx.setLineDash([8, 4]);
    octx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
    octx.setLineDash([]);
  }
  if (state.cropDrag) {
    const { x0, y0, x1, y1 } = state.cropDrag;
    octx.strokeStyle = "#39c26d";
    octx.setLineDash([6, 4]);
    octx.strokeRect(
      Math.min(x0, x1) * w, Math.min(y0, y1) * h,
      Math.abs(x1 - x0) * w, Math.abs(y1 - y0) * h
    );
    octx.setLineDash([]);
  }

  // points
  const dot = (x, y, fill, ring) => {
    octx.beginPath();
    octx.arc(x * w, y * h, 5, 0, Math.PI * 2);
    octx.fillStyle = fill;
    octx.fill();
    octx.lineWidth = 2;
    octx.strokeStyle = ring;
    octx.stroke();
  };
  for (const [x, y] of p.fg) dot(x, y, "#39e06d", "#ffffff");
  for (const [x, y] of p.bg) dot(x, y, "#101010", "#39e06d");

  // hover crosshair
  if (state.tool === "hover" && state.hoverPos && !state.playing) {
    const [x, y] = state.hoverPos;
    octx.strokeStyle = "rgba(255,255,255,0.55)";
    octx.lineWidth = 1;
    octx.beginPath();
    octx.moveTo(x * w, 0); octx.lineTo(x * w, h);
    octx.moveTo(0, y * h); octx.lineTo(w, y * h);
    octx.stroke();
  }
}

function setPreviews(previewsB64) {
  if (!previewsB64 || previewsB64.length !== 4) return;
  for (let i = 0; i < 4; i++) {
    drawImageB64Png($("prev" + i), previewsB64[i]);
  }
  document.querySelectorAll(".preview-slot").forEach((el, i) => {
    el.classList.toggle("selected", i === state.maskIdx);
  });
}

function updateBufferButtons() {
  const row = $("buffer-row");
  if (row.childElementCount !== state.numBuffers) {
    row.innerHTML = "";
    for (let i = 0; i < state.numBuffers; i++) {
      const btn = document.createElement("button");
      btn.className = "tool";
      btn.style.minWidth = "56px";
      btn.onclick = () => selectBuffer(i);
      row.appendChild(btn);
    }
  }
  for (let i = 0; i < state.numBuffers; i++) {
    const btn = row.children[i];
    const info = state.status && state.status.buffers ? state.status.buffers[i] : null;
    const mb = info ? info.mb.toFixed(1) : "0.0";
    const has = info ? info.has_prompts : false;
    const selected = i === state.buffer;
    if (btn.dataset.sel !== String(selected)) {
      btn.classList.toggle("selected", selected);
      btn.dataset.sel = String(selected);
    }
    const label = `${i + 1}${has ? " ●" : ""}<br><small>${mb} MB</small>`;
    if (btn.innerHTML !== label) btn.innerHTML = label;
  }
}

function updateTextUI() {
  const st = state.status || {};
  const fs = $("text-fieldset");
  fs.style.display = st.model && st.model.version === "samv3" ? "" : "none";

  const cs = $("crop-fieldset");
  const ct = $("tool-crop");
  const showCrop = st.config ? st.config.crop : false;
  cs.style.display = showCrop ? "" : "none";
  ct.style.display = showCrop ? "" : "none";
  $("buffer-fieldset").style.display = st.config && st.config.disable_save ? "none" : "";

  // text status line
  const el = $("text-status");
  if (st.has_text_preview) {
    el.textContent = "Text preview active: select a candidate above, then click Store Prompt";
  } else if (st.text_draft !== undefined && st.text_draft) {
    el.textContent = `Text draft (Buffer ${state.buffer + 1}): ${st.text_draft}`;
  } else if (st.text_stored) {
    el.textContent = `Text prompt (Buffer ${state.buffer + 1}): ${st.text_stored}`;
  } else {
    el.textContent = "";
  }
}

function updateInfoFromDisplay(res) {
  state.open = true;
  // merge fresh per-buffer/text fields into the status snapshot so the UI
  // (text status line, etc.) doesn't wait for the next 2s status poll
  state.status = state.status || {};
  if (res.text_draft !== undefined) state.status.text_draft = res.text_draft;
  if (res.text_stored !== undefined) state.status.text_stored = res.text_stored;
  state.status.has_text_preview = !!res.text_preview;
  state.frameIdx = res.frame_idx;
  state.isPaused = res.is_paused;
  state.isReversed = res.is_reversed;
  state.endReached = res.end_reached;
  state.buffer = res.buffer;
  state.maskIdx = res.mask_idx;
  state.numBuffers = res.num_buffers;
  state.score = res.score;
  state.video.fps = res.fps || state.video.fps;
  state.video.total_frames = res.total_frames;
  if (res.prompts) {
    state.prompts = res.prompts;
    syncPromptHistory(res.prompts);
  }

  // Store Prompt only makes sense with working prompts or a live text candidate
  const hasStorablePrompt =
    (!!res.prompts && res.prompts.boxes.length + res.prompts.fg.length + res.prompts.bg.length > 0) ||
    state.status.has_text_preview;
  $("btn-store").disabled = !hasStorablePrompt;
  $("btn-undo-prompt").disabled = !(res.num_prompt_mems > 0);

  const total = state.video.total_frames;
  $("frame-text").textContent = total > 1 ? `Frame: ${res.frame_idx}/${total}` : "Frame: (live)";
  $("score-text").textContent = `Score: ${res.score.toFixed(1)}`;

  const slider = $("seek-slider");
  if (total > 1) {
    slider.disabled = false;
    slider.max = total - 1;
    slider.value = res.frame_idx;
  } else {
    slider.disabled = true;
  }

  $("btn-play").innerHTML = state.isPaused ? "&#x25B6;" : "&#x2758;&#x2758;";
  $("btn-reverse").classList.toggle("selected", state.isReversed);

  const banner = $("status-banner");
  if (state.video.use_webcam) {
    banner.textContent = "webcam";
  } else if (state.endReached && state.isPaused) {
    banner.textContent = "End of video - press Play to track from here, or seek back";
  } else if (!state.isPaused) {
    banner.textContent = "Tracking...";
  } else {
    banner.textContent = "";
  }
  overlayCanvas.style.pointerEvents = state.isPaused ? "auto" : "none";

  updateBufferButtons();
  updateTextUI();
  appendLog(res.log);
}

// Keep the 2x2 mask grid exactly the same size (and thus aspect ratio) as the video.
// Deterministic: width = min(video natural width, (available stage width - gap) / 2),
// where the stage now shrink-fits the video+grid so the control panel sits flush
// against the grid. The available width is measured on #main (not the row) so the
// layout always uses the maximum size that fits, without a feedback loop.
function syncMaskGridSize() {
  const main = $("main");
  const panel = $("panel");
  const canvasWrap = $("canvas-wrap");
  const previews = $("previews");
  if (!main || !panel || !canvasWrap || !previews) return;
  if (!state.open || frameCanvas.width < 2) {
    canvasWrap.style.width = "";
    previews.style.width = "";
    return;
  }
  const gap = 8;   // must match #video-row gap in style.css
  const mainPad = 20; // must match #main left+right padding in style.css
  const avail = main.clientWidth - panel.clientWidth - mainPad;
  const naturalW = frameCanvas.width; // buffer px == natural rendered width
  const w = Math.round(Math.min(naturalW, (avail - gap) / 2));
  canvasWrap.style.width = w + "px";
  previews.style.width = w + "px";
}

function renderDisplay(res) {
  drawImageB64(frameCanvas, res.frame_b64).then(() => {
    overlayCanvas.width = frameCanvas.width;
    overlayCanvas.height = frameCanvas.height;
    syncMaskGridSize();
    drawOverlay();
  });
  // updateInfoFromDisplay first: setPreviews toggles the selected slot
  // using state.maskIdx, which updateInfoFromDisplay refreshes
  updateInfoFromDisplay(res);
  setPreviews(res.previews_b64);
}

// ------------------------------------------------------------------ status poll

async function pollStatus() {
  try {
    const st = await api("/api/status");
    state.status = st;
    state.open = st.open;
    state.cfg = st.config;

    state.model = st.model;
    state.video = { ...state.video, ...st.video };
    state.isReversed = st.ui.is_reversed;
    state.isPaused = st.ui.is_paused;
    state.buffer = st.ui.buffer;
    state.maskIdx = st.ui.mask_idx;
    state.numBuffers = st.ui.num_buffers;
    state.score = st.score;

    // header
    $("hdr-model").textContent = st.model.name || "-";
    $("hdr-tokens").textContent = st.model.tokens ? `${st.model.tokens} tokens` : "";
    $("hdr-device").textContent = st.model.device ? `${st.model.device}/${st.model.dtype}` : "";
    $("hdr-video").textContent = st.video.use_webcam ? "webcam" : (st.video.path || "");

    // info bar
    $("vram-text").textContent = st.vram_mb != null ? `VRAM: ${st.vram_mb} MB` : "VRAM: -";
    $("prompts-text").textContent = `Prompts: ${st.num_prompt_mems}`;
    $("history-text").textContent = `History: ${st.num_prev_mems}`;
    if (!state.playing) $("score-text").textContent = `Score: ${st.score.toFixed(1)}`;

    // toggles mirror
    $("tgl-preview").checked = st.ui.show_preview;
    $("tgl-invert").checked = st.ui.invert_mask;
    $("tgl-history").checked = st.ui.enable_history;

    $("btn-close").disabled = !st.open;
    $("btn-undo-prompt").disabled = !(st.open && st.num_prompt_mems > 0);

    applyHideInfo(!!st.config.hide_info);
    updateTextUI();
    updateBufferButtons();
    appendLog(st.log);
  } catch (err) {
    /* server may be briefly busy (model load) - ignore */
  }
}

// ------------------------------------------------------------------ actions

async function withBusy(label, fn) {
  if (state.busy) return;
  state.busy = true;
  showLoading(label);
  try {
    await fn();
  } catch (err) {
    toast(err.message || String(err), true);
  } finally {
    state.busy = false;
    hideLoading();
  }
}

function syncPromptsUI() {
  // tool buttons
  ["hover", "box", "fg", "bg", "crop"].forEach((t) => {
    const btn = $("tool-" + t);
    if (btn) btn.classList.toggle("selected", state.tool === t);
  });
  const hints = {
    hover: "Click to place a foreground point (switches to FG tool)",
    box: "Click & drag to draw a bounding box",
    fg: "Click to add a foreground point",
    bg: "Click to add a background point",
    crop: "Drag a rectangle, then click Apply Crop",
  };
  $("tool-hint").textContent = hints[state.tool] || "";
}

function selectTool(tool) {
  state.tool = tool;
  syncPromptsUI();
  drawOverlay();
}

async function sendPrompts() {
  const res = await api("/api/set_prompts", {
    method: "POST",
    body: {
      boxes: state.prompts.boxes,
      fg: state.prompts.fg,
      bg: state.prompts.bg,
    },
  });
  renderDisplay(res);
}

function addPromptPoint(kind, x, y) {
  if (kind === "fg") state.prompts.fg.push([x, y]);
  else state.prompts.bg.push([x, y]);
  state.promptHistory.push({ kind: kind === "fg" ? "fg" : "bg" });
  updateUndoLastButton();
  sendPrompts().catch((e) => toast(e.message, true));
}

function undoLastPrompt() {
  const last = state.promptHistory.pop();
  if (!last) return;
  if (last.kind === "box") state.prompts.boxes.pop();
  else if (last.kind === "fg") state.prompts.fg.pop();
  else state.prompts.bg.pop();
  updateUndoLastButton();
  drawOverlay();
  sendPrompts().catch((e) => toast(e.message, true));
}

function updateUndoLastButton() {
  const btn = $("tool-undo");
  if (btn) btn.disabled = state.promptHistory.length === 0;
}

// Keep the client-side placement history consistent with the server's
// authoritative prompt set; reset it whenever the counts diverge (e.g. the
// server cleared working prompts after Store Prompt or a buffer switch).
function syncPromptHistory(serverPrompts) {
  const counts = { box: 0, fg: 0, bg: 0 };
  for (const h of state.promptHistory) counts[h.kind]++;
  const ok =
    serverPrompts.boxes.length === counts.box &&
    serverPrompts.fg.length === counts.fg &&
    serverPrompts.bg.length === counts.bg;
  if (!ok) state.promptHistory = [];
  updateUndoLastButton();
}

async function selectBuffer(idx) {
  const res = await api("/api/select_buffer", { method: "POST", body: { idx } });
  renderDisplay(res);
}

async function selectMask(idx) {
  const res = await api("/api/select_mask", { method: "POST", body: { idx } });
  renderDisplay(res);
}

async function storePrompt() {
  const res = await api("/api/store_prompt", { method: "POST", body: {} });
  renderDisplay(res);
  toast("Prompt stored");
}

async function undoPrompt() {
  const res = await api("/api/undo_prompt", { method: "POST", body: {} });
  renderDisplay(res);
  toast("Last stored prompt removed");
}

async function toggleUI(name) {
  const res = await api("/api/toggle", { method: "POST", body: { name } });
  renderDisplay(res);
}

// ------------------------------------------------------------------ playback

function playIntervalMs() {
  // The script paces playback with its 60fps display window (frame delay
  // max(16ms - elapsed, 1)), i.e. playback runs at the speed the model allows
  return 16;
}

function stopPlayback() {
  state.playing = false;
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
  $("btn-play").innerHTML = "&#x25B6;";
  overlayCanvas.style.pointerEvents = "auto";
  const banner = $("status-banner");
  if (state.endReached) {
    banner.textContent = "End of video - press Play to track from here, or seek back";
  } else {
    banner.textContent = "";
  }
}

async function startPlayback() {
  state.playing = true;
  $("btn-play").innerHTML = "&#x2758;&#x2758;";
  overlayCanvas.style.pointerEvents = "none";
  $("status-banner").textContent = "Tracking...";

  let last = performance.now();
  const tick = async () => {
    if (!state.playing) return;
    try {
      const res = await api("/api/frame?advance=1");
      renderDisplay(res);
      if (res.is_paused) stopPlayback();
      // catch up slightly if the model is slower than realtime
      const now = performance.now();
      state.playTimer = setTimeout(tick, Math.min(playIntervalMs(), now - last + 15));
      last = now;
    } catch (err) {
      stopPlayback();
      toast(err.message, true);
    }
  };
  state.playTimer = setTimeout(tick, playIntervalMs());
}

async function setPlay(playing) {
  const res = await api("/api/play", { method: "POST", body: { playing } });
  renderDisplay(res);
  if (res.is_paused) {
    stopPlayback();
    // QoL, like the script: when pausing, switch back from FG tool to hover
    if (state.tool === "fg") selectTool("hover");
  } else {
    startPlayback();
  }
}

async function seek(frame) {
  const res = await api("/api/seek", { method: "POST", body: { frame } });
  renderDisplay(res);
  if (res.is_paused) stopPlayback();
}

async function step(n) {
  const res = await api("/api/step", { method: "POST", body: { n } });
  renderDisplay(res);
}

async function setReverse(on) {
  const res = await api("/api/reverse", { method: "POST", body: { reversed: on } });
  renderDisplay(res);
}

// ------------------------------------------------------------------ canvas interaction

function normPos(evt) {
  const rect = overlayCanvas.getBoundingClientRect();
  const x = (evt.clientX - rect.left) / rect.width;
  const y = (evt.clientY - rect.top) / rect.height;
  return { x: Math.min(Math.max(x, 0), 1), y: Math.min(Math.max(y, 0), 1) };
}

let downPos = null;
let moved = false;

overlayCanvas.addEventListener("mousedown", (evt) => {
  if (!state.open || !state.isPaused || state.playing) return;
  const p = normPos(evt);
  downPos = p;
  moved = false;
  if (state.tool === "box") {
    state.boxDrag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    drawOverlay();
  } else if (state.tool === "crop") {
    state.cropDrag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    drawOverlay();
  }
});

overlayCanvas.addEventListener("mousemove", (evt) => {
  if (!state.open) return;
  const p = normPos(evt);
  if (downPos) {
    const dx = p.x - downPos.x;
    const dy = p.y - downPos.y;
    if (Math.hypot(dx, dy) > 0.005) moved = true;
  }
  if (state.boxDrag) {
    state.boxDrag.x1 = p.x;
    state.boxDrag.y1 = p.y;
    drawOverlay();
  } else if (state.cropDrag) {
    state.cropDrag.x1 = p.x;
    state.cropDrag.y1 = p.y;
    drawOverlay();
  } else if (state.tool === "hover") {
    state.hoverPos = [p.x, p.y];
    drawOverlay();
  }
});

window.addEventListener("mouseup", (evt) => {
  if (!downPos) return;
  const drag = state.boxDrag || state.cropDrag;
  const isDragTool = state.tool === "box" || state.tool === "crop";
  if (isDragTool && drag) {
    const w = Math.abs(drag.x1 - drag.x0);
    const h = Math.abs(drag.y1 - drag.y0);
    if (w > 0.01 && h > 0.01) {
      const tl = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1)];
      const br = [Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)];
      if (state.tool === "box") {
        state.prompts.boxes.push([tl, br]);
        state.promptHistory.push({ kind: "box" });
        state.boxDrag = null;
        sendPrompts().catch((e) => toast(e.message, true));
      } else {
        state.cropRect = [tl, br];
        state.cropDrag = null;
        drawOverlay();
      }
    } else {
      state.boxDrag = null;
      state.cropDrag = null;
      drawOverlay();
    }
    downPos = null;
    return;
  }
  downPos = null;
});

overlayCanvas.addEventListener("click", (evt) => {
  if (!state.open || !state.isPaused || state.playing) return;
  if (moved) return; // was a drag
  const p = normPos(evt);
  if (state.tool === "hover") {
    // Hover click acts as an FG point and switches to the FG tool (script behavior)
    addPromptPoint("fg", p.x, p.y);
    selectTool("fg");
  } else if (state.tool === "fg") {
    addPromptPoint("fg", p.x, p.y);
  } else if (state.tool === "bg") {
    addPromptPoint("bg", p.x, p.y);
  }
});

// ------------------------------------------------------------------ open dialog

async function loadDir(path) {
  const res = await api("/api/dirs?path=" + encodeURIComponent(path || ""));
  // entering a (new) directory starts with no selection
  state.dirSelected = null;
  $("dir-selected").textContent = "no file selected";
  const list = $("dir-list");
  list.innerHTML = "";

  const addEntry = (icon, name, isDir, fullPath) => {
    const div = document.createElement("div");
    div.className = "dir-entry" + (isDir ? " dir" : " file");
    div.innerHTML = `<span class="icon">${icon}</span>${name}`;
    // single click: select (folder or file); double click: open (folder or file)
    div.onclick = () => {
      state.dirSelected = fullPath;
      state.dirSelectedIsDir = isDir;
      list.querySelectorAll(".dir-entry.selected").forEach((el) => el.classList.remove("selected"));
      div.classList.add("selected");
      $("dir-selected").textContent = fullPath;
    };
    div.ondblclick = () => {
      if (isDir) loadDir(fullPath).catch((e) => toast(e.message, true));
      else doOpen();
    };
    list.appendChild(div);
  };

  if (res.parent) addEntry("↰", ".. (up)", true, res.parent);
  for (const e of res.entries) {
    const isDir = e.is_dir === true;
    addEntry(isDir ? "📁" : "🎬", e.name, isDir, res.path + "/" + e.name);
  }
  $("dir-input").value = res.path;
}

function openDialog() {
  $("open-dialog").style.display = "flex";
  state.dirSelected = null;
  state.dirSelectedIsDir = null;
  $("dir-selected").textContent = "no file selected";
  const startPath = (state.status && state.status.video.path) ? state.status.video.path : "";
  loadDir(startPath ? startPath.replace(/\/[^/]*$/, "") : "").catch((e) => toast(e.message, true));
}

function closeDialog() {
  $("open-dialog").style.display = "none";
}

async function doOpen() {
  // a folder is selected: expand it in place (double-click equivalent), no error
  if (state.dirSelected && state.dirSelectedIsDir) {
    loadDir(state.dirSelected).catch((e) => toast(e.message, true));
    return;
  }
  closeDialog();
  await withBusy("Opening video & loading model (this can take a while)...", async () => {
    let res;
    if ($("open-webcam").checked) {
      res = await api("/api/open", { method: "POST", body: { webcam: true } });
    } else if (state.dirSelected) {
      res = await api("/api/open", { method: "POST", body: { path: state.dirSelected } });
    } else {
      throw new Error("Select a video file first");
    }
    // refresh full state (model/video info), then fetch the first frame
    await pollStatus();
    const disp = await api("/api/frame");
    renderDisplay(disp);
    loadSettingsIntoUI();
    toast("Video opened");
  });
}

// ------------------------------------------------------------------ settings

function loadSettingsIntoUI() {
  if (!state.cfg) return;
  const c = state.cfg;
  $("cfg-model_path").value = c.model_path || "";
  $("cfg-image_path").value = c.image_path || "";
  $("cfg-device").value = c.device || "";
  $("cfg-display_size").value = c.display_size;
  $("cfg-base_size_px").value = c.base_size_px;
  $("cfg-num_buffers").value = c.num_buffers;
  $("cfg-max_memories").value = c.max_memories;
  $("cfg-max_pointers").value = c.max_pointers;
  $("cfg-objscore_threshold").value = c.objscore_threshold;
  $("cfg-bg_color_hex").value = c.bg_color_hex;
  $("cfg-ffmpeg").value = c.ffmpeg || "";
  $("cfg-use_float32").checked = !!c.use_float32;
  $("cfg-use_aspect_ratio").checked = !!c.use_aspect_ratio;
  $("cfg-keep_bad_objscores").checked = !!c.keep_bad_objscores;
  $("cfg-keep_history_on_new_prompts").checked = !!c.keep_history_on_new_prompts;
  $("cfg-hide_info").checked = !!c.hide_info;
  $("cfg-use_webcam").checked = !!c.use_webcam;
  $("cfg-disable_save").checked = !!c.disable_save;
  $("cfg-crop").checked = !!c.crop;
}

async function applySettings() {
  const body = {
    model_path: $("cfg-model_path").value || null,
    image_path: $("cfg-image_path").value || null,
    device: $("cfg-device").value || getDeviceDefault(),
    display_size: parseInt($("cfg-display_size").value, 10),
    base_size_px: parseInt($("cfg-base_size_px").value, 10),
    num_buffers: parseInt($("cfg-num_buffers").value, 10),
    max_memories: parseInt($("cfg-max_memories").value, 10),
    max_pointers: parseInt($("cfg-max_pointers").value, 10),
    objscore_threshold: parseFloat($("cfg-objscore_threshold").value),
    bg_color_hex: $("cfg-bg_color_hex").value || "ff00ff00",
    ffmpeg: $("cfg-ffmpeg").value || null,
    use_float32: $("cfg-use_float32").checked,
    use_aspect_ratio: $("cfg-use_aspect_ratio").checked,
    keep_bad_objscores: $("cfg-keep_bad_objscores").checked,
    keep_history_on_new_prompts: $("cfg-keep_history_on_new_prompts").checked,
    hide_info: $("cfg-hide_info").checked,
    use_webcam: $("cfg-use_webcam").checked,
    disable_save: $("cfg-disable_save").checked,
    crop: $("cfg-crop").checked,
  };
  const res = await api("/api/config", { method: "POST", body });
  state.cfg = res.config;
  applyHideInfo(!!res.config.hide_info);
  updateTextUI();
  if (res.need_reopen) {
    toast("Settings applied - re-Open the video to apply model/buffer changes");
  } else {
    toast("Settings applied");
  }
  pollStatus();
}

let cachedDeviceDefault = null;
function getDeviceDefault() {
  return cachedDeviceDefault || "cpu";
}

// ------------------------------------------------------------------ keyboard

const TOOLS = ["hover", "box", "fg", "bg"];

document.addEventListener("keydown", (evt) => {
  // The repair picker owns its playback, A/B and arrow keys while it is open. Let its
  // later listener handle them instead of changing authoring tools/buffers.
  if (state.segRepairPick && $("seg-preview-dialog").style.display !== "none" &&
      [" ", "ArrowLeft", "ArrowRight", "a", "A", "b", "B"].includes(evt.key)) return;
  const tag = (evt.target && evt.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

  const k = evt.key;
  if (k === " ") {
    evt.preventDefault();
    if (!state.open) return;
    setPlay(state.isPaused).catch((e) => toast(e.message, true));
  } else if (k === "ArrowLeft" || k === "ArrowRight") {
    evt.preventDefault();
    const i = TOOLS.indexOf(state.tool);
    const n = k === "ArrowRight" ? (i + 1) % TOOLS.length : (i - 1 + TOOLS.length) % TOOLS.length;
    selectTool(TOOLS[n]);
  } else if (k === "ArrowUp" || k === "ArrowDown") {
    evt.preventDefault();
    if (!state.open) return;
    const n = k === "ArrowDown" ? state.maskIdx + 1 : state.maskIdx - 1;
    if (n >= 0 && n <= 3) selectMask(n).catch((e) => toast(e.message, true));
  } else if (k === "p" || k === "P") {
    if (!state.open) return;
    toggleUI("preview").catch((e) => toast(e.message, true));
  } else if (k === "i" || k === "I") {
    if (!state.open) return;
    toggleUI("invert").catch((e) => toast(e.message, true));
  } else if (k === "r" || k === "R") {
    if (!state.open) return;
    setReverse(!state.isReversed).catch((e) => toast(e.message, true));
  } else if (k === "b" || k === "B" || k === "v" || k === "V") {
    if (!state.open) return;
    const n = k === "b" || k === "B" ? 1 : -1;
    const nb = (state.buffer + n + state.numBuffers) % state.numBuffers;
    selectBuffer(nb).catch((e) => toast(e.message, true));
  } else if (k === "Tab") {
    evt.preventDefault();
    if (!state.open) return;
    storePrompt().catch((e) => toast(e.message, true));
  } else if (k === "=" || k === "+") {
    state.zoom = Math.min(state.zoom + 0.1, 3);
    applyZoom();
  } else if (k === "-" || k === "_") {
    state.zoom = Math.max(state.zoom - 0.1, 0.4);
    applyZoom();
  } else if ((k === "z" || k === "Z") && (evt.ctrlKey || evt.metaKey)) {
    evt.preventDefault();
    if (state.open && state.promptHistory.length > 0) undoLastPrompt();
  } else if (k === "g" || k === "G") {
    saveState().catch((e) => toast(e.message, true));
  } else if (k === "q" || k === "Q") {
    closeVideo().catch((e) => toast(e.message, true));
  }
});

function applyZoom() {
  $("canvas-wrap").style.transformOrigin = "top left";
  $("canvas-wrap").style.transform = `scale(${state.zoom})`;
}

// ------------------------------------------------------------------ top-level actions

async function saveState() {
  if (!state.open) {
    toast("Open a video first", true);
    return;
  }
  await withBusy("Saving tracking state ...", async () => {
    const res = await api("/api/save_state", { method: "POST", body: {} });
    if (res.saved === false) {
      toast(res.message || "Nothing to save (no prompts stored)");
    } else {
      toast(`Saved: ${res.path} (buffers ${res.objects.map((i) => i + 1).join(", ")})`);
    }
  });
}

async function closeVideo() {
  stopPlayback();
  await withBusy("Closing video...", async () => {
    if (state.open) {
      // Like the script: quitting the UI saves the tracking state (when prompts
      // exist). Save-on-quit is best-effort and never blocks the close.
      try {
        const res = await api("/api/save_state", { method: "POST", body: {} });
        if (res.saved !== false) toast(`Saved: ${res.path}`);
      } catch (err) {
        /* save on quit is best-effort: never block the close */
      }
    }
    await api("/api/close", { method: "POST", body: {} });
    state.open = false;
    state.playing = false;
    state.promptHistory = [];
    updateUndoLastButton();
    fctx.clearRect(0, 0, frameCanvas.width, frameCanvas.height);
    octx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    for (let i = 0; i < 4; i++) {
      const prevCanvas = $("prev" + i);
      prevCanvas.getContext("2d").clearRect(0, 0, prevCanvas.width, prevCanvas.height);
    }
    $("status-banner").textContent = "";
    $("btn-store").disabled = true;
    $("btn-undo-prompt").disabled = true;
    syncMaskGridSize();
    pollStatus();
  });
}

async function saveBuffer() {
  await withBusy("Saving buffer...", async () => {
    const res = await api("/api/save_buffer", { method: "POST", body: {} });
    toast(`Buffer saved: ${res.path}`);
    pollStatus();
  });
}

// ------------------------------------------------------------------ upload video dir

// Keep in sync with VIDEO_EXTS in server.py
const UPLOAD_VIDEO_EXTS = new Set([
  ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg",
  ".wmv", ".flv", ".3gp", ".3g2", ".asf", ".ogv", ".ts", ".mxf",
]);

let uploadCtx = null;  // { cancelled: bool, xhr: XMLHttpRequest|null }

function isUploadVideo(name) {
  const dot = name.lastIndexOf(".");
  return dot > 0 && UPLOAD_VIDEO_EXTS.has(name.slice(dot).toLowerCase());
}

function fmtBytes(num) {
  if (num >= 1e9) return (num / 1e9).toFixed(2) + " GB";
  if (num >= 1e6) return (num / 1e6).toFixed(1) + " MB";
  if (num >= 1e3) return (num / 1e3).toFixed(1) + " KB";
  return num + " B";
}

function setUploadProgress(frac, statusText) {
  const f = Math.max(0, Math.min(1, frac));
  $("upload-progress-fill").style.width = (f * 100).toFixed(1) + "%";
  if (statusText) $("upload-status").textContent = statusText;
}

function uploadFile(dirName, relPath, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    uploadCtx.xhr = xhr;
    xhr.open("POST", "/api/upload_file");
    xhr.setRequestHeader("X-Upload-Dir", dirName);
    xhr.setRequestHeader("X-Upload-RelPath", relPath);
    xhr.responseType = "text";
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded); };
    xhr.onload = () => {
      uploadCtx.xhr = null;
      if (xhr.status >= 200 && xhr.status < 300) {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch (err) {}
        resolve(data);
      } else {
        let msg = "HTTP " + xhr.status;
        try { msg = JSON.parse(xhr.responseText).error || msg; } catch (err) {}
        reject(new Error(msg));
      }
    };
    xhr.onerror = () => { uploadCtx.xhr = null; reject(new Error("Network error during upload")); };
    xhr.onabort = () => { uploadCtx.xhr = null; reject(new Error("Upload cancelled")); };
    xhr.send(file);
  });
}

// Top level only: subfolders (and everything inside them) are ignored so
// huge nested trees never block the upload. Cancellable via uploadCtx.
async function walkUploadDir(handle, out) {
  let scanned = 0;
  for await (const entry of handle.values()) {
    if (uploadCtx && uploadCtx.cancelled) return;
    scanned += 1;
    if (entry.kind === "file" && isUploadVideo(entry.name)) {
      setUploadProgress(0, `Scanning ... ${scanned} entries: reading ${entry.name}`);
      out.push({ rel: entry.name, file: await entry.getFile() });
    }
    setUploadProgress(0, `Scanning ... ${scanned} top-level entries, ${out.length} video file(s) found`);
  }
}

// Modal: list files that already exist on the server; resolve with
// "overwrite" or "skip" (applied to all listed files).
function askConflict(existing) {
  return new Promise((resolve) => {
    const list = $("conflict-list");
    list.textContent = "";
    for (const rel of existing) {
      const div = document.createElement("div");
      div.textContent = rel;
      list.appendChild(div);
    }
    $("conflict-dialog").style.display = "flex";
    const done = (action) => {
      $("conflict-dialog").style.display = "none";
      $("btn-conflict-overwrite").onclick = null;
      $("btn-conflict-skip").onclick = null;
      resolve(action);
    };
    $("btn-conflict-overwrite").onclick = () => done("overwrite");
    $("btn-conflict-skip").onclick = () => done("skip");
  });
}

async function startUpload(dirName, items) {
  if (!items.length) {
    $("upload-dialog").style.display = "none";
    uploadCtx = null;
    toast("No video files found in the selected folder", true);
    $("btn-upload-dir").disabled = false;
    return;
  }
  uploadCtx = { cancelled: false, xhr: null };
  $("btn-upload-dir").disabled = true;
  $("upload-dialog").style.display = "flex";
  $("btn-upload-cancel").disabled = false;
  $("btn-upload-close").disabled = true;
  $("upload-dest").textContent = `-> videos/${dirName}`;
  let skippedCount = 0;
  try {
    // Ask the server which of these files already exist and let the user
    // choose overwrite vs. skip before anything is written.
    setUploadProgress(0, `Checking ${items.length} file(s) for existing files ...`);
    let toUpload = items;
    const check = await api("/api/upload_check",
      { method: "POST", body: { name: dirName, files: items.map((it) => it.rel) } });
    if (check.existing && check.existing.length) {
      const action = await askConflict(check.existing);
      if (action === "skip") {
        const skipSet = new Set(check.existing);
        skippedCount = check.existing.length;
        toUpload = items.filter((it) => !skipSet.has(it.rel));
      }
    }
    if (uploadCtx.cancelled) {  // Cancel clicked while checking
      setUploadProgress(0, "Cancelled (nothing was uploaded)");
      return;
    }
    if (!toUpload.length) {
      setUploadProgress(1, `Skipped ${skippedCount} existing file(s) - nothing uploaded`);
      return;
    }
    const totalBytes = toUpload.reduce((sum, it) => sum + it.file.size, 0);
    setUploadProgress(0, `Preparing ${toUpload.length} video file(s), ${fmtBytes(totalBytes)} ...`);
    await api("/api/upload_dir", { method: "POST", body: { name: dirName } });
    let doneBytes = 0;
    for (let i = 0; i < toUpload.length; i++) {
      const it = toUpload[i];
      const before = doneBytes;
      await uploadFile(dirName, it.rel, it.file, (loaded) =>
        setUploadProgress((before + loaded) / totalBytes, `File ${i + 1}/${toUpload.length}: ${it.rel}`));
      doneBytes += it.file.size;
      setUploadProgress(doneBytes / totalBytes,
        `File ${i + 1}/${toUpload.length} done (${fmtBytes(doneBytes)}/${fmtBytes(totalBytes)})`);
    }
    setUploadProgress(1,
      `Done: ${toUpload.length} uploaded, ${fmtBytes(doneBytes)}` +
      (skippedCount ? `, ${skippedCount} skipped (existing)` : "") +
      ` -> videos/${dirName}`);
  } catch (err) {
    setUploadProgress(0, uploadCtx.cancelled
      ? "Cancelled (incomplete files were not kept on the server)"
      : "Failed: " + err.message);
  } finally {
    // Runs on every exit path (done, all skipped, cancelled, error) so the
    // dialog can always be closed and the button re-enabled.
    $("btn-upload-cancel").disabled = true;
    $("btn-upload-close").disabled = false;
    $("btn-upload-dir").disabled = false;
  }
}

async function pickUploadDir() {
  const btn = $("btn-upload-dir");
  btn.disabled = true;
  try {
    if (typeof window.showDirectoryPicker === "function") {
      let root;
      try {
        root = await window.showDirectoryPicker();
      } catch (err) {
        if (err && err.name === "AbortError") {  // user closed the picker
          btn.disabled = false;
          return;
        }
        throw err;
      }
      // Show the progress dialog while scanning so a slow folder listing is
      // visible and can be cancelled instead of looking like a hang.
      uploadCtx = { cancelled: false, xhr: null };
      $("upload-dialog").style.display = "flex";
      $("btn-upload-cancel").disabled = false;
      $("btn-upload-close").disabled = true;
      $("upload-dest").textContent = `-> videos/${root.name}`;
      setUploadProgress(0, `Scanning ${root.name} (top level only) ...`);
      const items = [];
      await walkUploadDir(root, items);
      if (uploadCtx.cancelled) {
        $("upload-dialog").style.display = "none";
        uploadCtx = null;
        btn.disabled = false;
        return;
      }
      await startUpload(root.name, items);
    } else {
      // non-Chromium fallback: webkitdirectory file input. Note: the browser
      // itself enumerates the whole tree before the change event fires, so
      // subfolders cannot be skipped on this path.
      btn.disabled = false;
      $("upload-dir-input").click();
    }
  } catch (err) {
    btn.disabled = false;
    toast(err.message, true);
  }
}

// ------------------------------------------------------------------ tab switching
// Both views stay mounted in the DOM; switching only toggles visibility so
// each interface's state (loaded video, prompts, queue, form fields) survives.

function setView(view) {
  state.view = view;
  const isAuthoring = view === "authoring";
  $("view-authoring").classList.toggle("active", isAuthoring);
  $("view-segmentation").classList.toggle("active", !isAuthoring);
  $("tab-authoring").classList.toggle("active", isAuthoring);
  $("tab-segmentation").classList.toggle("active", !isAuthoring);
  applyHideInfo(state.cfg ? state.cfg.hide_info : false);
  if (isAuthoring) {
    // Re-measure the authoring layout now that it is visible again.
    requestAnimationFrame(() => {
      syncMaskGridSize();
      if (state.open) api("/api/frame").then(renderDisplay).catch(() => {});
    });
  }
}

// ------------------------------------------------------------------ video segmentation

function escHtml(str) {
  return String(str == null ? "" : str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function segStatusLabel(job) {
  if (job.status === "running") {
    if (job.phase === "loading") return "loading";
    if (job.phase === "saving") return "saving";
    return "running";
  }
  return job.status; // queued | done | error | cancelled
}

// 追加要求1: human-readable duration for the ETA display
function fmtDuration(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

// 追加要求2: the Preview button has 4 visual states:
//   none -> "Preview" (starts generation, or joins the queue)
//   queued -> 0% ring (waiting for the preview slot; FIFO)
//   generating -> circular progress ring + percent
//   ready -> play icon (opens the video dialog)
//   repair accepted -> "Regenerate" (refresh the full MP4 once, on demand)
const RING_CIRC = 50.265; // 2 * PI * 8
function segPreviewButtonHtml(job) {
  const id = escHtml(job.job_id);
  const st = job.preview_status || "none";
  if (st === "generating" || st === "queued") {
    const p = st === "queued" ? 0 : Math.max(0, Math.min(1, job.preview_progress || 0));
    const off = (RING_CIRC * (1 - p)).toFixed(2);
    const title = st === "queued"
      ? "Waiting in the preview queue (one preview at a time)..."
      : "Generating mask preview video...";
    return `<button class="seg-preview generating" data-id="${id}" disabled
      title="${title}">
      <svg class="seg-ring" viewBox="0 0 20 20" width="14" height="14" aria-hidden="true">
        <circle class="seg-ring-bg" cx="10" cy="10" r="8"></circle>
        <circle class="seg-ring-fg" cx="10" cy="10" r="8"
          stroke-dasharray="${RING_CIRC.toFixed(2)}" stroke-dashoffset="${off}"></circle>
      </svg>
      <span>${Math.round(p * 100)}%</span>
    </button>`;
  }
  if (job.preview_needs_regen) {
    return `<button class="seg-preview" data-id="${id}"
      title="Regenerate the full mask preview video with accepted repair PNGs">Regenerate</button>`;
  }
  if (st === "ready") {
    return `<button class="seg-preview ready" data-id="${id}"
      title="Play the generated mask preview video">&#x25B6;</button>`;
  }
  const title = job.has_tars
    ? "Generate the mask preview video (frames go to ./generated_mask_videos/<name>/)"
    : "Unavailable: this job has no tar results (saved as mp4 or deleted)";
  const extra = st === "error" ? " error" : "";
  return `<button class="seg-preview${extra}" data-id="${id}"
    ${job.has_tars ? "" : "disabled"} title="${title}">Preview</button>`;
}

// Frame repair for finished jobs (isolated flawed frames):
//   none/error -> "Repair" (opens the frame-range + direction dialog)
//   running    -> progress ring
//   ready      -> play the repair preview / accept repair / discard + chip
//   accepted   -> static "repaired" chip (result files already rewritten)
function fmtRepairFrames(frames) {
  if (!frames || !frames.length) return "";
  const fs = frames.slice().sort((a, b) => a - b);
  const parts = [];
  let start = fs[0];
  let prev = fs[0];
  for (let i = 1; i < fs.length; i++) {
    if (fs[i] === prev + 1) { prev = fs[i]; continue; }
    parts.push(start === prev ? String(start) : start + "-" + prev);
    start = prev = fs[i];
  }
  parts.push(start === prev ? String(start) : start + "-" + prev);
  return parts.join(", ");
}

// "130" or "130-135" from a clip entry's [s, e] range
function segClipRangeText(clip) {
  if (clip && Array.isArray(clip.range) && clip.range.length === 2) {
    return clip.range[0] === clip.range[1]
      ? String(clip.range[0])
      : `${clip.range[0]}-${clip.range[1]}`;
  }
  return "?";
}

function segRepairHtml(job) {
  const id = escHtml(job.job_id);
  const st = job.repair_status || "none";
  const clips = Array.isArray(job.repair_clips) ? job.repair_clips : [];
  const pending = clips.filter((c) => c.status === "pending");
  const accepted = clips.filter((c) => c.status === "accepted");
  const acceptedText = fmtRepairFrames(accepted.flatMap((c) => c.frames || []));
  const acceptedChip = accepted.length
    ? `<span class="seg-repair-chip ok"
        title="Frame repair applied to preview PNGs; regenerate the full preview video when ready: ${escHtml(acceptedText)}">repaired: ${escHtml(acceptedText)}</span>`
    : "";
  if (st === "running") {
    const p = Math.max(0, Math.min(1, job.repair_progress || 0));
    const off = (RING_CIRC * (1 - p)).toFixed(2);
    return `<button class="seg-repair running" data-id="${id}" disabled
      title="Re-tracking each clip from its seed frame's saved mask...">
      <svg class="seg-ring" viewBox="0 0 20 20" width="14" height="14" aria-hidden="true">
        <circle class="seg-ring-bg" cx="10" cy="10" r="8"></circle>
        <circle class="seg-ring-fg" cx="10" cy="10" r="8"
          stroke-dasharray="${RING_CIRC.toFixed(2)}" stroke-dashoffset="${off}"></circle>
      </svg>
      <span>Repair ${Math.round(p * 100)}%</span>
    </button>`;
  }
  if (st === "accepting") {
    const p = Math.max(0, Math.min(1, job.repair_progress || 0));
    const off = (RING_CIRC * (1 - p)).toFixed(2);
    const active = clips.find((c) => c.index === job.repair_active_clip)
      || pending[0] || clips[0];
    const rangeText = active ? segClipRangeText(active) : "";
    const phaseTitle = job.repair_phase
      ? escHtml(job.repair_phase)
      : "Applying the repair: the repaired PNG frames are being copied...";
    return `
      <button class="seg-repair running" data-id="${id}" disabled
        title="${phaseTitle}">
        <svg class="seg-ring" viewBox="0 0 20 20" width="14" height="14" aria-hidden="true">
          <circle class="seg-ring-bg" cx="10" cy="10" r="8"></circle>
          <circle class="seg-ring-fg" cx="10" cy="10" r="8"
            stroke-dasharray="${RING_CIRC.toFixed(2)}" stroke-dashoffset="${off}"></circle>
        </svg>
        <span>Applying ${Math.round(p * 100)}%</span>
      </button>
      <span class="seg-repair-chip" title="Repair clip in progress (result files are being rewritten): ${escHtml(rangeText)}">applying: ${escHtml(rangeText)}</span>`;
  }
  if (pending.length) {
    let html = pending.map((c) => `
      <button class="seg-repair-preview" data-id="${id}" data-clip="${c.index}"
        title="Play the repaired preview for frames ${escHtml(segClipRangeText(c))} - then Accept or Discard this clip in the dialog">&#x25B6; ${escHtml(segClipRangeText(c))}</button>`).join("");
    html += `<span class="seg-repair-chip"
      title="Pending repair clips (not applied yet): ${escHtml(pending.map(segClipRangeText).join(", "))}">pending: ${escHtml(pending.map(segClipRangeText).join(", "))}</span>`;
    html += acceptedChip;
    return html;
  }
  if (job.accepted || !job.has_tars) return acceptedChip;
  return `<button class="seg-repair-open" data-id="${id}"
    title="Repair isolated flawed frames: opens the preview video to set A/B points on the progress bar; each A/B pair becomes one repair clip that can be accepted or discarded separately">Repair</button>${acceptedChip}`;
}

function segJobHtml(job) {
  const cls = job.status; // queued|running|done|error|cancelled
  const pct = Math.max(0, Math.min(100, (job.progress || 0) * 100));
  let frames = "";
  if (job.total_frames > 0) {
    frames = `frame ${job.current_frame}/${job.total_frames}`;
  } else if (job.status === "running") {
    frames = job.phase === "loading" ? "loading\u2026" : "starting\u2026";
  }
  // 追加要求1: estimated remaining time for the running job
  if (job.status === "running" && job.eta_seconds != null && job.total_frames > 0) {
    frames += ` \u00b7 ~${fmtDuration(job.eta_seconds)} left`;
  }
  let resultsHtml = "";
  if (job.saved_paths && job.saved_paths.length) {
    resultsHtml = `<div class="seg-job-results">saved: ${job.saved_paths.map(escHtml).join(", ")}</div>`;
  }
  const errHtml = job.error ? `<div class="seg-job-err">${escHtml(job.error)}</div>` : "";
  let previewErrHtml = "";
  if (job.status === "done" && job.preview_status === "error" && job.preview_error) {
    previewErrHtml = `<div class="seg-job-err">preview: ${escHtml(job.preview_error)}</div>`;
  }
  let repairErrHtml = "";
  if (job.status === "done" && job.repair_error) {
    repairErrHtml = `<div class="seg-job-err">repair: ${escHtml(job.repair_error)}</div>`;
  }
  // 追加要求2: bottom-right actions for finished jobs
  let actionsHtml = "";
  if (job.status === "done") {
    const acceptTitle = job.accepted
      ? "Mask frames already moved to ./videos/<garment>/<name>/"
      : "Move the generated mask frames to ./videos/<garment>/<name>/";
    actionsHtml = `<div class="seg-job-actions">
      ${segPreviewButtonHtml(job)}
      ${segRepairHtml(job)}
      <button class="seg-accept" data-id="${escHtml(job.job_id)}"
        ${job.has_tars && !job.accepted ? "" : "disabled"} title="${acceptTitle}">Accept</button>
      <button class="seg-delete danger" data-id="${escHtml(job.job_id)}"
        title="Delete this job's result files (tars/mp4), generated mask frames and the preview video. Frames already accepted under ./videos/ are kept.">Delete</button>
    </div>`;
  }
  const canCancel = job.status === "queued" || job.status === "running";
  const cancelBtn = canCancel
    ? `<button class="seg-cancel danger" data-id="${escHtml(job.job_id)}">Cancel</button>` : "";
  // Cancelled task bars: a Delete button that clears the bar from the list
  const removeBtn = job.status === "cancelled"
    ? `<button class="seg-job-remove danger" data-id="${escHtml(job.job_id)}"
        title="Remove this cancelled task from the list">Delete</button>` : "";
  const statusLabel = segStatusLabel(job);
  const ptCount = (job.prompt_paths || []).length;
  const ptChip = ptCount > 1
    ? `<span class="seg-job-frames" title="${escHtml(job.prompt_paths.join("\n"))}">${ptCount} \u00d7 .pt merged</span>`
    : "";
  return `<div class="seg-job ${cls}">
    <div class="seg-job-head">
      <span class="seg-status ${cls}">${statusLabel}</span>
      <span class="seg-job-label" title="${escHtml(job.video_path)}">${escHtml(job.label)}</span>
      ${ptChip}
      <span class="seg-job-frames">${frames}</span>
      ${cancelBtn}
      ${removeBtn}
    </div>
    <div class="seg-prog"><div class="seg-prog-fill" style="width:${pct}%"></div></div>
    ${job.message ? `<div class="seg-job-msg">${escHtml(job.message)}</div>` : ""}
    ${errHtml}
    ${previewErrHtml}
    ${repairErrHtml}
    ${resultsHtml}
    ${actionsHtml}
  </div>`;
}

function renderSegQueue(data) {
  const list = $("seg-queue-list");
  const scrollTop = list ? list.scrollTop : 0;
  const jobs = (data && data.jobs) || [];
  const c = (data && data.counts) || {};
  state.segJobs = jobs;

  // Summary line in the queue legend
  const parts = [];
  if (c.queued) parts.push(c.queued + " queued");
  if (c.running) parts.push(c.running + " running");
  if (c.done) parts.push(c.done + " done");
  if (c.error) parts.push(c.error + " error");
  if (c.cancelled) parts.push(c.cancelled + " cancelled");
  $("seg-queue-summary").textContent = parts.length ? "— " + parts.join(", ") : "";

  // Tab badge: show when there is active work
  const activeCount = (c.queued || 0) + (c.running || 0);
  const badge = $("tab-seg-badge");
  if (badge) {
    if (activeCount > 0) {
      badge.style.display = "inline-block";
      badge.textContent = String(activeCount);
    } else {
      badge.style.display = "none";
    }
  }

  if (!list) return;
  if (!jobs.length) {
    list.innerHTML = '<div class="hint seg-empty">No tasks yet. Submit a task to begin.</div>';
    list.scrollTop = scrollTop;
    return;
  }

  const seenBatches = new Set();
  let html = "";
  for (const job of jobs) {
    if (job.batch_id && !seenBatches.has(job.batch_id)) {
      seenBatches.add(job.batch_id);
      const n = jobs.filter((j) => j.batch_id === job.batch_id).length;
      html += `<div class="seg-batch-head">
        <span class="seg-batch-name">📁 ${escHtml(job.batch_label || "folder")}</span>
        <span class="seg-batch-count">${n} video${n !== 1 ? "s" : ""}</span>
        <span class="spacer"></span>
      </div>`;
    }
    html += segJobHtml(job);
  }
  list.innerHTML = html;
  list.scrollTop = scrollTop;

  list.querySelectorAll(".seg-cancel").forEach((btn) => {
    btn.onclick = () => cancelSegJob(btn.dataset.id);
  });
  list.querySelectorAll(".seg-job-remove").forEach((btn) => {
    btn.onclick = () => removeCancelledSegJob(btn.dataset.id);
  });
  // 追加要求2: Preview (generate ring -> play), Accept, Delete
  list.querySelectorAll(".seg-preview").forEach((btn) => {
    btn.onclick = () => onPreviewClick(btn.dataset.id);
  });
  list.querySelectorAll(".seg-accept").forEach((btn) => {
    btn.onclick = () => acceptSegJob(btn.dataset.id);
  });
  list.querySelectorAll(".seg-repair-open").forEach((btn) => {
    btn.onclick = () => openSegRepairPicker(btn.dataset.id);
  });
  list.querySelectorAll(".seg-repair-preview").forEach((btn) => {
    btn.onclick = () =>
      openSegRepairPreview(btn.dataset.id, +(btn.dataset.clip || 0));
  });
  list.querySelectorAll(".seg-delete").forEach((btn) => {
    btn.onclick = () => deleteSegJob(btn.dataset.id);
  });
}

async function pollSegStatus() {
  try {
    const data = await api("/api/seg/queue");
    renderSegQueue(data);
  } catch (err) {
    /* server briefly busy - ignore */
  }
}

function readSegConfig() {
  return {
    model_path: $("seg-model_path").value.trim() || null,
    device: $("seg-device").value.trim() || null,
    base_size_px: parseInt($("seg-base_size_px").value, 10),
    num_buffers: parseInt($("seg-num_buffers").value, 10),
    bg_color_hex: $("seg-bg_color_hex").value.trim() || "ff00ff00",
    ffmpeg: $("seg-ffmpeg").value.trim() || null,
    use_aspect_ratio: $("seg-use_aspect_ratio").checked,
    use_float32: $("seg-use_float32").checked,
    pure_text: $("seg-pure_text").checked,
    pure_text_score_threshold: parseFloat($("seg-pt-score").value),
  };
}

async function submitSegTask() {
  const type = $("seg-type").value;
  const promptPaths = Array.from(document.querySelectorAll("#seg-prompt-list input[type=text]"))
    .map((el) => el.value.trim())
    .filter(Boolean);
  const videoPaths = Array.from(document.querySelectorAll("#seg-video-list input[type=text]"))
    .map((el) => el.value.trim())
    .filter(Boolean);
  const dir = $("seg-dir").value.trim();
  const hint = $("seg-submit-hint");
  hint.style.color = "var(--danger)";

  if (!promptPaths.length) { hint.textContent = "Pick a prompt (.pt) file"; return; }
  const body = {
    kind: type,
    prompt_path: promptPaths[0],
    prompt_paths: promptPaths,
    config: readSegConfig(),
  };
  if (type === "video") {
    if (!videoPaths.length) { hint.textContent = "Pick an input video"; return; }
    body.video_path = videoPaths[0];
    body.video_paths = videoPaths;
  } else {
    if (!dir) { hint.textContent = "Pick a videos folder"; return; }
    body.input_dir = dir;
  }
  hint.style.color = "var(--muted)";
  hint.textContent = "Submitting\u2026";
  try {
    await withBusy("Submitting task\u2026", async () => {
      const res = await api("/api/seg/submit", { method: "POST", body });
      const n = (res.jobs || []).length;
      toast(`Submitted ${n} job${n !== 1 ? "s" : ""} to the queue`);
      await pollSegStatus();
    });
    hint.style.color = "var(--muted)";
    hint.textContent = `Queued ${body.kind === "dir" ? "folder" : "video"}`;
  } catch (err) {
    hint.style.color = "var(--danger)";
    hint.textContent = err.message || String(err);
  }
}

async function cancelSegJob(jobId) {
  try {
    await api("/api/seg/cancel", { method: "POST", body: { job_id: jobId } });
    await pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  }
}

async function removeCancelledSegJob(jobId) {
  const ok = confirm(
    "Remove this cancelled task from the list?\n\n" +
    "This only clears the task bar (cancelled jobs save no results)."
  );
  if (!ok) return;
  try {
    await api("/api/seg/delete_job", { method: "POST", body: { job_id: jobId } });
    await pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  }
}

async function clearSegAll() {
  const ok = confirm(
    "Clear ALL generated segmentation artifacts?\n\n" +
    "Equivalent to clear_all.sh:\n" +
    "  - ./saved_images/run_video/   (task results)\n" +
    "  - ./generated_mask_videos/    (preview frames + preview videos)\n" +
    "  - ./videos/<group>/<name>/    (accepted mask frames)\n\n" +
    "Source .mp4 videos are kept. Finished task bars are removed; " +
    "queued/running tasks are not touched."
  );
  if (!ok) return;
  try {
    const res = await api("/api/seg/clear", { method: "POST", body: {} });
    const nJobs = (res && res.removed_jobs) || 0;
    const nPaths = (res && res.removed_paths || []).length;
    toast(`Cleared ${nJobs} finished task(s) and ${nPaths} artifact path(s)`);
    await pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  }
}

// ---- 追加要求2: preview / accept / delete for finished jobs ----

function segJobById(jobId) {
  return (state.segJobs || []).find((j) => j.job_id === jobId) || null;
}

async function onPreviewClick(jobId) {
  const job = segJobById(jobId);
  if (!job) return;
  if (job.preview_status === "generating") return; // ring is visible; wait
  if (job.preview_status === "queued") return; // 0% ring: waiting for its turn
  if (job.preview_needs_regen) {
    try {
      const res = await api("/api/seg/preview", { method: "POST", body: { job_id: jobId } });
      toast(res.queued
        ? "Regeneration queued - it starts when the other preview finishes"
        : "Regenerating full mask preview video…");
      await pollSegStatus();
    } catch (err) {
      toast(err.message || String(err), true);
    }
    return;
  }
  if (job.preview_status === "ready") {
    openSegPreview(job);
    return;
  }
  // "none" (or retry after an error): start generation
  try {
    const res = await api("/api/seg/preview", { method: "POST", body: { job_id: jobId } });
    toast(res.queued
      ? "Preview queued - it starts when the other preview finishes"
      : "Generating mask preview video\u2026");
    await pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  }
}

function openSegPreview(job) {
  const video = $("seg-preview-video");
  $("seg-preview-title").textContent =
    `Mask preview \u00b7 ${job.label || job.job_id}`;
  video.src = job.preview_url;
  $("seg-repair-pick").style.display = "none";
  $("seg-repair-actions").style.display = "none";
  $("seg-preview-dialog").style.display = "flex";
  video.play().catch(() => { /* user can press play manually */ });
}

function closeSegPreview() {
  const video = $("seg-preview-video");
  video.pause();
  video.removeAttribute("src");
  video.load(); // stop downloading
  $("seg-repair-pick").style.display = "none";
  $("seg-repair-timeline").style.display = "none";
  $("seg-repair-clips").style.display = "none";
  $("seg-repair-actions").style.display = "none";
  $("seg-preview-dialog").style.display = "none";
  state.segRepairPick = null;
  state.segRepairSelectedClipIdx = null;
  state.segRepairTargetFrame = null;
  state.segRepairClipIdx = null;
}

async function acceptSegJob(jobId) {
  const ok = confirm(
    "Move this job's generated mask frames to ./videos/<garment>/<name>/ ?"
  );
  if (!ok) return;
  try {
    const res = await api("/api/seg/accept", { method: "POST", body: { job_id: jobId } });
    toast((res && res.job && res.job.message) || "Mask frames accepted");
    await pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  }
}

async function deleteSegJob(jobId) {
  const ok = confirm(
    "Delete this job's result files (tars/mp4), generated mask frames and the " +
    "preview video?\n\nFrames already accepted under ./videos/ will be kept."
  );
  if (!ok) return;
  try {
    await api("/api/seg/delete", { method: "POST", body: { job_id: jobId } });
    toast("Result files and task removed");
    await pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  }
}

// ---- frame repair (isolated flawed frames) ----
function findSegJob(jobId) {
  return (state.segJobs || []).find((j) => j.job_id === jobId) || null;
}

// Frame repair picker: set A/B points on the preview video's progress bar
// (A=B = single frame) instead of typing frame numbers. The preview mp4 is
// encoded at 30 fps with exactly one frame per mask frame, so the mp4
// frame at time t maps 1:1 to mask frame index floor(t * 30).
function openSegRepairPicker(jobId) {
  const job = findSegJob(jobId);
  if (!job) return;
  if (!job.preview_url) {
    toast("Preview is not ready yet - press Repair again when it is ready");
    onPreviewClick(jobId);
    return;
  }
  state.segRepairJobId = jobId;
  state.segRepairClipIdx = null;
  state.segRepairPick = { jobId, a: null, b: null, clips: [] };
  state.segRepairSelectedClipIdx = null;
  state.segRepairTargetFrame = 0;
  const video = $("seg-preview-video");
  $("seg-preview-title").textContent =
    `Repair range \u00b7 ${job.label || job.job_id}`;
  video.src = job.preview_url;
  $("seg-repair-pick").style.display = "flex";
  $("seg-repair-timeline").style.display = "flex";
  $("seg-repair-clips").style.display = "flex";
  $("seg-repair-actions").style.display = "none";
  $("seg-preview-dialog").style.display = "flex";
  renderSegRepairClips();
  updateSegRepairPickUi();
  video.play().catch(() => { /* user can press play manually */ });
}

function segRepairCursorFrame() {
  const pick = state.segRepairPick;
  const job = pick ? findSegJob(pick.jobId) : null;
  const total = job && job.total_frames > 0 ? job.total_frames : 0;
  const frame = Math.floor($("seg-preview-video").currentTime * 30 + 1e-6);
  if (total <= 0) return frame;
  return Math.max(0, Math.min(frame, total - 1));
}

function setSegRepairFrame(frame, { pause = true } = {}) {
  const pick = state.segRepairPick;
  const job = pick ? findSegJob(pick.jobId) : null;
  const total = job && job.total_frames > 0 ? job.total_frames : 0;
  if (total <= 0) return;
  const safeFrame = Math.max(0, Math.min(Math.round(frame), total - 1));
  const video = $("seg-preview-video");
  if (pause) video.pause();
  // Assign on every range `input` event.  Do not wait for `change` (which
  // only fires after the thumb is released): browsers then decode and paint
  // the target frame continuously while the user drags.
  state.segRepairTargetFrame = safeFrame;
  // Seek to the centre of the frame interval. Seeking on an exact frame
  // boundary can be rounded to its predecessor by the browser's decoder,
  // producing apparent repeated frames or occasional two-frame jumps.
  video.currentTime = (safeFrame + 0.5) / 30;
  $("seg-repair-scrubber").value = String(safeFrame);
  $("seg-repair-cursor").textContent = "frame " + safeFrame;
  $("seg-repair-timeline-label").textContent = `frame ${safeFrame} / ${total - 1}`;
}

function selectSegRepairClip(index, { play = true } = {}) {
  const pick = state.segRepairPick;
  const clip = pick && pick.clips[index];
  if (!clip) return;
  state.segRepairSelectedClipIdx = index;
  setSegRepairFrame(clip.s);
  renderSegRepairClips();
  renderSegRepairTimeline();
  if (play) $("seg-preview-video").play().catch(() => {});
}

function renderSegRepairTimeline() {
  const scrubber = $("seg-repair-scrubber");
  const markers = $("seg-repair-markers");
  const pick = state.segRepairPick;
  const job = pick ? findSegJob(pick.jobId) : null;
  const total = job && job.total_frames > 0 ? job.total_frames : 0;
  if (!pick || total <= 0) {
    scrubber.max = "0";
    scrubber.value = "0";
    scrubber.disabled = true;
    markers.innerHTML = "";
    return;
  }
  scrubber.disabled = false;
  scrubber.max = String(total - 1);
  // A/B and keyboard adjustments can be one seek ahead of the decoder.
  // While paused, preserve that explicitly selected target instead of
  // snapping the progress thumb back to the last decoded frame (often A
  // immediately after Set B).
  const video = $("seg-preview-video");
  const displayFrame = video.paused && Number.isInteger(state.segRepairTargetFrame)
    ? state.segRepairTargetFrame
    : segRepairCursorFrame();
  scrubber.value = String(displayFrame);
  const span = Math.max(total - 1, 1);
  const draftMarker = pick.a == null ? "" : (() => {
    const left = (pick.a / span) * 100;
    return `<span class="seg-repair-marker draft" style="left:${left}%"
      title="A: frame ${pick.a}" aria-label="A: frame ${pick.a}"></span>`;
  })();
  markers.innerHTML = draftMarker + pick.clips.map((clip, index) => {
    // Positions are expressed against the first/last frame endpoints so a
    // clip touching the last frame still ends inside the displayed track.
    const left = (clip.s / span) * 100;
    const width = Math.min(Math.max(((clip.e - clip.s + 1) / total) * 100, 0.6), 100 - left);
    const selected = index === state.segRepairSelectedClipIdx ? " selected" : "";
    const range = clip.s === clip.e ? `frame ${clip.s}` : `frames ${clip.s}-${clip.e}`;
    return `<button type="button" class="seg-repair-marker${selected}" data-clip-idx="${index}"
      style="left:${left}%;width:${width}%" title="Loop ${escHtml(range)}"></button>`;
  }).join("");
  markers.querySelectorAll(".seg-repair-marker").forEach((marker) => {
    marker.onclick = (evt) => {
      evt.preventDefault();
      selectSegRepairClip(+marker.dataset.clipIdx);
    };
  });
}

function setSegRepairPoint(which) {
  const pick = state.segRepairPick;
  if (!pick) return;
  // Read the visible preview frame. This also respects seeking performed with
  // the native video controls, which has no client-side target-frame update.
  const selectedFrame = segRepairCursorFrame();
  pick[which] = selectedFrame;
  state.segRepairTargetFrame = selectedFrame;
  // A committed clip only loops after the user explicitly clicks its chip or
  // timeline marker. Setting a new A/B range must not start a loop.
  if (which === "a") state.segRepairSelectedClipIdx = null;
  if (pick.a != null && pick.b != null) commitDraftClip();
  renderSegRepairClips();
  updateSegRepairPickUi();
}

// The draft A/B pair becomes a committed clip (B auto-commits the range;
// a lone A stays a draft = single-frame clip until Start).
function commitDraftClip() {
  const pick = state.segRepairPick;
  if (!pick || pick.a == null || pick.b == null) return;
  const job = findSegJob(pick.jobId);
  const total = job && job.total_frames > 0 ? job.total_frames : 0;
  const s = Math.min(pick.a, pick.b);
  const e = Math.max(pick.a, pick.b);
  pick.a = null;
  pick.b = null;
  const overlap = pick.clips.find((c) => !(e + 1 < c.s || s > c.e + 1));
  if (overlap) {
    toast(`Frames ${s}-${e} overlap or touch clip ${overlap.s}-${overlap.e}`, true);
    return;
  }
  let direction = "forward";
  if (total > 0 && s === 0) direction = "backward";
  pick.clips.push({ s, e, direction });
  // Do not select the new clip automatically: after Set B, Space should play
  // forward from B, not loop back to A.
  state.segRepairSelectedClipIdx = null;
}

// Render the committed-clip chips (each with a direction select + remove).
function renderSegRepairClips() {
  const box = $("seg-repair-clips");
  if (!box) return;
  const pick = state.segRepairPick;
  if (!pick) { box.innerHTML = ""; return; }
  const job = findSegJob(pick.jobId);
  const total = job && job.total_frames > 0 ? job.total_frames : 0;
  const items = pick.clips.map((c, i) => {
    const noFwd = total > 0 && c.s === 0; // no saved frame before the clip
    const noBwd = total > 0 && c.e === total - 1; // no saved frame after
    const dir = noFwd ? "backward" : noBwd ? "forward" : c.direction;
    c.direction = dir;
    const rangeText = c.s === c.e ? String(c.s) : `${c.s}\u2013${c.e}`;
    const selected = i === state.segRepairSelectedClipIdx ? " selected" : "";
    return `<span class="seg-repair-clip${selected}" data-clip-idx="${i}" title="Select and loop this clip">
      <b>${escHtml(rangeText)}</b>
      <select class="seg-clip-dir" data-clip-idx="${i}"
        title="Seed direction for this clip">
        <option value="forward" ${dir === "forward" ? "selected" : ""}
          ${noFwd ? "disabled" : ""}>forward</option>
        <option value="backward" ${dir === "backward" ? "selected" : ""}
          ${noBwd ? "disabled" : ""}>backward</option>
      </select>
      <button class="seg-clip-del" data-clip-idx="${i}"
        title="Remove this clip">&#x2715;</button>
    </span>`;
  }).join("");
  box.innerHTML = items;
  box.querySelectorAll(".seg-clip-dir").forEach((sel) => {
    sel.onchange = () => {
      const pk = state.segRepairPick;
      if (!pk) return;
      pk.clips[+sel.dataset.clipIdx].direction = sel.value;
    };
  });
  box.querySelectorAll(".seg-clip-del").forEach((btn) => {
    btn.onclick = (evt) => {
      evt.stopPropagation();
      const pk = state.segRepairPick;
      if (!pk) return;
      const deleted = +btn.dataset.clipIdx;
      pk.clips.splice(deleted, 1);
      if (state.segRepairSelectedClipIdx === deleted) state.segRepairSelectedClipIdx = null;
      else if (state.segRepairSelectedClipIdx > deleted) state.segRepairSelectedClipIdx -= 1;
      renderSegRepairClips();
      renderSegRepairTimeline();
      updateSegRepairPickUi();
    };
  });
  box.querySelectorAll(".seg-repair-clip").forEach((chip) => {
    chip.onclick = (evt) => {
      if (evt.target.tagName === "SELECT" || evt.target.tagName === "BUTTON") return;
      selectSegRepairClip(+chip.dataset.clipIdx);
    };
  });
}

function updateSegRepairPickUi() {
  const pick = state.segRepairPick;
  if (!pick) return;
  const job = findSegJob(pick.jobId);
  const total = job && job.total_frames > 0 ? job.total_frames : 0;
  let hint;
  if (pick.a == null) {
    hint =
      "Play/scrub to a flawed frame, then press Set A (key A) and Set B " +
      "(key B) to add a clip (A=B = single frame). Each A/B pair becomes " +
      "its own clip and can be accepted or discarded separately.";
  } else {
    const a = pick.a;
    const s = pick.b == null ? a : Math.min(a, pick.b);
    const e = pick.b == null ? a : Math.max(a, pick.b);
    const noFwd = total > 0 && s === 0;
    const noBwd = total > 0 && e === total - 1;
    if (pick.b == null) {
      hint = `Draft: frame ${a} (single frame - set B to widen the clip)`;
    } else {
      hint = `Draft: frames ${s}-${e} (${e - s + 1} frame${e - s ? "s" : ""})`;
    }
    if (noFwd)
      hint += " A is the first frame, so forward (previous-frame seed) is unavailable.";
    if (noBwd)
      hint += " B is the last frame, so backward (next-frame seed) is unavailable.";
  }
  if (pick.clips.length) {
    hint += ` ${pick.clips.length} clip${pick.clips.length > 1 ? "s" : ""} selected.`;
  }
  $("seg-repair-range-hint").textContent = hint;
  $("seg-repair-start").disabled = pick.a == null && pick.clips.length === 0;
  renderSegRepairTimeline();
}

async function submitSegRepair() {
  const pick = state.segRepairPick;
  if (!pick) return;
  const job = findSegJob(pick.jobId);
  const total = job && job.total_frames > 0 ? job.total_frames : 0;
  const clips = pick.clips.slice();
  if (pick.a != null) {
    // lone A draft = single-frame clip
    const a = pick.a;
    let direction = "forward";
    if (total > 0 && a === 0) direction = "backward";
    clips.push({ s: a, e: a, direction });
  }
  if (!clips.length) return;
  const label = clips
    .map((c) => (c.s === c.e ? String(c.s) : `${c.s}-${c.e}`))
    .join(", ");
  try {
    await api("/api/seg/repair", {
      method: "POST",
      body: {
        job_id: pick.jobId,
        clips: clips.map((c) => ({
          frames: c.s === c.e ? String(c.s) : `${c.s}-${c.e}`,
          direction: c.direction,
        })),
      },
    });
    closeSegPreview();
    state.segRepairPick = null;
    toast(`Repair started: ${clips.length} clip(s) - ${label}`);
    pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  }
}

// Reuses the mask-preview video dialog; the footer gains accept / discard.
function openSegRepairPreview(jobId, clipIndex) {
  const job = findSegJob(jobId);
  const clips = (job && Array.isArray(job.repair_clips)) ? job.repair_clips : [];
  const pending = clips.filter((c) => c.status === "pending" && c.url);
  if (!pending.length) return;
  const clip = pending.find((c) => c.index === clipIndex) || pending[0];
  state.segRepairJobId = jobId;
  state.segRepairClipIdx = clip.index;
  const applying = job.repair_status === "accepting";
  $("seg-repair-accept").style.display = applying ? "none" : "";
  $("seg-repair-discard").style.display = applying ? "none" : "";
  $("seg-repair-rewrite-tar").checked = false;
  $("seg-repair-tar-opt").style.display = applying ? "none" : "";
  // clip selector (hidden when there is only one pending clip)
  const sel = $("seg-repair-clip-sel");
  sel.innerHTML = pending
    .map((c) => `<option value="${c.index}" ${c.index === clip.index ? "selected" : ""}>` +
      `clip ${escHtml(segClipRangeText(c))}</option>`)
    .join("");
  sel.style.display = pending.length > 1 ? "" : "none";
  const rangeText = segClipRangeText(clip);
  const seedText = clip.direction === "backward"
    ? "next frame's mask, tracked backward"
    : "previous frame's mask, tracked forward";
  let clipText = "";
  if (Array.isArray(clip.clip) && clip.clip.length === 2) {
    clipText = ` Clip shows frames ${clip.clip[0]}-${clip.clip[1]}. `;
  }
  $("seg-repair-hint").textContent = applying
    ? "The repair is being applied (the result files are being rewritten) - progress is shown on the job row."
    : `Frames ${rangeText} re-tracked from the ${seedText}.` + clipText +
      "Accept rewrites the preview frames + mp4 for this clip; Discard keeps the original result.";
  const video = $("seg-preview-video");
  $("seg-preview-title").textContent =
    `Repair preview \u00b7 ${job.label || job.job_id} \u00b7 clip ${rangeText}`;
  video.src = clip.url;
  $("seg-repair-pick").style.display = "none";
  $("seg-repair-clips").style.display = "none";
  $("seg-repair-actions").style.display = "flex";
  $("seg-preview-dialog").style.display = "flex";
  video.play().catch(() => { /* user can press play manually */ });
}

async function acceptSegRepair(jobId, clipIndex, rewriteTar = false) {
  if (state.segRepairAccepting) return;
  const job = findSegJob(jobId);
  if (job && job.repair_status === "accepting") {
    toast("The repair is already being applied - progress shows on the job row");
    return;
  }
  const ok = confirm(
    "Apply this clip repair? Only the corresponding generated preview PNG " +
    "frames will be overwritten. Use Regenerate later to update the full preview video." +
    (rewriteTar
      ? " The result tar(s) will be fully re-copied too (slow)."
      : " The result tar(s) are left unchanged (faster).")
  );
  if (!ok) return;
  state.segRepairAccepting = true;
  try {
    const res = await api("/api/seg/repair/accept", {
      method: "POST",
      body: { job_id: jobId, clip: clipIndex, rewrite_tar: !!rewriteTar },
    });
    closeSegPreview();
    toast("Repair accepted - applying\u2026");
    pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  } finally {
    state.segRepairAccepting = false;
  }
}

async function discardSegRepair(jobId, clipIndex) {
  const ok = confirm(
    "Discard this clip repair? The original result files are kept as-is " +
    "(the other clips are not affected)."
  );
  if (!ok) return;
  try {
    await api("/api/seg/repair/discard", {
      method: "POST",
      body: { job_id: jobId, clip: clipIndex },
    });
    closeSegPreview();
    toast("Clip discarded - original results are kept");
    pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  }
}

// ---- segmentation file/folder browse ----
let segBrowseCtx = null; // { kind, targetId, currentPath, sel }

let segPromptRowSeq = 0;

function addSegPromptRow(value) {
  const list = $("seg-prompt-list");
  const row = document.createElement("div");
  row.className = "row";
  const input = document.createElement("input");
  input.type = "text";
  input.id = "seg-prompt-" + segPromptRowSeq++;
  input.placeholder = "server path to a tracking-state .pt file";
  input.value = value || "";
  const browse = document.createElement("button");
  browse.type = "button";
  browse.textContent = "Browse";
  browse.onclick = () => openSegBrowse("pt", input.id, input.value);
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "seg-row-remove";
  remove.title = "Remove this .pt file";
  remove.textContent = "\u2715";
  remove.onclick = () => {
    if (list.children.length > 1) {
      row.remove();
    } else {
      input.value = "";
      input.focus();
    }
  };
  row.appendChild(input);
  row.appendChild(browse);
  row.appendChild(remove);
  list.appendChild(row);
  return input;
}

let segVideoRowSeq = 0;

function addSegVideoRow(value) {
  const list = $("seg-video-list");
  const row = document.createElement("div");
  row.className = "row";
  const input = document.createElement("input");
  input.type = "text";
  input.id = "seg-video-" + segVideoRowSeq++;
  input.placeholder = "server path to a video file";
  input.value = value || "";
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "seg-row-remove";
  remove.title = "Remove this video";
  remove.textContent = "\u2715";
  remove.onclick = () => {
    if (list.children.length > 1) {
      row.remove();
    } else {
      input.value = "";
      input.focus();
    }
  };
  row.appendChild(input);
  row.appendChild(remove);
  list.appendChild(row);
  return input;
}

function addSegVideoSelection(paths) {
  const list = $("seg-video-list");
  const existing = new Set(
    Array.from(list.querySelectorAll("input[type=text]")).map((el) => el.value.trim())
  );
  let added = 0;
  for (const p of paths) {
    if (existing.has(p)) continue;
    addSegVideoRow(p);
    existing.add(p);
    added += 1;
  }
  if (added > 0) {
    toast(`Added ${added} video${added !== 1 ? "s" : ""} to the task`);
  }
}

function addSegPromptSelection(anchorInputId, paths) {
  const list = $("seg-prompt-list");
  const anchor = $(anchorInputId);
  const rows = Array.from(list.children);
  const anchorRow = anchor ? anchor.parentElement : null;
  const anchorIdx = anchorRow ? rows.indexOf(anchorRow) : -1;
  const existing = new Set(
    rows
      .map((r, i) => (i === anchorIdx ? "" : (r.querySelector("input") || {}).value || ""))
      .map((v) => v.trim())
      .filter(Boolean)
  );
  const chosen = (paths || []).filter((p) => !existing.has(p));
  if (chosen.length === 0) {
    toast("Those .pt files are already in the list", true);
    return;
  }
  if (anchor) anchor.value = chosen[0];
  let prev = anchorRow;
  for (let i = 1; i < chosen.length; i++) {
    const input = addSegPromptRow(chosen[i]);
    const row = input.parentElement;
    if (prev) list.insertBefore(row, prev.nextSibling);
    prev = row;
  }
  toast(`Selected ${chosen.length} .pt file${chosen.length !== 1 ? "s" : ""}`);
}

function openSegBrowse(kind, targetId, startPath) {
  segBrowseCtx = {
    kind, targetId,
    currentPath: (startPath && startPath.endsWith("/")) ? startPath :
      (startPath ? startPath.replace(/\/[^\/]*$/, "") : ""),
    sel: null,
    selSet: new Set(),   // multi-select (video kind): selected file paths
    anchor: null,        // index into ctx.files of the last plain/toggled file
    files: [],           // file entries in display order (for Shift ranges)
  };
  const titles = {
    pt: "Pick prompt file(s) (.pt)",
    video: "Pick video files",
    dir: "Pick a videos folder",
  };
  $("seg-browse-title").textContent = titles[kind] || "Browse";
  $("seg-browse-hint").textContent = kind === "dir"
    ? "Single click: select \u00b7 Double click: open folder / confirm"
    : "Single click: select \u00b7 Shift+click: select a range \u00b7 \u2318+click: toggle one file \u00b7 Double click: open folder";
  $("seg-browse-dialog").style.display = "flex";
  loadSegDir(kind, segBrowseCtx.currentPath).catch((e) => toast(e.message, true));
}

async function loadSegDir(kind, path) {
  const res = await api(
    "/api/seg/browse?path=" + encodeURIComponent(path || "") + "&kind=" + kind
  );
  const list = $("seg-browse-list");
  list.innerHTML = "";
  $("seg-browse-selected").textContent = "nothing selected";
  $("seg-browse-input").value = res.path;
  if (segBrowseCtx) segBrowseCtx.currentPath = res.path;
  if (segBrowseCtx) segBrowseCtx.sel = null;
  if (segBrowseCtx && kind !== "dir") {
    segBrowseCtx.selSet = new Set();
    segBrowseCtx.anchor = null;
    segBrowseCtx.files = [];
  }
  const multi = kind !== "dir" && !!segBrowseCtx;

  function refreshBrowseSelection() {
    if (!multi) return;
    list.querySelectorAll(".dir-entry.file.selected").forEach((el) => el.classList.remove("selected"));
    segBrowseCtx.files.forEach((f) => {
      if (segBrowseCtx.selSet.has(f.fullPath)) f.el.classList.add("selected");
    });
    const label = $("seg-browse-selected");
    const names = Array.from(segBrowseCtx.selSet);
    if (names.length > 0) {
      label.textContent = names.length + " file" + (names.length !== 1 ? "s" : "") + " selected";
      label.title = names.join("\n");
    } else {
      label.textContent = "nothing selected";
      label.title = "";
    }
  }

  const addEntry = (icon, name, isDir, fullPath) => {
    const div = document.createElement("div");
    div.className = "dir-entry" + (isDir ? " dir" : " file");
    div.innerHTML = `<span class="icon">${icon}</span>${escHtml(name)}`;
    if (isDir) {
      // Folders: single click only highlights (or picks, for the dir kind);
      // double click opens. In multi-select mode the file selection is
      // untouched.
      div.onclick = () => {
        list.querySelectorAll(".dir-entry.selected").forEach((el) => el.classList.remove("selected"));
        div.classList.add("selected");
        if (!multi) {
          segBrowseCtx.sel = fullPath;
          $("seg-browse-selected").textContent = fullPath;
          $("seg-browse-selected").title = "";
        } else {
          refreshBrowseSelection();
        }
      };
      div.ondblclick = () => {
        loadSegDir(kind, fullPath).catch((e) => toast(e.message, true));
      };
    } else if (multi) {
      const fileIndex = segBrowseCtx.files.length;
      segBrowseCtx.files.push({ fullPath, el: div });
      div.onclick = (evt) => {
        if (evt.shiftKey && segBrowseCtx.anchor !== null) {
          const lo = Math.min(segBrowseCtx.anchor, fileIndex);
          const hi = Math.max(segBrowseCtx.anchor, fileIndex);
          for (let i = lo; i <= hi; i++) {
            segBrowseCtx.selSet.add(segBrowseCtx.files[i].fullPath);
          }
        } else if (evt.metaKey || evt.ctrlKey) {
          if (segBrowseCtx.selSet.has(fullPath)) segBrowseCtx.selSet.delete(fullPath);
          else segBrowseCtx.selSet.add(fullPath);
          segBrowseCtx.anchor = fileIndex;
        } else {
          segBrowseCtx.selSet.clear();
          segBrowseCtx.selSet.add(fullPath);
          segBrowseCtx.anchor = fileIndex;
        }
        refreshBrowseSelection();
      };
      // In multi-select mode double-clicking a file does not confirm (it
      // would reset the selection to that single file); use Select instead.
    } else {
      // Single click only selects; double click confirms the file.
      div.onclick = () => {
        segBrowseCtx.sel = fullPath;
        list.querySelectorAll(".dir-entry.selected").forEach((el) => el.classList.remove("selected"));
        div.classList.add("selected");
        $("seg-browse-selected").textContent = fullPath;
      };
      div.ondblclick = () => {
        confirmSegBrowse();
      };
    }
    list.appendChild(div);
  };

  if (res.parent) addEntry("↰", ".. (up)", true, res.parent);
  for (const e of res.entries) {
    const isDir = e.is_dir === true;
    const icon = isDir ? "📁" : (kind === "pt" ? "📦" : "🎬");
    addEntry(icon, e.name, isDir, res.path + "/" + e.name);
  }
}

function confirmSegBrowse() {
  const ctx = segBrowseCtx;
  if (!ctx) return;
  if (ctx.kind === "video") {
    if (!ctx.selSet || ctx.selSet.size === 0) {
      toast("Select at least one video file", true);
      return;
    }
    addSegVideoSelection(Array.from(ctx.selSet));
    $("seg-browse-dialog").style.display = "none";
    ctx.selSet = new Set();
    return;
  }
  if (ctx.kind === "pt") {
    if (!ctx.selSet || ctx.selSet.size === 0) {
      toast("Select at least one .pt file", true);
      return;
    }
    addSegPromptSelection(ctx.targetId, Array.from(ctx.selSet));
    $("seg-browse-dialog").style.display = "none";
    ctx.selSet = new Set();
    return;
  }
  let val;
  if (ctx.kind === "dir") {
    val = ctx.sel || ctx.currentPath;
  } else {
    val = ctx.sel; // must be an explicit file selection
  }
  if (!val) { toast("Select an item first", true); return; }
  $(ctx.targetId).value = val;
  $("seg-browse-dialog").style.display = "none";
  ctx.sel = null;
}

function wireSeg() {
  // tabs
  $("tab-authoring").onclick = () => setView("authoring");
  $("tab-segmentation").onclick = () => setView("segmentation");

  // form
  $("seg-type").onchange = () => {
    const t = $("seg-type").value;
    $("seg-video-field").style.display = t === "video" ? "" : "none";
    $("seg-dir-field").style.display = t === "dir" ? "" : "none";
  };
  $("seg-submit").onclick = () => submitSegTask();
  addSegPromptRow();
  $("seg-prompt-add").onclick = () => addSegPromptRow();
  addSegVideoRow();
  $("seg-video-browse").onclick = () => {
    const first = document.querySelector("#seg-video-list input[type=text]");
    openSegBrowse("video", "seg-video-list", first ? first.value : "");
  };
  $("seg-video-add").onclick = () => addSegVideoRow();
  $("seg-dir-browse").onclick = () => openSegBrowse("dir", "seg-dir", $("seg-dir").value);
  $("seg-clear-all").onclick = () => clearSegAll();

  // browse dialog
  $("seg-browse-refresh").onclick = () => {
    if (segBrowseCtx)
      loadSegDir(segBrowseCtx.kind, $("seg-browse-input").value).catch((e) => toast(e.message, true));
  };
  $("seg-browse-input").addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" && segBrowseCtx)
      loadSegDir(segBrowseCtx.kind, $("seg-browse-input").value).catch((e) => toast(e.message, true));
  });
  $("seg-browse-cancel").onclick = () => {
    $("seg-browse-dialog").style.display = "none";
    segBrowseCtx = null;
  };
  $("seg-browse-ok").onclick = confirmSegBrowse;

  // 追加要求2: mask preview video dialog
  $("seg-preview-close").onclick = closeSegPreview;
  $("seg-preview-dialog").addEventListener("click", (evt) => {
    if (evt.target === $("seg-preview-dialog")) closeSegPreview();
  });

  // frame repair A/B picker + accept/discard actions in the video dialog
  $("seg-repair-set-a").onclick = () => setSegRepairPoint("a");
  $("seg-repair-set-b").onclick = () => setSegRepairPoint("b");
  $("seg-repair-start").onclick = submitSegRepair;
  $("seg-repair-scrubber").oninput = () => {
    // Scrubbing is intentionally a pause: it gives A/B placement a stable
    // frame, and selecting a marked clip is the explicit way to start a loop.
    // setSegRepairFrame seeks on `input`, not `change`, so the displayed
    // video follows the thumb.  Do not re-render the timeline here: before
    // the asynchronous video seek completes that would reset the thumb to
    // the previous frame and make the preview appear frozen.
    state.segRepairSelectedClipIdx = null;
    setSegRepairFrame(+$('seg-repair-scrubber').value);
    renderSegRepairClips();
  };
  $("seg-repair-clip-sel").onchange = () => {
    if (state.segRepairJobId)
      openSegRepairPreview(state.segRepairJobId, +$("seg-repair-clip-sel").value);
  };
  $("seg-repair-discard").onclick = () => {
    if (state.segRepairJobId && state.segRepairClipIdx != null)
      discardSegRepair(state.segRepairJobId, state.segRepairClipIdx);
  };
  $("seg-repair-accept").onclick = () => {
    if (state.segRepairJobId && state.segRepairClipIdx != null)
      acceptSegRepair(
        state.segRepairJobId, state.segRepairClipIdx,
        $("seg-repair-rewrite-tar").checked
      );
  };
  $("seg-preview-video").addEventListener("loadedmetadata", () => {
    if (state.segRepairPick) {
      renderSegRepairTimeline();
      setSegRepairFrame(segRepairCursorFrame(), { pause: false });
    }
  });
  $("seg-preview-video").addEventListener("timeupdate", () => {
    const pick = state.segRepairPick;
    if (!pick) return;
    const frame = segRepairCursorFrame();
    const selected = pick.clips[state.segRepairSelectedClipIdx];
    // A selected A/B range loops without changing the native video's source.
    if (selected && frame >= selected.e) {
      state.segRepairTargetFrame = selected.s;
      $("seg-preview-video").currentTime = (selected.s + 0.5) / 30;
      $("seg-preview-video").play().catch(() => {});
      return;
    }
    // Normal playback (including the native controls) owns the target. A
    // paused keyboard seek does not: its seeked event can arrive out of order.
    if (!$("seg-preview-video").paused) state.segRepairTargetFrame = frame;
    $("seg-repair-cursor").textContent = "frame " + frame;
    $("seg-repair-scrubber").value = String(frame);
    const job = findSegJob(pick.jobId);
    const total = job && job.total_frames > 0 ? job.total_frames : 0;
    $("seg-repair-timeline-label").textContent = total ? `frame ${frame} / ${total - 1}` : `frame ${frame}`;
  });
  $("seg-preview-video").addEventListener("seeked", () => {
    // `timeupdate` is deliberately throttled by browsers during seeks.
    // Update the picker as soon as the decoded frame is ready instead.
    const pick = state.segRepairPick;
    if (!pick) return;
    // Keep the requested frame authoritative here. With rapid key repeats,
    // browsers may emit a late `seeked` for a cancelled earlier seek; using
    // its decoded currentTime would undo a later right-arrow increment.
    const frame = Number.isInteger(state.segRepairTargetFrame)
      ? state.segRepairTargetFrame
      : segRepairCursorFrame();
    $("seg-repair-scrubber").value = String(frame);
    $("seg-repair-cursor").textContent = "frame " + frame;
    const job = findSegJob(pick.jobId);
    const total = job && job.total_frames > 0 ? job.total_frames : 0;
    $("seg-repair-timeline-label").textContent = total ? `frame ${frame} / ${total - 1}` : `frame ${frame}`;
  });
  document.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape") {
      if ($("seg-preview-dialog").style.display !== "none") {
        closeSegPreview();
        return;
      }
      return;
    }
    if (!state.segRepairPick) return;
    const tag = (evt.target && evt.target.tagName) || "";
    if (evt.key === " ") {
      evt.preventDefault();
      const video = $("seg-preview-video");
      if (video.paused) {
        // Resume from the explicitly selected B/current frame. A clip only
        // loops when it was explicitly selected by clicking its chip/marker.
        if (state.segRepairSelectedClipIdx == null &&
            Number.isInteger(state.segRepairTargetFrame)) {
          video.currentTime = (state.segRepairTargetFrame + 0.5) / 30;
        }
        video.play().catch(() => {});
      }
      else {
        video.pause();
        // `timeupdate` is throttled (often by several frames at 30 fps), so
        // synchronise from currentTime at the exact pause point. Otherwise
        // the first arrow key starts from an older target and appears to jump.
        const pausedFrame = segRepairCursorFrame();
        state.segRepairTargetFrame = pausedFrame;
        $("seg-repair-scrubber").value = String(pausedFrame);
        $("seg-repair-cursor").textContent = "frame " + pausedFrame;
        const job = findSegJob(state.segRepairPick.jobId);
        const total = job && job.total_frames > 0 ? job.total_frames : 0;
        $("seg-repair-timeline-label").textContent = total
          ? `frame ${pausedFrame} / ${total - 1}`
          : `frame ${pausedFrame}`;
      }
      return;
    }
    if (evt.key === "ArrowLeft" || evt.key === "ArrowRight") {
      evt.preventDefault();
      state.segRepairSelectedClipIdx = null;
      const video = $("seg-preview-video");
      const baseFrame = !video.paused
        ? segRepairCursorFrame()
        : Number.isInteger(state.segRepairTargetFrame)
        ? state.segRepairTargetFrame
        : segRepairCursorFrame();
      setSegRepairFrame(baseFrame + (evt.key === "ArrowRight" ? 1 : -1));
      renderSegRepairClips();
      renderSegRepairTimeline();
      return;
    }
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (evt.key === "a" || evt.key === "A") setSegRepairPoint("a");
    else if (evt.key === "b" || evt.key === "B") setSegRepairPoint("b");
  });
}

// ------------------------------------------------------------------ wiring

function wire() {
  // Chrome only exposes showDirectoryPicker (fast, top-level-only picker) in
  // a secure context. Over plain http from another machine the webkitdirectory
  // fallback is used instead, and the browser scans every subfolder - warn.
  if (typeof window.showDirectoryPicker !== "function") {
    $("picker-warn").style.display = "block";
    $("btn-upload-dir").title =
      "Legacy folder picker active (plain http): the browser scans ALL subfolders, " +
      "which is slow and not cancellable for large trees. Restart the server with " +
      "--https and open https://<ip>:8765 (accept the cert warning) for the fast picker.";
  }
  $("btn-open").onclick = openDialog;
  $("btn-save").onclick = saveState;
  $("btn-close").onclick = closeVideo;
  $("btn-upload-dir").onclick = () => pickUploadDir();
  $("upload-dir-input").addEventListener("change", (evt) => {
    const files = Array.from(evt.target.files || []);
    evt.target.value = "";
    if (!files.length) return;
    const dirName = files[0].webkitRelativePath.split("/")[0];
    const items = files
      .filter((f) => f.webkitRelativePath === dirName + "/" + f.name && isUploadVideo(f.name))
      .map((f) => ({ rel: f.name, file: f }));
    startUpload(dirName, items).catch((e) => toast(e.message, true));
  });
  $("btn-upload-cancel").onclick = () => {
    if (!uploadCtx || uploadCtx.cancelled) return;
    uploadCtx.cancelled = true;
    if (uploadCtx.xhr) uploadCtx.xhr.abort();
  };
  $("btn-upload-close").onclick = () => {
    $("upload-dialog").style.display = "none";
    uploadCtx = null;
  };

  // playback
  $("btn-play").onclick = () => {
    if (!state.open) return;
    setPlay(state.isPaused).catch((e) => toast(e.message, true));
  };
  $("btn-step-back").onclick = () => step(-1).catch((e) => toast(e.message, true));
  $("btn-step-fwd").onclick = () => step(1).catch((e) => toast(e.message, true));
  $("btn-seek0").onclick = () => seek(0).catch((e) => toast(e.message, true));
  $("btn-sekend").onclick = () =>
    seek(state.video.total_frames - 1).catch((e) => toast(e.message, true));
  $("btn-reverse").onclick = () =>
    setReverse(!state.isReversed).catch((e) => toast(e.message, true));
  $("seek-slider").addEventListener("change", (evt) =>
    seek(parseInt(evt.target.value, 10)).catch((e) => toast(e.message, true))
  );

  // tools
  for (const t of ["hover", "box", "fg", "bg", "crop"]) {
    const btn = $("tool-" + t);
    if (btn) btn.onclick = () => selectTool(t);
  }
  $("tool-undo").onclick = () => {
    if (state.promptHistory.length > 0) undoLastPrompt();
  };
  $("tool-clear").onclick = () => {
    state.prompts = { boxes: [], fg: [], bg: [] };
    state.promptHistory = [];
    updateUndoLastButton();
    sendPrompts().catch((e) => toast(e.message, true));
  };
  updateUndoLastButton();

  // prompt actions
  $("btn-store").onclick = () => storePrompt().catch((e) => toast(e.message, true));
  $("btn-undo-prompt").onclick = () => undoPrompt().catch((e) => toast(e.message, true));
  $("btn-clear-prompts").onclick = () =>
    api("/api/clear_prompts", { method: "POST", body: {} }).then(renderDisplay)
      .catch((e) => toast(e.message, true));
  $("btn-clear-history").onclick = () =>
    api("/api/clear_history", { method: "POST", body: {} }).then(renderDisplay)
      .catch((e) => toast(e.message, true));

  // toggles
  $("tgl-preview").onchange = (e) => toggleUI("preview").catch((err) => { e.target.checked = !e.target.checked; toast(err.message, true); });
  $("tgl-invert").onchange = (e) => toggleUI("invert").catch((err) => { e.target.checked = !e.target.checked; toast(err.message, true); });
  $("tgl-history").onchange = (e) => toggleUI("history").catch((err) => { e.target.checked = !e.target.checked; toast(err.message, true); });

  // mask preview slots
  document.querySelectorAll(".preview-slot").forEach((el, i) => {
    el.onclick = () => selectMask(i).catch((e) => toast(e.message, true));
  });
  new ResizeObserver(() => syncMaskGridSize()).observe($("video-row"));

  // buffers
  $("btn-save-buffer").onclick = saveBuffer;
  $("btn-clear-buffer").onclick = () =>
    withBusy("Clearing buffer...", async () => {
      await api("/api/clear_buffer", { method: "POST", body: {} });
      pollStatus();
    }).catch((e) => toast(e.message, true));

  // text prompts
  $("btn-text-set").onclick = () =>
    withBusy("Running text detection...", async () => {
      const text = $("text-input").value.trim();
      if (!text) throw new Error("Type a text prompt first");
      const res = await api("/api/text_prompt", { method: "POST", body: { text } });
      renderDisplay(res);
    });
  $("btn-text-reuse").onclick = () =>
    withBusy("Reusing text prompt...", async () => {
      const res = await api("/api/reuse_text_prompt", { method: "POST", body: {} });
      renderDisplay(res);
    });

  // crop
  $("btn-crop-apply").onclick = () =>
    withBusy("Applying crop...", async () => {
      if (!state.cropRect) throw new Error("Draw a crop rectangle first (Crop tool)");
      const res = await api("/api/crop", { method: "POST", body: { tlbr: state.cropRect } });
      renderDisplay(res);
    });
  $("btn-crop-reset").onclick = () =>
    withBusy("Resetting crop...", async () => {
      state.cropRect = null;
      const res = await api("/api/crop", { method: "POST", body: { tlbr: [[0, 0], [1, 1]] } });
      renderDisplay(res);
    });

  // settings
  $("btn-settings-apply").onclick = () =>
    withBusy("Applying settings...", applySettings).catch((e) => toast(e.message, true));

  // open dialog
  $("btn-dir-refresh").onclick = () => {
    loadDir($("dir-input").value).catch((e) => toast(e.message, true));
  };
  $("dir-input").addEventListener("keydown", (evt) => {
    if (evt.key === "Enter") loadDir($("dir-input").value).catch((e) => toast(e.message, true));
  });
  $("btn-dir-cancel").onclick = closeDialog;
  $("btn-dir-open").onclick = doOpen;
}

// ------------------------------------------------------------------ init

async function init() {
  wire();
  wireSeg();
  setView("authoring");
  await pollStatus();
  pollSegStatus();
  setInterval(pollSegStatus, 1500);
  // If a video is already open (e.g. right after a browser refresh), fetch the
  // current frame + mask previews once so the view isn't left black until the
  // next user action
  if (state.open) {
    api("/api/frame").then(renderDisplay).catch(() => {
      /* server may be briefly busy (model load) - the 2s poll covers it */
    });
  }
  loadSettingsIntoUI();
  if (state.status && state.status.model) {
    cachedDeviceDefault = state.status.config.device;
  } else {
    cachedDeviceDefault = "cpu";
  }
  selectTool("hover");
  setInterval(pollStatus, 2000);
}

init();
