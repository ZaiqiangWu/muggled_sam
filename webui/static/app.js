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

function segJobHtml(job) {
  const cls = job.status; // queued|running|done|error|cancelled
  const pct = Math.max(0, Math.min(100, (job.progress || 0) * 100));
  let frames = "";
  if (job.total_frames > 0) {
    frames = `frame ${job.current_frame}/${job.total_frames}`;
  } else if (job.status === "running") {
    frames = job.phase === "loading" ? "loading\u2026" : "starting\u2026";
  }
  let resultsHtml = "";
  if (job.saved_paths && job.saved_paths.length) {
    resultsHtml = `<div class="seg-job-results">saved: ${job.saved_paths.map(escHtml).join(", ")}</div>`;
  }
  const errHtml = job.error ? `<div class="seg-job-err">${escHtml(job.error)}</div>` : "";
  const canCancel = job.status === "queued" || job.status === "running";
  const cancelBtn = canCancel
    ? `<button class="seg-cancel danger" data-id="${escHtml(job.job_id)}">Cancel</button>` : "";
  const statusLabel = segStatusLabel(job);
  return `<div class="seg-job ${cls}">
    <div class="seg-job-head">
      <span class="seg-status ${cls}">${statusLabel}</span>
      <span class="seg-job-label" title="${escHtml(job.video_path)}">${escHtml(job.label)}</span>
      <span class="seg-job-frames">${frames}</span>
      ${cancelBtn}
    </div>
    <div class="seg-prog"><div class="seg-prog-fill" style="width:${pct}%"></div></div>
    ${job.message ? `<div class="seg-job-msg">${escHtml(job.message)}</div>` : ""}
    ${errHtml}
    ${resultsHtml}
  </div>`;
}

function renderSegQueue(data) {
  const list = $("seg-queue-list");
  const scrollTop = list ? list.scrollTop : 0;
  const jobs = (data && data.jobs) || [];
  const c = (data && data.counts) || {};

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
  const prompt = $("seg-prompt").value.trim();
  const video = $("seg-video").value.trim();
  const dir = $("seg-dir").value.trim();
  const hint = $("seg-submit-hint");
  hint.style.color = "var(--danger)";

  if (!prompt) { hint.textContent = "Pick a prompt (.pt) file"; return; }
  const body = { kind: type, prompt_path: prompt, config: readSegConfig() };
  if (type === "video") {
    if (!video) { hint.textContent = "Pick an input video"; return; }
    body.video_path = video;
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

async function clearSegDone() {
  try {
    await api("/api/seg/clear", { method: "POST", body: {} });
    await pollSegStatus();
  } catch (err) {
    toast(err.message || String(err), true);
  }
}

// ---- segmentation file/folder browse ----
let segBrowseCtx = null; // { kind, targetId, currentPath, sel }

function openSegBrowse(kind, targetId, startPath) {
  segBrowseCtx = {
    kind, targetId,
    currentPath: (startPath && startPath.endsWith("/")) ? startPath :
      (startPath ? startPath.replace(/\/[^\/]*$/, "") : ""),
    sel: null,
  };
  const titles = {
    pt: "Pick a prompt file (.pt)",
    video: "Pick a video file",
    dir: "Pick a videos folder",
  };
  $("seg-browse-title").textContent = titles[kind] || "Browse";
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

  const addEntry = (icon, name, isDir, fullPath) => {
    const div = document.createElement("div");
    div.className = "dir-entry" + (isDir ? " dir" : " file");
    div.innerHTML = `<span class="icon">${icon}</span>${escHtml(name)}`;
    // Single click only selects (folders included); double click always
    // opens a folder. A folder is chosen only via single-click + Select.
    div.onclick = () => {
      segBrowseCtx.sel = fullPath;
      list.querySelectorAll(".dir-entry.selected").forEach((el) => el.classList.remove("selected"));
      div.classList.add("selected");
      $("seg-browse-selected").textContent = fullPath;
    };
    div.ondblclick = () => {
      if (isDir) {
        loadSegDir(kind, fullPath).catch((e) => toast(e.message, true));
      } else {
        confirmSegBrowse();
      }
    };
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
  $("seg-prompt-browse").onclick = () => openSegBrowse("pt", "seg-prompt", $("seg-prompt").value);
  $("seg-video-browse").onclick = () => openSegBrowse("video", "seg-video", $("seg-video").value);
  $("seg-dir-browse").onclick = () => openSegBrowse("dir", "seg-dir", $("seg-dir").value);
  $("seg-clear-done").onclick = () => clearSegDone();

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
