/* Muggled SAM - Web UI client
 *
 * Mirrors the interactive workflow of save_prompts_run_video.py:
 *  - Open (file browser on the server) replaces --input_video
 *  - Save button saves ./saved_tracking_state.pt
 *  - Tools: Hover / Box / FG Point / BG Point / Clear (+ Crop when enabled)
 *  - Playback: play/pause (track), seek, step, reverse
 *  - Buffers, recording, text prompts (SAM3), preview/invert toggles
 */
"use strict";

// ------------------------------------------------------------------ state

const state = {
  open: false,
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
  $("header").style.display = hide ? "none" : "flex";
  $("footer").style.display = hide ? "none" : "flex";
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
  if (res.prompts) state.prompts = res.prompts;

  // Store Prompt only makes sense with working prompts or a live text candidate
  const hasStorablePrompt =
    (!!res.prompts && res.prompts.boxes.length + res.prompts.fg.length + res.prompts.bg.length > 0) ||
    state.status.has_text_preview;
  $("btn-store").disabled = !hasStorablePrompt;

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
    $("tgl-record").checked = st.ui.is_record_enabled;

    $("btn-close").disabled = !st.open;

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
  sendPrompts().catch((e) => toast(e.message, true));
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
  await withBusy("Saving ./saved_tracking_state.pt ...", async () => {
    const res = await api("/api/save_state", { method: "POST", body: {} });
    toast(`Saved: ${res.path} (buffers ${res.objects.map((i) => i + 1).join(", ")})`);
  });
}

async function closeVideo() {
  stopPlayback();
  await withBusy("Closing video...", async () => {
    if (state.open) {
      // Like the script: quitting the UI saves the tracking state (when prompts exist)
      try {
        const res = await api("/api/save_state", { method: "POST", body: {} });
        toast(`Saved: ${res.path}`);
      } catch (err) {
        if (!/No prompts found/i.test(err.message)) throw err;
      }
    }
    await api("/api/close", { method: "POST", body: {} });
    state.open = false;
    state.playing = false;
    fctx.clearRect(0, 0, frameCanvas.width, frameCanvas.height);
    octx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    $("status-banner").textContent = "";
    $("btn-store").disabled = true;
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

// ------------------------------------------------------------------ wiring

function wire() {
  $("btn-open").onclick = openDialog;
  $("btn-save").onclick = saveState;
  $("btn-close").onclick = closeVideo;

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
  $("tool-clear").onclick = () => {
    state.prompts = { boxes: [], fg: [], bg: [] };
    sendPrompts().catch((e) => toast(e.message, true));
  };

  // prompt actions
  $("btn-store").onclick = () => storePrompt().catch((e) => toast(e.message, true));
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
  $("tgl-record").onchange = (e) => toggleUI("record").catch((err) => { e.target.checked = !e.target.checked; toast(err.message, true); });

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
  await pollStatus();
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
