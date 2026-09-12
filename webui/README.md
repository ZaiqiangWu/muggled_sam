# Muggled SAM Web UI

A single-page web app with two tabs (run from any browser on the network):

- **Prompt Authoring** — reproduces the functionality and parameters of
  `save_prompts_run_video.py` (interactive prompt labeling + save tracking state).
- **Video Segmentation** — submit queued segmentation jobs that reproduce
  `load_prompts_run_video.py` / `load_prompts_run_dir.py` (apply a saved `.pt`
  to one video, or to every `.mp4` in a folder), with live per-video progress.

Both tabs stay mounted, so switching between them never loses state (an open
video + prompts on the authoring side, submitted jobs + progress on the
segmentation side).

## Differences from the script (per `task.md`)

- **Open button** replaces `--input_video xxx.mp4`: it opens an interactive
  file browser for files **on the server** running this page (not a local
  browser upload). A "webcam" checkbox covers `-cam`.
- **Save button** saves `./tracking_states/<video_stem>.pt` (repo root; the
  folder is created if missing), e.g. `clip.mp4` -> `tracking_states/clip.pt`
  (webcam -> `tracking_states/webcam.pt`). In the script this happened
  automatically when the display window closed.

Everything else mirrors the script: the same model calls
(`encode_image` / `encode_prompts` / `generate_masks` /
`initialize_video_masking` / `step_video_masking`), the same per-frame
tracking/recording logic, the same buffer save (ffmpeg mp4 with tarfile
fallback), and all command-line parameters (see Settings panel mapping below).

## Running

The web layer is pure Python standard library, but the model code needs the
same dependencies as the script (torch, opencv, numpy). From the repository
root:

```bash
# activate an environment that has torch / cv2 / numpy, e.g.:
conda activate wu

python webui/server.py
# optional: python webui/server.py --port 8765 --host 0.0.0.0
```

Then open `http://<server-host>:8765/` in a browser.
The server binds `0.0.0.0` by default, so any IP on the network can connect.

`main()` does `os.chdir(repo_root)` before serving, so relative paths behave
exactly like in the script:

- model weights: `./model_weights/sam3.pt` (override with `Model path` in Settings)
- saved tracking state: `./tracking_states/<video_stem>.pt` (repo root, folder created on demand)
- saved buffer recordings: `./saved_images/run_video/...`
- uploaded video folders: `./videos/<picked folder name>/` (folder created on demand)

## Usage (mirrors the script's UI)

1. Click **Open**, browse the server filesystem, pick a video (or tick webcam),
   then click **Open** again. The model loads and the first frame is shown.
2. Draw prompts on the frame:
   - **Hover** tool: a click places a foreground point (then switches to FG,
     like the script's hover behavior);
   - **Box** tool: drag a bounding box;
   - **FG Point / BG Point**: add points;
   - **Undo Last** (`Ctrl/Cmd+Z`): undo the last placed box or point
     (repeatable; only affects working prompts, not stored ones);
   - **Clear**: wipe all working prompts.
   The 4 mask previews update live; select one with the preview buttons or
   `↑`/`↓`.
3. **Store Prompt** (`Tab`) stores the current prompts (or the selected text
   candidate) as a tracking prompt for the selected buffer.
4. **Play/Pause** (space, the script's Track button) starts/stops tracking.
   Playback runs at the speed the model allows (the script paces its display
   window at 60 fps). Tracking auto-pauses at the last frame.
5. **Buffers** (`v`/`b`): each buffer holds one tracked object. Switching
   buffers clears working prompts, exactly like the script.
   **Enable Recording** records the masked frames of tracked buffers into
   memory; **Save Buffer** writes them out (ffmpeg mp4 if an ffmpeg path is
   configured, otherwise a tarfile of PNGs), and **Clear Buffer** wipes them.
6. **Upload video dir** (right of Open/Save/Close): opens a local folder
   picker (folders only, Chromium `showDirectoryPicker`, `webkitdirectory`
   input fallback), filters the folder's contents to video files
   (recursively, same extension list as the Open dialog, non-video files are
   skipped), and uploads them — preserving the folder's internal structure —
   to `./videos/<picked folder name>/` on the server. A progress dialog shows
   per-file progress and supports Cancel (an interrupted file is not kept).
   Afterwards use **Open** and browse into `videos/` to pick an uploaded clip.
7. **Save** (or `g`) writes `./tracking_states/<video_stem>.pt` with the
   tracking state of every buffer that has prompts (plus their text
   prompts).
   **Close** (`q`) saves state first, then releases the video — the same as
   quitting the script's window.

### Text prompts (SAM3 only)

**Set Text Prompt** runs the detector on the current frame and shows up to 4
candidate masks (min foreground size `max(32, H*W//2000)`, top-4 by score).
Select a candidate, then click **Store Prompt**. **Reuse Text Prompt**
re-runs this buffer's stored phrase on the current frame.

### Crop

Enable `--crop` in Settings, then use the **Crop** tool to drag a rectangle
on the frame and click **Apply Crop** (the script shows a crop window at
startup; the web version does the same thing interactively on the first frame).

## Settings panel = script command-line parameters

| Setting (panel) | Script flag |
| --- | --- |
| Model path | `-m / --model_path` (default `./model_weights/sam3.pt`) |
| Image path | `-i / --image_path` (kept for parity; unused for videos) |
| Device | `-d / --device` |
| Use float32 | `-f32 / --use_float32` |
| Use aspect ratio | `-ar / --use_aspect_ratio` |
| Base size px | `-b / --base_size_px` |
| Display size | `-s / --display_size` |
| Num buffers | `-n / --num_buffers` |
| Max memories | `--max_memories` |
| Max pointers | `--max_pointers` |
| Keep bad objscores | `--keep_bad_objscores` |
| Keep history on new prompts | `--keep_history_on_new_prompts` (always-on in the CLI) |
| Object score threshold | `--objscore_threshold` |
| Hide info | `--hide_info` |
| Use webcam | `-cam / --use_webcam` |
| Disable save | `-nosave / --disable_save` |
| FFmpeg path | `--ffmpeg` (empty = save tarfiles; e.g. `ffmpeg` for mp4) |
| Enable crop | `--crop` |
| Background color | `-bg / --bg_color_hex` |

Changing model / device / float32 / buffer count requires re-Opening the video
(the panel tells you when that's needed).

## Video Segmentation tab (Task 2)

Switch to the **Video Segmentation** tab (top of the page). It has two parts:
a **New segmentation task** form (left) and a **Task queue** (right).

### Submitting a task

Pick a **Task type**:

- **Single video** — equivalent to
  `python load_prompts_run_video.py --prompt_path <pt> --input_video <video>`.
  Use **Browse** to pick the server-side **Prompt file (.pt)** and the
  **Input video**.
- **Video folder** — equivalent to
  `python load_prompts_run_dir.py --prompt_path <pt> --input_dir <dir>`.
  Use **Browse** to pick the **Prompt file (.pt)** and the **Videos folder**.
  The folder's top-level `*.mp4` files are each queued as their own job (same
  as the script, which runs the per-video script on each clip).

The **Settings** dropdown exposes the relevant `load_prompts_run_video.py`
parameters: model path, device, base size, num buffers, background color,
ffmpeg path, aspect-ratio / float32, and the `--pure_text` mode (runs each
saved text prompt on every frame, SAM3 only) with its score threshold.

### Queue & progress

Submitted jobs run **one at a time, in order**, on a background worker thread.
The queue lists every job with:

- a status pill (`queued` / `loading` / `running` / `saving` / `done` /
  `error` / `cancelled`),
- a live progress bar + `frame N/total`,
- an estimated remaining time while running (`~Xm Ys left`), computed from
  the frame-processing rate (model/prompt loading time is excluded),
- the saved result path(s) once done, or the error message on failure.

Folder batches are grouped under a header showing the folder name and clip
count, so each clip's progress is individually visible. **Cancel** works on
queued jobs (dropped) and running jobs (finishes the current frame, then stops
without saving). **Clear all** removes completed/failed/cancelled jobs from
the list **and** wipes all generated segmentation artifacts - the Python
equivalent of the repo-root `clear_all.sh`: `./saved_images/run_video/`,
`./generated_mask_videos/`, and every `./videos/<group>/<name>/` frame folder
that has a sibling `./videos/<group>/<name>.mp4` (source videos are kept).

Finished jobs **persist across server restarts** — they are written to
`webui/seg_job_history.json` and restored on startup (jobs that were still
queued/running when the server stopped are shown as cancelled with a note).
They are removed only by **Clear all**.

### Preview / Accept / Delete (finished jobs)

Each finished job card has three buttons on the bottom right:

- **Preview** — generates the mask preview video (the `check_generated_masks.py`
  flow: extract the job's tar files, merge multiple object masks into rgba
  png frames, encode a white-background mp4). The frames are kept in
  `./generated_mask_videos/<video_stem>/` (NOT moved to `./videos/...` yet)
  and the video in `./generated_mask_videos/<video_stem>_mask.mp4`. While
  generating, the button shows a circular progress ring; when done it becomes
  a play button, and pressing it again opens a video dialog (seekable
  progress bar, X in the top-right corner; the video just stops when finished,
  the window stays open). Requires `imageio` + `imageio-ffmpeg` (same as
  `util/multithread_video_writer.py`).
- **Accept** — moves the generated frames to `./videos/<garment>/<video_stem>/`
  (`<garment>` is the video stem without its last `_`-separated part, exactly
  like `check_generated_masks.py`).
- **Delete** — deletes the job's saved result files (tars or mp4), the
  generated frames and the preview mp4. Frames already accepted under
  `./videos/<garment>/<video_stem>/` are kept.

Preview/Accept need tar results; jobs saved as mp4 (ffmpeg configured) show
those two buttons disabled.

Results are written to `<repo_root>/saved_images/run_video/<video_stem>/`
(ffmpeg mp4 if an ffmpeg path is set, otherwise a PNG tarfile) — the same
output layout as the script.

### Segmentation API

- `GET  /api/seg/queue` — queue + counts + progress
- `POST /api/seg/submit` — `{kind, prompt_path, video_path|input_dir, config}`
- `POST /api/seg/cancel` — `{job_id}`
- `POST /api/seg/clear` — **Clear all**: remove finished jobs + wipe all
  generated artifacts (clear_all.sh equivalent)
- `POST /api/seg/delete_job` — `{job_id}` remove a cancelled job's bar (also from the persisted history)
- `GET  /api/seg/browse?path=&kind=pt|video|dir` — server path browser
- `POST /api/seg/preview` — `{job_id}` start mask preview generation
- `GET  /api/seg/preview_video?job_id=` — stream the preview mp4 (Range-aware)
- `POST /api/seg/accept` — `{job_id}` move frames to `./videos/<garment>/<name>/`
- `POST /api/seg/delete` — `{job_id}` delete the job's result artifacts

## Files

- `server.py` — stdlib HTTP server; authoring session + background
  segmentation queue (both mirror the reference scripts)
- `static/index.html`, `static/style.css`, `static/app.js` — the web page
- `task.md` — task description
- `seg_job_history.json` — created at runtime; persisted finished
  segmentation jobs (survives restarts, cleared by **Clear all**)

## Notes

- Single-user tool: one shared session, all requests serialized.
- Like the script, the web server keeps `ReversibleLoopingVideoReader` as the
  single source of truth for playback (forward/reverse/looping).
- Mask previews are encoded with `imageio` + `imageio-ffmpeg` (the same
  dependency `util/multithread_video_writer.py` / `check_generated_masks.py`
  use); only the Preview action needs it.
- Per the task, this was developed only — it has not been run/tested on this
  machine.
