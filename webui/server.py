#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web UI for `save_prompts_run_video.py`

A single-file, dependency-free (Python stdlib only) HTTP server that reproduces
the functionality and command-line parameters of `save_prompts_run_video.py`
in a web page served on port 8765 (bound to 0.0.0.0 so any IP can connect).

Differences from the original script (per webui/task.md):
  * `--input_video xxx.mp4` is replaced by an interactive "Open" button that
    lets the user browse and pick a file on the server running this page.
  * A "Save" button saves `./tracking_states/<video_stem>.pt` (repo root)
    (in the script this happened automatically when the window closed).

Run from the repository root:
    python webui/server.py
then open http://<host>:8765/ in a browser (from any machine on the network).

The server re-uses the exact model & helper code from the `muggled_sam`
package (make_sam_from_state_dict, SAMVideoObjectResults, make_tracking_state,
FrameCompositing, get_contours_from_mask, ReversibleLoopingVideoReader, ...)
and follows the script's per-frame logic (prompt encode -> generate_masks,
store -> initialize_video_masking, tracking -> step_video_masking, recording,
buffer saving with ffmpeg/tar fallback) so results match the original.
"""

# ---------------------------------------------------------------------------------------------------------------------
# %% Imports (standard library only for the web layer; model deps come from the repo)

import argparse
import base64
import json
import os
import os.path as osp
import sys
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Make the `muggled_sam` package (repo root = parent of this folder) importable
_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from muggled_sam.make_sam import make_sam_from_state_dict  # noqa: E402
from muggled_sam.demo_helpers.ui.video import ReversibleLoopingVideoReader  # noqa: E402
from muggled_sam.demo_helpers.ui.helpers.images import (  # noqa: E402
    FrameCompositing,
    CheckerPattern,
    get_image_hw_for_max_side_length,
)
from muggled_sam.demo_helpers.shared_ui_layout import BaseUIControl  # noqa: E402
from muggled_sam.demo_helpers.misc import (  # noqa: E402
    PeriodicVRAMReport,
    make_device_config,
    get_default_device_string,
)
from muggled_sam.demo_helpers.contours import get_contours_from_mask, pixelize_contours  # noqa: E402
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults  # noqa: E402
from muggled_sam.demo_helpers.video_prompt_state import make_tracking_state  # noqa: E402
from muggled_sam.demo_helpers.saving import save_video_frames, get_save_name  # noqa: E402
from muggled_sam.demo_helpers.ffmpeg import (  # noqa: E402
    get_default_ffmpeg_command,
    verify_ffmpeg_path,
    save_video_stream,
)
from muggled_sam.demo_helpers.history_keeper import HistoryKeeper  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# %% Defaults (mirroring the argparse defaults in save_prompts_run_video.py)

DEFAULT_PORT = 8765
DEFAULT_HOST = "0.0.0.0"  # any IP can connect

DEFAULT_IMAGE_PATH = None
DEFAULT_MODEL_PATH = None
DEFAULT_DISPLAY_SIZE = 900
DEFAULT_BASE_SIZE = 2048
DEFAULT_MAX_MEMORY_HISTORY = 6
DEFAULT_MAX_POINTER_HISTORY = 15
DEFAULT_NUM_OBJECT_BUFFERS = 4
DEFAULT_OBJECT_SCORE_THRESHOLD = 0.0
DEFAULT_BG_COLOR_HEX = "ff00ff00"
DEFAULT_FFMPEG = get_default_ffmpeg_command()

# Relative paths are resolved against the repo root so that `./model_weights/sam3.pt`
# works even if the server is launched from elsewhere. Tracking states are saved
# to <repo_root>/tracking_states/<video_stem>.pt (folder created on demand),
# e.g. clip.mp4 -> tracking_states/clip.pt (webcam -> tracking_states/webcam.pt).
DEFAULT_MODEL_FILE = osp.join(_REPO_ROOT, "model_weights", "sam3.pt")
STATE_SAVE_DIR = osp.join(_REPO_ROOT, "tracking_states")
# 'Upload video dir' destination: <repo_root>/videos/<picked folder name>/...
UPLOAD_ROOT = osp.join(_REPO_ROOT, "videos")
_UPLOAD_LOCK = threading.Lock()


def _validate_upload_dirname(name):
    """Single, safe path component for the picked folder (no traversal)."""
    name = str(name or "").strip()
    if not name or name in (".", "..") or len(name) > 128:
        raise ValueError(f"Invalid upload folder name: {name!r}")
    if "/" in name or "\\" in name or "\0" in name:
        raise ValueError(f"Invalid upload folder name: {name!r}")
    return name


def _validate_upload_relpath(rel):
    """Relative file path inside the picked folder (no traversal/abs paths)."""
    rel = str(rel or "").replace("\\", "/").strip().lstrip("/")
    if not rel or len(rel) > 512 or "\0" in rel:
        raise ValueError(f"Invalid upload file path: {rel!r}")
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid upload file path: {rel!r}")
    return "/".join(parts)

VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg",
    ".wmv", ".flv", ".3gp", ".3g2", ".asf", ".ogv", ".ts", ".mxf",
}


# ---------------------------------------------------------------------------------------------------------------------
# %% Helper data types (copied from save_prompts_run_video.py)


@dataclass
class MaskResults:
    """Storage for (per-object) displayable masking results"""

    preds: torch.Tensor
    idx: int = 0
    objscore: float = 0.0

    @classmethod
    def create(cls, mask_predictions, mask_index=1, object_score=0.0):
        """Helper used to create an empty instance of mask results"""
        empty_predictions = torch.full_like(mask_predictions, -7)
        return cls(empty_predictions, mask_index, object_score)

    def clear(self):
        self.preds = torch.zeros_like(self.preds)
        self.objscore = 0.0
        return self

    def update(self, mask_predictions, mask_index, object_score=None):
        if mask_predictions is not None:
            self.preds = mask_predictions
        if mask_index is not None:
            self.idx = mask_index
        if object_score is not None:
            self.objscore = object_score
        return self


@dataclass
class SaveBufferData:
    """Storage for (per-object) encoded png save data"""

    png_per_frame_dict: dict = field(default_factory=dict)
    bytes_per_frame_dict: dict = field(default_factory=dict)
    total_bytes: int = 0

    @classmethod
    def create(cls):
        """Helper used to create an empty instance of save buffer data"""
        return cls({}, {}, 0)

    def clear(self):
        self.png_per_frame_dict = {}
        self.bytes_per_frame_dict = {}
        self.total_bytes = 0
        return self


# ---------------------------------------------------------------------------------------------------------------------
# %% Image encoding helpers


def b64_jpeg(image_bgr, quality=90):
    """Encode a BGR numpy image to a base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        ok, buf = cv2.imencode(".jpg", image_bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def b64_png_gray(image_gray, max_side_length=None):
    """Encode a single-channel numpy image to a base64 PNG string.

    If max_side_length is given, the image is downscaled so that its longest
    side matches it, preserving the original aspect ratio.
    """
    if max_side_length is not None:
        img_h, img_w = image_gray.shape[0:2]
        longest = max(img_h, img_w)
        if longest > max_side_length:
            scale = max_side_length / longest
            new_w = max(1, int(round(img_w * scale)))
            new_h = max(1, int(round(img_h * scale)))
            image_gray = cv2.resize(image_gray, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", image_gray)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def scale_to_max_side(image_bgr, max_side_length):
    """Scale an image so its longest side matches max_side_length (aspect preserved)."""
    img_h, img_w = image_bgr.shape[0:2]
    if max(img_h, img_w) <= max_side_length:
        return image_bgr
    out_h, out_w = get_image_hw_for_max_side_length(image_bgr, max_side_length)
    return cv2.resize(image_bgr, dsize=(out_w, out_h))


# ---------------------------------------------------------------------------------------------------------------------
# %% Session


class Session:
    """
    Holds all model + video + per-object state.

    The method names follow the script's flow:
      open()            -> script setup (model load, video open, initial encode)
      _refresh_display  -> PAUSED state (generate_masks / display-only tracking step)
      advance_and_track -> TRACKING state (per-object step_video_masking + recording)
      store_prompt      -> 'Store Prompt' button
      save_state        -> the Save button (torch.save of make_tracking_state output)
    """

    def __init__(self):
        self.lock = threading.RLock()

        # Config (mirrors the argparse fields of save_prompts_run_video.py)
        self.cfg = {
            "image_path": DEFAULT_IMAGE_PATH,      # -i (kept for parity; unused for videos)
            "model_path": DEFAULT_MODEL_PATH,      # -m
            "display_size": DEFAULT_DISPLAY_SIZE,  # -s
            "device": get_default_device_string(),  # -d
            "use_float32": False,                  # -f32
            "use_aspect_ratio": True,              # -ar
            "base_size_px": DEFAULT_BASE_SIZE,     # -b
            "num_buffers": DEFAULT_NUM_OBJECT_BUFFERS,  # -n
            "max_memories": DEFAULT_MAX_MEMORY_HISTORY,
            "max_pointers": DEFAULT_MAX_POINTER_HISTORY,
            "keep_bad_objscores": False,
            "keep_history_on_new_prompts": True,
            "objscore_threshold": DEFAULT_OBJECT_SCORE_THRESHOLD,
            "hide_info": False,
            "use_webcam": False,                   # -cam
            "disable_save": False,                 # -nosave
            "ffmpeg": None,                        # --ffmpeg
            "crop": False,
            "bg_color_hex": DEFAULT_BG_COLOR_HEX,  # -bg
        }

        # Model
        self.sammodel = None
        self.model_config_dict = None
        self.model_version = None
        self.model_name = None
        self.model_device = None
        self.model_dtype = None
        self.device_config_dict = None
        self._model_key = None

        # Video
        self.vreader = None
        self.vreader_it = None
        self.video_path = None
        self.video_fps = 30.0
        self.total_frames = 0
        self.frame_hw = (1, 1)
        self.frame_idx = 0
        self.is_reversed = False
        self.end_reached = False
        self.is_paused = True
        self._cached_full_frame = None
        self._cached_full_frame_idx = None

        # Image encoding cache (like imgenc_idx_keeper in the script)
        self.encoded_img = None
        self.token_hw = None
        self.preencode_img_hw = None
        self.imgenc_idx = -1
        self.imgenc_config_dict = {}

        # Crop
        self.yx_crop_slice = None
        self.crop_tlbr_norm = None

        # Per-object storage
        self.num_obj_buffers = DEFAULT_NUM_OBJECT_BUFFERS
        self.objiter = list(range(self.num_obj_buffers))
        self.memory_list = []
        self.prompt_store_log = []  # per-buffer list of 'Store Prompt' events (for undo)
        self.maskresults_list = []
        self.savebuffers_list = []

        # Text prompts (SAM3)
        self.text_prompts_by_object = {}
        self.text_prompt_drafts_by_object = {}
        self.text_candidate_previews = {}
        self.detector_model = None

        # Working prompts (normalized), shared like the original UI overlays
        # (boxes_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)
        self.prompts = ([], [], [])
        self._prompts_consumed_idx = None  # (frame_idx, id of prompt set) bookkeeping

        # UI state
        self.buffer_select_idx = 0
        self.selected_mask_idx = 1
        self.show_preview = False
        self.invert_mask = False
        self.enable_history = True
        self.is_record_enabled = False

        # Masking / compositing
        self.mask_color_bgra = None
        self.save_masking = None
        self.checker = None

        # ffmpeg
        self.use_ffmpeg = False
        self.ffmpeg_path = None

        # Misc
        self.vram_report = PeriodicVRAMReport(update_period_ms=2000)
        self.history = HistoryKeeper()
        self.is_open = False
        self.last_error = None
        self.log = deque(maxlen=60)

    # ------------------------------------------------------------------ logging

    def _log(self, message):
        self.log.append(str(message))
        print(message, flush=True)

    # ------------------------------------------------------------------ config

    def config_snapshot(self):
        return dict(self.cfg)

    def set_config(self, updates: dict):
        """
        Apply a subset of config values. Returns (config, need_reopen) where
        need_reopen is True when the video must be (re)opened for the change
        to take effect (model weights / device / float32 / buffer count).
        """
        with self.lock:
            need_reopen = False
            model_keys = {"model_path", "device", "use_float32", "num_buffers"}
            for key, value in updates.items():
                if key not in self.cfg:
                    continue
                if key == "ffmpeg":
                    value = value if value not in (None, "") else None
                if value == self.cfg.get(key):
                    continue
                if key in model_keys:
                    need_reopen = True
                self.cfg[key] = value

            if "ffmpeg" in updates:
                self._verify_ffmpeg()

            if "bg_color_hex" in updates:
                self._setup_masking()

            if "num_buffers" in updates or "disable_save" in updates:
                self.num_obj_buffers = self._effective_num_buffers()
                self.objiter = list(range(self.num_obj_buffers))

            return self.config_snapshot(), need_reopen

    def _effective_num_buffers(self):
        # Like the script: num_obj_buffers = num_buffers if enable_saving else 1
        enable_saving = not self.cfg["disable_save"]
        return self.cfg["num_buffers"] if enable_saving else 1

    def _setup_masking(self):
        # Set up masking/chroma-keying for saving frames
        self.mask_color_bgra = FrameCompositing.parse_hex_color(self.cfg["bg_color_hex"])
        self.save_masking = FrameCompositing(self.mask_color_bgra)
        self.checker = CheckerPattern(mask_color_bgra=self.mask_color_bgra)

    def _verify_ffmpeg(self):
        # Note: verify_ffmpeg_path raises SystemExit on bad paths -> catch both
        try:
            self.use_ffmpeg, self.ffmpeg_path = verify_ffmpeg_path(self.cfg["ffmpeg"])
            if self.use_ffmpeg:
                self._log(f"Found valid FFmpeg path: @ {self.ffmpeg_path}")
        except (Exception, SystemExit) as err:  # noqa: BLE001
            self.use_ffmpeg, self.ffmpeg_path = False, None
            self._log(f"Invalid FFmpeg path ({err}). Will save buffers as tarfiles.")

    # ------------------------------------------------------------------ open

    def open(self, video_path=None, use_webcam=None):
        """Load model + open video. Mirrors the script's setup section."""
        with self.lock:
            # Webcam / path
            use_webcam = self.cfg["use_webcam"] if use_webcam is None else bool(use_webcam)
            self.cfg["use_webcam"] = use_webcam
            if use_webcam:
                video_path = 0
            else:
                if video_path in (None, ""):
                    raise ValueError("No video path given (use the Open button)")
                video_path = self._resolve_path(str(video_path))
                if osp.isdir(video_path):
                    raise IsADirectoryError(f"That is a folder, not a video file: {video_path}")
                if not osp.isfile(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")

            # Set up device config
            device_str = self.cfg["device"] or get_default_device_string()
            self.device_config_dict = make_device_config(device_str, self.cfg["use_float32"])

            # Get pathing to model weights (the script hard-codes sam3.pt when unset)
            model_path = self.cfg["model_path"]
            if model_path in (None, "", DEFAULT_MODEL_PATH):
                model_path = DEFAULT_MODEL_FILE
            else:
                model_path = self._resolve_path(str(model_path))
            if not osp.exists(model_path):
                raise FileNotFoundError(f"Model weights not found: {model_path}")

            # Check that we can call ffmpeg if flag was given
            self._verify_ffmpeg()

            # Set up masking/chroma-keying for saving frames (fail early on bad color)
            self._setup_masking()

            # Set up shared image encoder settings
            self.imgenc_config_dict = {
                "max_side_length": self.cfg["base_size_px"],
                "use_square_sizing": not self.cfg["use_aspect_ratio"],
            }

            # Get the model name, for reporting
            self.model_name = osp.basename(str(model_path))

            # Load model weights
            self._log(f"Loading model weights... @ {model_path}")
            self._load_model(model_path)

            # Set up access to video
            self._release_video()
            self.vreader = ReversibleLoopingVideoReader(video_path)
            self.vreader_it = iter(self.vreader)
            self.video_path = video_path
            self.video_fps = self.vreader.get_fps() or 30.0
            sample_frame = self.vreader.get_sample_frame()
            self.total_frames = self.vreader.total_frames
            self.frame_hw = sample_frame.shape[0:2]
            self.frame_idx = 0
            self.is_reversed = False
            self.end_reached = False
            self.is_paused = True
            self.vreader.pause(True)

            # Crop: default to the full frame; the user can draw a crop box
            # (the script shows an interactive crop window when --crop is set)
            self.yx_crop_slice = None
            self.crop_tlbr_norm = None
            if self.cfg["crop"]:
                self._log("Cropping enabled: drag a crop box on the first frame, then click Apply Crop")

            # Initial model run to make sure everything succeeds
            self._log("Encoding image data...")
            sample_for_encode = sample_frame
            if self.cfg["crop"] and self.yx_crop_slice is not None:
                sample_for_encode = sample_frame[self.yx_crop_slice]
            self.encoded_img, self.token_hw, self.preencode_img_hw = self.sammodel.encode_image(
                sample_for_encode, **self.imgenc_config_dict
            )
            self.imgenc_idx = self.frame_idx

            # Run model without prompts as sanity check (gives initial result values)
            prompts = ([], [], [])
            encoded_prompts = self.sammodel.encode_prompts(*prompts)
            init_mask_preds, _ = self.sammodel.generate_masks(
                self.encoded_img, encoded_prompts, blank_promptless_output=False
            )

            # Model reporting (like the script's 'Config' print)
            image_hw_str = f"{self.preencode_img_hw[0]} x {self.preencode_img_hw[1]}"
            token_hw_str = f"{self.token_hw[0]} x {self.token_hw[1]}"
            self._log(
                f"Config ({self.model_name}): Device: {self.model_device} ({self.model_dtype}), "
                f"Resolution HW: {image_hw_str}, Tokens HW: {token_hw_str}"
            )

            # Set up per-object storage for masking/saving results
            self.num_obj_buffers = self._effective_num_buffers()
            self.objiter = list(range(self.num_obj_buffers))
            self.maskresults_list = [MaskResults.create(init_mask_preds) for _ in self.objiter]
            self.savebuffers_list = [SaveBufferData.create() for _ in self.objiter]
            self.memory_list = [
                SAMVideoObjectResults.create(
                    self.cfg["max_memories"], self.cfg["max_pointers"], prompt_history_length=32
                )
                for _ in self.objiter
            ]
            self.prompt_store_log = [[] for _ in self.objiter]

            # Reset prompt/text/UI state
            self.prompts = ([], [], [])
            self.text_prompts_by_object = {}
            self.text_prompt_drafts_by_object = {}
            self.text_candidate_previews = {}
            self.detector_model = None
            self.buffer_select_idx = 0
            self.selected_mask_idx = 1
            self.show_preview = False
            self.invert_mask = False
            self.enable_history = True
            self.is_record_enabled = False
            self._cached_full_frame = None
            self._cached_full_frame_idx = None

            self.is_open = True

            # Store history for use on reload (like the script's HistoryKeeper)
            if use_webcam:
                self.history.store(model_path=model_path)
            else:
                self.history.store(video_path=video_path, model_path=model_path)

            # Read the first frame so the client can render immediately
            self._cache_frame(self._read_current_frame())
            return self.status_dict()

    def _load_model(self, model_path):
        # Reload only when the model-affecting config changed
        key = (str(model_path), self.cfg["device"], self.cfg["use_float32"])
        if self.sammodel is not None and self._model_key == key:
            return
        del self.sammodel
        self.sammodel = None
        self.model_config_dict, self.sammodel = make_sam_from_state_dict(model_path)
        assert self.sammodel.name in ("samv2", "samv3"), (
            "Only SAMv2/v3 models are supported for video predictions!"
        )
        self.sammodel.to(**self.device_config_dict)
        self.model_version = self.sammodel.name
        self.model_device = self.device_config_dict["device"]
        self.model_dtype = str(self.device_config_dict["dtype"]).split(".")[-1]
        self._model_key = key
        self.detector_model = None

    def _release_video(self):
        if self.vreader is not None:
            try:
                self.vreader.release()
            except Exception:  # noqa: BLE001
                pass
            self.vreader = None
            self.vreader_it = None

    def close(self):
        with self.lock:
            self._release_video()
            self.is_open = False
            self._log("Video closed")

    def _resolve_path(self, path):
        path = os.path.expanduser(str(path))
        if not osp.isabs(path):
            cwd_path = osp.abspath(path)
            if osp.exists(cwd_path):
                return cwd_path
            return osp.join(_REPO_ROOT, path)
        return path

    # ------------------------------------------------------------------ frame access
    # (The reader's own iterator is the single source of truth for playback,
    #  exactly like the script's `for is_paused, frame_idx, frame in vreader` loop.)

    def _read_current_frame(self):
        """Read the frame at the reader's current position (paused)."""
        self.vreader.pause(True)
        is_paused, frame_idx, frame = next(self.vreader_it)
        self.frame_idx = frame_idx
        return frame

    def _advance_reader(self):
        """Advance one frame using the reader's forward/reverse/looping logic."""
        if self.total_frames <= 1 and self.is_reversed:
            # Reverse indexing would divide by zero on 1-frame/webcam sources
            self.is_reversed = False
        self.vreader.pause(False)
        is_paused, frame_idx, frame = next(self.vreader_it)
        self.frame_idx = frame_idx
        return frame

    def _cache_frame(self, full_frame):
        self._cached_full_frame = full_frame
        self._cached_full_frame_idx = self.frame_idx
        return full_frame

    def _get_full_frame(self, frame_idx=None):
        """Return the uncropped frame at the given index (seeking if needed)."""
        if frame_idx is not None and int(frame_idx) != self.frame_idx:
            self.vreader.pause(True)
            clamped = max(0, min(int(frame_idx), max(self.total_frames - 1, 0)))
            self.frame_idx = clamped
            self.vreader.set_playback_position(clamped)
            self._cache_frame(self._read_current_frame())
            return self._cached_full_frame
        if self._cached_full_frame_idx == self.frame_idx and self._cached_full_frame is not None:
            return self._cached_full_frame
        self._cache_frame(self._read_current_frame())
        return self._cached_full_frame

    def _crop_frame(self, full_frame):
        if self.cfg["crop"] and self.yx_crop_slice is not None:
            return full_frame[self.yx_crop_slice]
        return full_frame

    def _encode_frame(self, frame_bgr):
        self.encoded_img, self.token_hw, self.preencode_img_hw = self.sammodel.encode_image(
            frame_bgr, **self.imgenc_config_dict
        )
        return self.encoded_img

    def _maybe_encode(self, frame_bgr):
        """Encode 'new' frames as needed (mirrors imgenc_idx_keeper)."""
        if self.imgenc_idx != self.frame_idx:
            self._encode_frame(frame_bgr)
            self.imgenc_idx = self.frame_idx
            return True
        return False

    # ------------------------------------------------------------------ state checks

    def _have_user_prompts(self):
        return self.sammodel.check_have_prompts(*self.prompts)

    def _have_track_prompts(self):
        return any(mem.check_has_prompts() for mem in self.memory_list)

    def _text_preview_active(self):
        preview = self.text_candidate_previews.get(self.buffer_select_idx)
        return preview is not None and preview["frame_idx"] == self.frame_idx

    def _run_display_tracking_step(self, buffer):
        """Display-only tracking step (the script's PAUSED-branch tracker run)."""
        if not self.memory_list[buffer].check_has_prompts():
            return
        selected_memory_dict = self.memory_list[buffer].to_dict()
        obj_score, _, mask_preds, _, _ = self.sammodel.step_video_masking(
            self.encoded_img, **selected_memory_dict
        )
        obj_score = float(obj_score.squeeze().float().cpu().numpy())
        self.maskresults_list[buffer].update(mask_preds, self.selected_mask_idx, obj_score)

    def _discard_stale_text_previews(self):
        # Text candidates are valid only for the frame on which detection ran
        stale = [
            idx for idx, p in self.text_candidate_previews.items() if p["frame_idx"] != self.frame_idx
        ]
        for idx in stale:
            self.text_candidate_previews.pop(idx)
            self.maskresults_list[idx].clear()

    # ------------------------------------------------------------------ PAUSED state

    def _refresh_display(self):
        """
        Recompute the (selected buffer's) displayed mask for the current frame.
        Mirrors the PAUSED branch of the script's main loop:

          - if the user has prompts (or no buffer has tracking prompts) and there
            is no text preview: encode prompts + generate_masks
          - else if the selected buffer has tracking prompts (and no user prompts)
            and the frame is new: run one display-only tracking step
        """
        full_frame = self._get_full_frame()
        frame_bgr = self._crop_frame(full_frame)
        self._discard_stale_text_previews()
        self._maybe_encode(frame_bgr)

        have_user = self._have_user_prompts()
        have_track = self._have_track_prompts()
        has_text = self._text_preview_active()

        paused_mask_preds = None
        paused_obj_score = None

        if (have_user or not have_track) and not has_text:
            encoded_prompts = self.sammodel.encode_prompts(*self.prompts)
            paused_mask_preds, _ = self.sammodel.generate_masks(
                self.encoded_img,
                encoded_prompts,
                mask_hint=None,
                blank_promptless_output=True,
            )

        elif have_track and not have_user and not has_text:
            selected_memory_dict = self.memory_list[self.buffer_select_idx].to_dict()
            paused_obj_score, _, paused_mask_preds, _, _ = self.sammodel.step_video_masking(
                self.encoded_img, **selected_memory_dict
            )
            paused_obj_score = float(paused_obj_score.squeeze().float().cpu().numpy())

        # Store user-interaction results for selected object while paused
        self.maskresults_list[self.buffer_select_idx].update(
            paused_mask_preds, self.selected_mask_idx, paused_obj_score
        )
        return full_frame

    # ------------------------------------------------------------------ prompt API

    def set_prompts(self, boxes=None, fg=None, bg=None):
        """Replace the working prompt set, then recompute (mirrors overlay edits)."""
        with self.lock:
            self._require_open()
            self.prompts = (list(boxes or []), list(fg or []), list(bg or []))
            self._refresh_display()
            return self.display_payload()

    def add_prompt(self, kind, x=None, y=None, x2=None, y2=None):
        """Add a single prompt (client convenience; full-set replacement is preferred)."""
        with self.lock:
            self._require_open()
            if kind == "clear":
                self.prompts = ([], [], [])
            elif kind == "fg_point":
                self.prompts[1].append((float(x), float(y)))
            elif kind == "bg_point":
                self.prompts[2].append((float(x), float(y)))
            elif kind == "box":
                xa, xb = sorted((float(x), float(x2)))
                ya, yb = sorted((float(y), float(y2)))
                self.prompts[0].append(((xa, ya), (xb, yb)))
            else:
                raise ValueError(f"Unknown prompt kind: {kind}")
            self._refresh_display()
            return self.display_payload()

    def select_mask(self, idx):
        """Select one of the 4 mask predictions (like the arrow keys / mask buttons).
        No model run needed - just swap which prediction is shown."""
        with self.lock:
            self._require_open()
            idx = int(idx)
            if idx < 0 or idx > 3:
                raise ValueError(f"Mask index out of range: {idx}")
            self.selected_mask_idx = idx
            self.maskresults_list[self.buffer_select_idx].idx = idx
            return self.display_payload()

    def store_prompt(self, buffer=None):
        """
        'Store Prompt' button: turn the current working prompts (or the selected
        text candidate) into a stored tracking prompt for the selected buffer.
        """
        with self.lock:
            self._require_open()
            buffer = self.buffer_select_idx if buffer is None else int(buffer)
            self._get_full_frame()  # make sure the frame/encoding is current

            selected_text_preview = self.text_candidate_previews.get(buffer)
            has_text_preview = (
                selected_text_preview is not None and selected_text_preview["frame_idx"] == self.frame_idx
            )

            if has_text_preview:
                if self.selected_mask_idx >= selected_text_preview["num_candidates"]:
                    raise ValueError("Select a non-empty text candidate before storing.")
                selected_mask = selected_text_preview["mask_predictions"][0, self.selected_mask_idx]
                init_mem = self.sammodel.initialize_from_mask(self.encoded_img, selected_mask)
                self.memory_list[buffer].store_prompt_result(self.frame_idx, init_mem)
                self._log_stored_prompt(buffer, has_ptr=False)
                if not self.cfg["keep_history_on_new_prompts"]:
                    self.memory_list[buffer].prevframe_buffer.clear()
                self.text_prompts_by_object[buffer] = selected_text_preview["text_prompt"]
                self.text_prompt_drafts_by_object[buffer] = selected_text_preview["text_prompt"]
                self.text_candidate_previews.pop(buffer, None)
                self._log(
                    f"Stored text candidate for Buffer {buffer + 1} "
                    f"({self.memory_list[buffer].get_num_memories()[0]} prompt frame(s) stored)."
                )
            else:
                if not self._have_user_prompts():
                    raise ValueError("No prompts to store: draw a box / points first (or use a text prompt).")
                _, init_mem, init_ptr = self.sammodel.initialize_video_masking(
                    self.encoded_img,
                    *self.prompts,
                    mask_index_select=self.selected_mask_idx,
                )
                self.memory_list[buffer].store_prompt_result(self.frame_idx, init_mem, init_ptr)
                self._log_stored_prompt(buffer, has_ptr=(init_ptr is not None))

            # Clear working prompts after storing (like ui_elems.clear_prompts())
            self.prompts = ([], [], [])

            # Show the tracked result at this frame (the script's next loop
            # iteration runs a display-only tracking step after a prompt is stored)
            if buffer == self.buffer_select_idx:
                self._run_display_tracking_step(buffer)
            return self.display_payload()

    def clear_prompts(self, buffer=None):
        """'Clear Prompts' button: wipe the stored prompt data of the selected buffer."""
        with self.lock:
            self._require_open()
            buffer = self.buffer_select_idx if buffer is None else int(buffer)
            self.memory_list[buffer].prompts_buffer.clear()
            self.prompt_store_log[buffer] = []
            self.maskresults_list[buffer].clear()
            self.text_prompts_by_object.pop(buffer, None)
            self.text_prompt_drafts_by_object.pop(buffer, None)
            self.text_candidate_previews.pop(buffer, None)
            return self.display_payload()

    def _log_stored_prompt(self, buffer, has_ptr):
        """Remember one 'Store Prompt' event so /api/undo_prompt can reverse it.
        Needed because text-candidate stores append no object pointer, which would
        otherwise misalign the prompts_buffer's three deques."""
        log = self.prompt_store_log[buffer]
        log.insert(0, {"has_ptr": has_ptr})
        if len(log) > 32:  # prompts_buffer capacity
            log.pop()

    def undo_prompt(self, buffer=None):
        """'Undo Prompt' button: remove the most recently stored prompt of the
        selected buffer (one 'Store Prompt' step back), keeping older ones."""
        with self.lock:
            self._require_open()
            buffer = self.buffer_select_idx if buffer is None else int(buffer)
            log = self.prompt_store_log[buffer]
            if not log:
                raise ValueError("Nothing to undo: no stored prompts in this buffer.")
            entry = log.pop(0)
            pb = self.memory_list[buffer].prompts_buffer
            pb.idx.popleft()
            pb.memory_history.popleft()
            if entry["has_ptr"] and len(pb.pointer_history) > 0:
                pb.pointer_history.popleft()
            if buffer == self.buffer_select_idx:
                if self.memory_list[buffer].check_has_prompts():
                    self._run_display_tracking_step(buffer)
                else:
                    self.maskresults_list[buffer].clear()
            return self.display_payload()

    def clear_history(self, buffer=None):
        """'Clear History' button: wipe the selected buffer's previous-frame memory."""
        with self.lock:
            self._require_open()
            buffer = self.buffer_select_idx if buffer is None else int(buffer)
            self.memory_list[buffer].prevframe_buffer.clear()
            return self.display_payload()

    def select_buffer(self, idx):
        """Switch the selected buffer (like pressing b/v). Switching buffers
        clears the working prompts, exactly like the script."""
        with self.lock:
            self._require_open()
            idx = int(idx)
            if idx < 0 or idx >= self.num_obj_buffers:
                raise ValueError(f"Buffer index out of range: {idx}")
            if idx != self.buffer_select_idx:
                self.prompts = ([], [], [])
            self.buffer_select_idx = idx
            return self.display_payload()

    # ------------------------------------------------------------------ text prompts (SAM3)

    def _get_detector(self):
        if self.detector_model is None:
            if self.model_version != "samv3":
                raise ValueError("Text prompts require a SAM3 model")
            self.detector_model = self.sammodel.make_detector_model()
        return self.detector_model

    def set_text_prompt(self, text, buffer=None):
        """
        'Set Text Prompt' button (SAM3 only): run text detection on the current
        frame and show up to 4 candidate masks. Mirrors the pending-text handling
        in the script's loop.
        """
        with self.lock:
            self._require_open()
            if self.model_version != "samv3":
                raise ValueError("Text prompts require a SAM3 model")
            buffer = self.buffer_select_idx if buffer is None else int(buffer)
            text = (text or "").strip()
            if not text:
                raise ValueError("Empty text prompt")
            if not self.is_paused:
                raise ValueError("Pause the video before setting a text prompt.")

            full_frame = self._get_full_frame()
            frame = self._crop_frame(full_frame)

            self._log(f"Finding Buffer {buffer + 1} text prompt: {text!r}")
            # The frame may have changed while paused, so do not reuse a
            # potentially stale image encoding from the prior iteration.
            self.encoded_img = self._encode_frame(frame)
            self.imgenc_idx = self.frame_idx

            detector_model = self._get_detector()
            detection_img, _, _ = detector_model.encode_detection_image(frame, **self.imgenc_config_dict)
            encoded_exemplars = detector_model.encode_exemplars(detection_img, text=text)
            detection_masks, _, detection_scores, _ = detector_model.generate_detections(
                detection_img, encoded_exemplars
            )

            # Keep the phrase available for reuse even if no candidates appear
            self.text_prompt_drafts_by_object[buffer] = text

            if detection_masks is None or detection_masks.shape[1] == 0:
                self._log(f"No text candidates were produced for {text!r}.")
                return self.display_payload()

            # Some detection queries can score well while producing an empty mask.
            # Exclude those from the candidate UI.
            all_masks = detection_masks[0]
            all_scores = detection_scores.flatten()
            min_foreground_pixels = max(32, all_masks.shape[-2] * all_masks.shape[-1] // 2000)
            foreground_pixel_counts = (all_masks > 0).flatten(1).sum(dim=1)
            valid_candidate_indices = torch.nonzero(
                foreground_pixel_counts >= min_foreground_pixels, as_tuple=False
            ).flatten()
            if valid_candidate_indices.numel() == 0:
                self._log(
                    f"No non-empty text masks found for {text!r}. "
                    "Try another frame or a different prompt."
                )
                return self.display_payload()

            # Show the best non-empty detections (up to 4); user decides which to store
            num_candidates = min(4, valid_candidate_indices.numel())
            best_scores, best_local_indices = torch.topk(
                all_scores[valid_candidate_indices], k=num_candidates
            )
            best_indices = valid_candidate_indices[best_local_indices]
            candidate_masks = torch.full(
                (1, 4, *detection_masks.shape[-2:]),
                -7,
                dtype=detection_masks.dtype,
                device=detection_masks.device,
            )
            candidate_masks[0, :num_candidates] = detection_masks[0, best_indices]
            candidate_scores = torch.zeros(
                (1, 4), dtype=best_scores.dtype, device=best_scores.device
            )
            candidate_scores[0, :num_candidates] = best_scores

            self.text_candidate_previews[buffer] = {
                "frame_idx": self.frame_idx,
                "text_prompt": text,
                "mask_predictions": candidate_masks,
                "scores": candidate_scores,
                "num_candidates": num_candidates,
            }
            self.maskresults_list[buffer].update(candidate_masks, 0, float(best_scores[0]))
            self.prompts = ([], [], [])
            self.selected_mask_idx = 0
            self.maskresults_list[buffer].idx = 0
            self._log(
                f"Showing {num_candidates} text candidate(s) for Buffer {buffer + 1}. "
                "Select one, then click Store Prompt to save it."
            )
            return self.display_payload()

    def reuse_text_prompt(self, buffer=None):
        """'Reuse Text Prompt' button: re-run this buffer's stored text on the
        current frame (adds another text-derived prompt frame once stored)."""
        with self.lock:
            self._require_open()
            buffer = self.buffer_select_idx if buffer is None else int(buffer)
            saved = self.text_prompt_drafts_by_object.get(
                buffer, self.text_prompts_by_object.get(buffer)
            )
            if not saved:
                raise ValueError(f"Buffer {buffer + 1} has no text prompt to reuse.")
            self._log(f"Reusing Buffer {buffer + 1} text prompt: {saved!r}")
            return self.set_text_prompt(saved, buffer=buffer)

    # ------------------------------------------------------------------ playback

    def seek(self, frame_idx):
        """Move the playback position (like dragging the slider).
        The script wipes displayed masking/prompts on playback adjustments."""
        with self.lock:
            self._require_open()
            if self.total_frames <= 1:
                return self.display_payload()
            # Accept normalized positions (float in [0, 1)) as well as frame indices
            if isinstance(frame_idx, float) and 0.0 <= frame_idx < 1.0:
                frame_idx = int(round(frame_idx * (self.total_frames - 1)))
            frame_idx = max(0, min(int(frame_idx), self.total_frames - 1))

            self.vreader.pause(True)
            self.vreader.set_playback_position(frame_idx)
            self.is_paused = True
            self.end_reached = False
            self._cache_frame(self._read_current_frame())

            # Wipe out masking/contours when jumping around playback
            for maskresult in self.maskresults_list:
                maskresult.clear()
            self.text_candidate_previews.clear()
            self.prompts = ([], [], [])

            # Encode the new frame (the script encodes 'new' frames as needed)
            self._maybe_encode(self._crop_frame(self._cached_full_frame))

            # If the selected buffer has tracking prompts, show its mask at this
            # frame (display-only step, exactly like the script's PAUSED branch)
            if self.memory_list[self.buffer_select_idx].check_has_prompts():
                selected_memory_dict = self.memory_list[self.buffer_select_idx].to_dict()
                obj_score, _, mask_preds, _, _ = self.sammodel.step_video_masking(
                    self.encoded_img, **selected_memory_dict
                )
                obj_score = float(obj_score.squeeze().float().cpu().numpy())
                self.maskresults_list[self.buffer_select_idx].update(
                    mask_preds, self.selected_mask_idx, obj_score
                )
            return self.display_payload()

    def step(self, n=1):
        """Step n frames while paused (like the slider's step buttons)."""
        with self.lock:
            self._require_open()
            if self.total_frames <= 1:
                return self.display_payload()
            target = self.frame_idx + int(n)
            return self.seek(target)

    def set_play(self, playing):
        """Play/Pause (Track) button. Mirrors the SWITCH_PAUSE_OFF / SWITCH_PAUSE_ON
        transition handling in the script."""
        with self.lock:
            self._require_open()
            playing = bool(playing)
            if playing == (not self.is_paused):
                return self.display_payload()

            if playing:
                # SWITCH_PAUSE_OFF: begin tracking
                self.is_paused = False
                self.end_reached = False  # reset here (like the script)
                self.vreader.pause(False)

                # Consume & clear the working prompts
                prompts = self.prompts
                self.prompts = ([], [], [])

                # If a prompt exists when tracking begins, assume we should use it
                if self.sammodel.check_have_prompts(*prompts):
                    _, init_mem, init_ptr = self.sammodel.initialize_video_masking(
                        self.encoded_img,
                        *prompts,
                        mask_index_select=self.maskresults_list[self.buffer_select_idx].idx,
                    )
                    self.memory_list[self.buffer_select_idx].store_prompt_result(
                        self.frame_idx, init_mem, init_ptr
                    )
                    if not self.cfg["keep_history_on_new_prompts"]:
                        self.memory_list[self.buffer_select_idx].prevframe_buffer.clear()

                # If there is no tracking data, clear any on-screen masking
                no_prompt_data = all(mem.check_has_prompts() == 0 for mem in self.memory_list)
                if no_prompt_data:
                    for maskresult in self.maskresults_list:
                        maskresult.clear()
            else:
                # SWITCH_PAUSE_ON: pause
                self.is_paused = True
                self.vreader.pause(True)
                # Consume & clear queued prompts, disable preview (like the script)
                self.prompts = ([], [], [])
                self.show_preview = False
            return self.display_payload()

    def set_reverse(self, reversed_state):
        """'Reverse' toggle: switch forward/reverse playback direction."""
        with self.lock:
            self._require_open()
            self.is_reversed = bool(reversed_state)
            if self.vreader is not None and self.total_frames > 1:
                self.vreader.toggle_reverse_state(self.is_reversed)
            return self.display_payload()

    # ------------------------------------------------------------------ TRACKING state

    def advance_and_track(self):
        """
        Advance one frame and run tracking. Mirrors the TRACKING branch of the
        script's loop plus its recording block and the 'stop at last frame' rule.
        """
        with self.lock:
            self._require_open()

            was_paused = self.is_paused
            full_frame = self._advance_reader()
            self._cache_frame(full_frame)
            frame_bgr = self._crop_frame(full_frame)

            # Stop tracking at last frame (forward direction only)
            if (
                not was_paused
                and not self.is_reversed
                and self.total_frames > 1
                and self.frame_idx >= self.total_frames - 1
            ):
                self.end_reached = True
                self.is_paused = True
                self.vreader.pause(True)
                self._log("Reached end of video. Stopping tracking.")

            # The tracking step for this frame runs if playback was active when
            # the request arrived (even when we just auto-paused at the last
            # frame - matches the script's per-iteration order)
            is_tracking = not was_paused

            self._discard_stale_text_previews()
            self._maybe_encode(frame_bgr)

            if is_tracking:
                discard_on_bad_objscore = not self.cfg["keep_bad_objscores"]
                for objidx in self.objiter:
                    # Don't run objects with no prompts
                    if not self.memory_list[objidx].check_has_prompts():
                        continue
                    obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = self.sammodel.step_video_masking(
                        self.encoded_img, **self.memory_list[objidx].to_dict()
                    )
                    obj_score = float(obj_score.squeeze().float().cpu().numpy())
                    tracked_mask_idx = int(best_mask_idx.squeeze().cpu())

                    # Only store history for high-scoring predictions
                    if obj_score < self.cfg["objscore_threshold"] and discard_on_bad_objscore:
                        mask_preds = mask_preds * 0.0
                    elif self.enable_history:
                        self.memory_list[objidx].store_frame_result(self.frame_idx, mem_enc, obj_ptr)

                    self.maskresults_list[objidx].update(mask_preds, tracked_mask_idx, obj_score)

                # Handle recording of segmentation data
                if self.is_record_enabled:
                    self._record_frame(full_frame, frame_bgr)

            return full_frame

    def _record_frame(self, full_frame, frame_bgr):
        """Record masked frames into the per-object save buffers (script logic)."""
        save_hw = frame_bgr.shape[0:2]
        for objidx in self.objiter:
            # Don't save anything for un-tracked objects
            if not self.memory_list[objidx].check_has_prompts():
                continue

            mask_preds, mask_idx = self.maskresults_list[objidx].preds, self.maskresults_list[objidx].idx
            try:
                save_mask_1ch_uint8 = BaseUIControl.create_hires_mask_uint8(
                    mask_preds, mask_idx, save_hw
                )
            except Exception:  # noqa: BLE001
                continue
            if self.invert_mask:
                save_mask_1ch_uint8 = cv2.bitwise_not(save_mask_1ch_uint8)

            # Select whether we use the existing frame/mask or expand to the full (uncropped) sizing
            save_frame = frame_bgr
            if self.cfg["crop"] and self.yx_crop_slice is not None:
                mask_bg = 255 if self.invert_mask else 0
                full_mask_1ch = np.full(full_frame.shape[0:2], mask_bg, dtype=np.uint8)
                full_mask_1ch[self.yx_crop_slice] = save_mask_1ch_uint8
                save_mask_1ch_uint8 = full_mask_1ch
                save_frame = full_frame

            # Mask out image for saving
            save_frame = self.save_masking.mask_frame(save_frame, save_mask_1ch_uint8)

            # Encode frame data in memory (saved in bulk on 'Save Buffer')
            ok_encode, png_encoding = cv2.imencode(".png", save_frame)
            if ok_encode:
                png_bytes = len(png_encoding)
                existing_bytes = self.savebuffers_list[objidx].bytes_per_frame_dict.get(self.frame_idx, 0)
                self.savebuffers_list[objidx].bytes_per_frame_dict[self.frame_idx] = png_bytes
                self.savebuffers_list[objidx].total_bytes += png_bytes - existing_bytes
                self.savebuffers_list[objidx].png_per_frame_dict[self.frame_idx] = png_encoding

    def toggle(self, name):
        """Toggle a UI switch (preview / invert / history / record)."""
        with self.lock:
            self._require_open()
            if name == "preview":
                self.show_preview = not self.show_preview
            elif name == "invert":
                self.invert_mask = not self.invert_mask
            elif name == "history":
                self.enable_history = not self.enable_history
            elif name == "record":
                self.is_record_enabled = not self.is_record_enabled
            else:
                raise ValueError(f"Unknown toggle: {name}")
            return self.display_payload()

    def set_crop(self, tlbr):
        """Apply a crop box (normalized [[x1,y1],[x2,y2]]) - the web version of
        the script's startup crop window (used when --crop is enabled)."""
        with self.lock:
            self._require_open()
            if not self.cfg["crop"]:
                raise ValueError("Cropping is disabled (enable the 'crop' setting first)")
            tlbr = np.asarray(tlbr, dtype=float)
            x1, y1 = tlbr[0]
            x2, y2 = tlbr[1]
            h, w = self.frame_hw
            x1px, y1px = int(round(x1 * w)), int(round(y1 * h))
            x2px, y2px = int(round(x2 * w)), int(round(y2 * h))
            x1px, y1px = max(0, min(x1px, w)), max(0, min(y1px, h))
            x2px, y2px = max(0, min(x2px, w)), max(0, min(y2px, h))
            # Bail if crop is too small (like run_crop_ui)
            if (abs(x2px - x1px) < 5) or (abs(y2px - y1px) < 5):
                x1px, y1px, x2px, y2px = 0, 0, w, h
            self.yx_crop_slice = (slice(y1px, y2px), slice(x1px, x2px))
            self.crop_tlbr_norm = ((x1px / w, y1px / h), (x2px / w, y2px / h))
            self._log(f"Crop applied: x {x1px}-{x2px}, y {y1px}-{y2px}")

            # Re-encode the current frame with the new crop
            self.imgenc_idx = -1
            full_frame = self._get_full_frame()
            self._refresh_display()
            return self.display_payload()

    # ------------------------------------------------------------------ saving

    def save_buffer(self, buffer=None):
        """'Save Buffer' button: save the recorded frames for a buffer
        (ffmpeg mp4 with tarfile fallback - mirrors save_segmentation_results)."""
        with self.lock:
            self._require_open()
            buffer = self.buffer_select_idx if buffer is None else int(buffer)
            png_per_frame_dict = self.savebuffers_list[buffer].png_per_frame_dict
            num_frames = len(png_per_frame_dict)
            if num_frames == 0:
                raise ValueError("No recorded frames to save")

            # Build save pathing
            base_path = self.video_path if not self.cfg["use_webcam"] else "webcam"
            save_folder, save_idx = get_save_name(base_path, "run_video")
            min_fidx, max_fidx = min(png_per_frame_dict.keys()), max(png_per_frame_dict.keys())
            save_name = f"{save_idx}_obj{1 + buffer}_{min_fidx}_to_{max_fidx}_frames"
            save_path_no_ext = osp.join(save_folder, save_name)

            # Save with ffmpeg
            use_ffmpeg = self.use_ffmpeg
            saved_path = None
            if use_ffmpeg:
                self._log("Encoding video...")
                ok_save, saved_path = save_video_stream(
                    self.ffmpeg_path, save_path_no_ext, self.video_fps, png_per_frame_dict,
                    print_progress_indicator=False,
                )
                if ok_save:
                    self._log(f"@ {saved_path}")
                else:
                    self._log("-> Error saving with FFmpeg, will fall back to saving tarfile")
                    use_ffmpeg = False

            # Not using 'else' here, because encoding may have failed
            if not use_ffmpeg:
                self._log(f"Saving frame data ({num_frames} frames)...")
                saved_path = save_video_frames(save_path_no_ext, png_per_frame_dict)
                self._log(f"@ {saved_path}")

            # Trigger clear of data so we don't re-save it
            self.savebuffers_list[buffer].clear()
            self.use_ffmpeg = use_ffmpeg
            return {"ok": True, "path": saved_path}

    def clear_buffer(self, buffer=None):
        """'Clear Buffer' button: wipe the recorded (in-memory) frames for a buffer."""
        with self.lock:
            self._require_open()
            buffer = self.buffer_select_idx if buffer is None else int(buffer)
            self.savebuffers_list[buffer].clear()
            return {"ok": True}

    def save_state(self):
        """
        The 'Save' button. Mirrors the end-of-script block:
        save <repo_root>/tracking_states/<video_stem>.pt (webcam: webcam.pt)
        with the tracking state of all buffers that have prompts
        (plus their text prompts).
        """
        with self.lock:
            self._require_open()
            has_any_prompts = any(mem.check_has_prompts() for mem in self.memory_list)
            if not has_any_prompts:
                raise ValueError("No prompts found. Store a prompt for at least one buffer first.")

            # Only save objects that actually have prompts
            save_data = {}
            for objidx in self.objiter:
                if self.memory_list[objidx].check_has_prompts():
                    save_data[objidx] = self.memory_list[objidx]

            saved_text_prompts = {
                objidx: self.text_prompts_by_object[objidx]
                for objidx in save_data
                if objidx in self.text_prompts_by_object
            }
            # Save to <repo_root>/tracking_states/, named after the video
            # (clip.mp4 -> tracking_states/clip.pt); webcam -> webcam.pt.
            os.makedirs(STATE_SAVE_DIR, exist_ok=True)
            if self.video_path and osp.isfile(self.video_path):
                stem = osp.splitext(osp.basename(self.video_path))[0]
            else:
                stem = "webcam"
            save_path = osp.join(STATE_SAVE_DIR, f"{stem}.pt")
            torch.save(make_tracking_state(save_data, saved_text_prompts), save_path)
            self._log(f"Saved tracking state to {osp.abspath(save_path)}")
            return {"ok": True, "path": save_path, "objects": sorted(save_data.keys())}

    # ------------------------------------------------------------------ rendering

    def _hires_mask_uint8(self, mask_preds, mask_idx, output_hw):
        """Binary hi-res mask (script helper BaseUIControl.create_hires_mask_uint8)."""
        return BaseUIControl.create_hires_mask_uint8(mask_preds, mask_idx, output_hw)

    def _render_display_image(self, full_frame):
        """
        Composite the display image (frame + selected mask outlines / checker
        preview + other objects' outlines), mirroring update_main_display_image
        plus the unselected-objects polygon overlay.
        """
        buffer = self.buffer_select_idx
        frame_bgr = self._crop_frame(full_frame)
        frame_h, frame_w = frame_bgr.shape[0:2]
        mr = self.maskresults_list[buffer]

        try:
            mask_uint8 = self._hires_mask_uint8(mr.preds, mr.idx, (frame_h, frame_w))
        except Exception:  # noqa: BLE001
            mask_uint8 = np.zeros((frame_h, frame_w), dtype=np.uint8)
        # Invert for display if needed
        disp_mask_uint8 = cv2.bitwise_not(mask_uint8) if self.invert_mask else mask_uint8

        if self.show_preview:
            # Checker background to suggest alpha (no contours, like the script)
            disp = self.checker.superimpose(frame_bgr, disp_mask_uint8)
            return disp

        disp = frame_bgr.copy()

        # Selected object: semi-transparent green overlay over the mask region
        if (disp_mask_uint8 > 0).any():
            overlay = disp.copy()
            overlay[disp_mask_uint8 > 0] = (0, 255, 0)  # BGR: pure green (R,G,B = 0,255,0)
            disp = cv2.addWeighted(overlay, 0.35, disp, 0.65, 0)

        # Unselected objects' outlines (dim, like the unselected overlay)
        for other in self.objiter:
            if other == buffer:
                continue
            other_mr = self.maskresults_list[other]
            try:
                other_mask = self._hires_mask_uint8(other_mr.preds, other_mr.idx, (frame_h, frame_w))
            except Exception:  # noqa: BLE001
                continue
            have_o, contours_o = get_contours_from_mask(other_mask, normalize=True)
            if have_o:
                contours_o_px = pixelize_contours(tuple(contours_o), disp.shape)
                for c in contours_o_px:
                    cv2.polylines(disp, [np.asarray(c, dtype=np.int32)], isClosed=True,
                                  color=(140, 110, 50), thickness=1)

        return disp

    def _render_previews(self):
        """Render the 4 mask preview thumbnails (script: update_mask_preview_buttons)."""
        buffer = self.buffer_select_idx
        preds = self.maskresults_list[buffer].preds  # Bx4xHxW
        try:
            preds_uint8 = ((preds.squeeze(0) > 0.0) * 255).byte().cpu().numpy()  # 4xHxW
        except Exception:  # noqa: BLE001
            preds_uint8 = np.zeros((4, 1, 1), dtype=np.uint8)
        if self.invert_mask:
            preds_uint8 = np.bitwise_not(preds_uint8)
        target = 320  # longest side of each preview; aspect ratio preserved
        return [b64_png_gray(preds_uint8[i], max_side_length=target) for i in range(4)]

    def display_payload(self):
        """Build the JSON payload for a display update (image + previews + state).
        Call with the session lock held."""
        full_frame = self._get_full_frame()
        disp = self._render_display_image(full_frame)
        disp = scale_to_max_side(disp, int(self.cfg["display_size"]))
        buffer = self.buffer_select_idx
        preview = self.text_candidate_previews.get(buffer)
        text_preview = None
        if preview is not None and preview["frame_idx"] == self.frame_idx:
            scores = [round(float(s), 3) for s in preview["scores"][0].tolist()]
            text_preview = {
                "text_prompt": preview["text_prompt"],
                "num_candidates": preview["num_candidates"],
                "scores": scores,
            }
        return {
            "frame_b64": b64_jpeg(disp),
            "previews_b64": self._render_previews(),
            "frame_idx": self.frame_idx,
            "total_frames": self.total_frames,
            "fps": self.video_fps,
            "is_paused": self.is_paused,
            "is_reversed": self.is_reversed,
            "end_reached": self.end_reached,
            "buffer": buffer,
            "mask_idx": self.selected_mask_idx,
            "score": round(self.maskresults_list[buffer].objscore, 3) if self.maskresults_list else 0.0,
            "num_buffers": self.num_obj_buffers,
            "num_prompt_mems": self.memory_list[buffer].get_num_memories()[0] if self.memory_list else 0,
            "prompts": {
                "boxes": [list(b) for b in self.prompts[0]],
                "fg": [list(p) for p in self.prompts[1]],
                "bg": [list(p) for p in self.prompts[2]],
            },
            "text_preview": text_preview,
            "text_draft": self.text_prompt_drafts_by_object.get(buffer),
            "text_stored": self.text_prompts_by_object.get(buffer),
            "use_webcam": self.cfg["use_webcam"],
            "log": list(self.log)[-12:],
        }

    def display(self, advance=False):
        """
        Full display request. If advance=True and playback is active, step the
        video (tracking) first; otherwise render the current state.
        """
        with self.lock:
            self._require_open()
            if advance and not self.is_paused:
                self.advance_and_track()
            return self.display_payload()

    # ------------------------------------------------------------------ status

    def status_dict(self):
        with self.lock:
            buffer = min(self.buffer_select_idx, max(self.num_obj_buffers - 1, 0))
            num_prompt_mems = num_prev_mems = 0
            score = 0.0
            if self.memory_list:
                num_prompt_mems, num_prev_mems = self.memory_list[buffer].get_num_memories()
            if self.maskresults_list:
                score = round(self.maskresults_list[buffer].objscore, 3)
            vram = self.vram_report.get_vram_usage()
            buffer_info = []
            for i in self.objiter:
                mem = self.memory_list[i] if i < len(self.memory_list) else None
                sb = self.savebuffers_list[i] if i < len(self.savebuffers_list) else None
                mp, pp = mem.get_num_memories() if mem is not None else (0, 0)
                buffer_info.append({
                    "idx": i,
                    "has_prompts": bool(mem.check_has_prompts()) if mem is not None else False,
                    "num_prompt_mems": mp,
                    "num_prev_mems": pp,
                    "mb": round(sb.total_bytes / 1_000_000, 2) if sb is not None else 0.0,
                    "text_prompt": self.text_prompts_by_object.get(i),
                    "text_draft": self.text_prompt_drafts_by_object.get(i),
                })
            token_hw_str = f"{self.token_hw[0]} x {self.token_hw[1]}" if self.token_hw else "-"
            return {
                "open": self.is_open,
                "warning": self.last_error,
                "model": {
                    "name": self.model_name,
                    "version": self.model_version,
                    "device": self.model_device,
                    "dtype": self.model_dtype,
                    "tokens": token_hw_str,
                },
                "video": {
                    "path": None if self.cfg["use_webcam"] else self.video_path,
                    "use_webcam": self.cfg["use_webcam"],
                    "fps": self.video_fps,
                    "total_frames": self.total_frames,
                    "frame_idx": self.frame_idx,
                    "hw": list(self.frame_hw) if self.frame_hw else None,
                },
                "ui": {
                    "buffer": self.buffer_select_idx,
                    "mask_idx": self.selected_mask_idx,
                    "is_paused": self.is_paused,
                    "is_reversed": self.is_reversed,
                    "show_preview": self.show_preview,
                    "invert_mask": self.invert_mask,
                    "enable_history": self.enable_history,
                    "is_record_enabled": self.is_record_enabled,
                    "num_buffers": self.num_obj_buffers,
                },
                "score": score,
                "num_prompt_mems": num_prompt_mems,
                "num_prev_mems": num_prev_mems,
                "vram_mb": vram,
                "buffers": buffer_info,
                "config": self.config_snapshot(),
                "has_text_preview": self._text_preview_active(),
                "text_draft": self.text_prompt_drafts_by_object.get(buffer),
                "text_stored": self.text_prompts_by_object.get(buffer),
                "crop": {
                    "enabled": self.cfg["crop"],
                    "tlbr_norm": self.crop_tlbr_norm,
                },
                "use_ffmpeg": self.use_ffmpeg,
                "log": list(self.log)[-12:],
            }

    def _require_open(self):
        if not self.is_open or self.sammodel is None or self.vreader is None:
            raise RuntimeError("No video is open. Use the Open button first.")


# Global session (single-user tool; all access serialized with the session lock)
SESSION = Session()


# ---------------------------------------------------------------------------------------------------------------------
# %% File browser helper


def list_dir(base_path):
    """List subdirectories + video files for the interactive Open dialog."""
    base_path = SESSION._resolve_path(base_path) if base_path else os.getcwd()
    if not osp.isdir(base_path):
        raise FileNotFoundError(f"Not a directory: {base_path}")
    entries = []
    for name in sorted(os.listdir(base_path), key=str.lower):
        if name.startswith("."):
            continue
        full = osp.join(base_path, name)
        if osp.isdir(full):
            entries.append({"name": name, "is_dir": True})
        else:
            ext = osp.splitext(name)[1].lower()
            if ext in VIDEO_EXTS:
                entries.append({"name": name, "is_dir": False})
    parent = osp.dirname(base_path)
    return {
        "path": base_path,
        "parent": parent if parent != base_path else None,
        "entries": entries,
    }


# ---------------------------------------------------------------------------------------------------------------------
# %% HTTP handler


class Handler(BaseHTTPRequestHandler):
    server_version = "MuggledSAMWebUI/1.0"
    session = SESSION

    # --- plumbing -----------------------------------------------------------

    def log_message(self, fmt, *args):  # keep the console clean
        pass

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path, content_type):
        with open(path, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # Never let the browser keep a stale copy of our JS/HTML/CSS, or a
        # plain reload keeps running old client code after we edit it.
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            body = json.loads(raw.decode("utf-8"))
            return body if isinstance(body, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _api_upload_dir(self, name):
        """Create <repo>/videos/<name> for an 'Upload video dir' run."""
        name = _validate_upload_dirname(name)
        dest_root = osp.realpath(UPLOAD_ROOT)
        dest = osp.realpath(osp.join(UPLOAD_ROOT, name))
        if osp.commonpath([dest_root, dest]) != dest_root:
            raise ValueError(f"Invalid upload folder name: {name!r}")
        with _UPLOAD_LOCK:
            os.makedirs(dest, exist_ok=True)
        return dest

    def _api_upload_file(self):
        """Write one raw-body upload to <repo>/videos/<dir>/<relpath>.

        The file is written to a .part temp file and renamed only after all
        Content-Length bytes have been received, so an interrupted upload
        never leaves a corrupt half file behind.
        """
        name = _validate_upload_dirname(self.headers.get("X-Upload-Dir"))
        rel = _validate_upload_relpath(self.headers.get("X-Upload-RelPath"))
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            raise ValueError("Empty upload body")
        dest_root = osp.realpath(UPLOAD_ROOT)
        dest_dir = osp.realpath(osp.join(UPLOAD_ROOT, name))
        if osp.commonpath([dest_root, dest_dir]) != dest_root:
            raise ValueError(f"Invalid upload folder name: {name!r}")
        target = osp.realpath(osp.join(dest_dir, rel))
        if osp.commonpath([dest_dir, target]) != dest_dir:
            raise ValueError(f"Invalid upload file path: {rel!r}")
        tmp = target + ".part"
        with _UPLOAD_LOCK:
            os.makedirs(osp.dirname(target), exist_ok=True)
            try:
                with open(tmp, "wb") as fh:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
                if remaining > 0:
                    raise IOError("Upload interrupted: incomplete body")
                os.replace(tmp, target)
            except Exception:
                if osp.exists(tmp):
                    os.remove(tmp)
                raise
        self._log_upload(f"Uploaded videos/{name}/{rel} ({length} bytes)")
        return {"ok": True, "path": target, "size": length}

    def _log_upload(self, msg):
        try:
            self.session._log(msg)
        except Exception:  # noqa: BLE001
            pass

    def _handle(self, method):
        try:
            parsed = urlparse(self.path)
            route = parsed.path if len(parsed.path) > 1 else "/"
            qs = parse_qs(parsed.query)
            # /api/upload_file carries a raw binary body; read it in the route
            body = None
            if method == "POST":
                if route != "/api/upload_file":
                    body = self._read_body()

            if method == "GET":
                self._route_get(route, qs)
            elif method == "POST":
                self._route_post(route, body)
            else:
                self._send_json({"ok": False, "error": "Method not allowed"}, status=405)
            self.session.last_error = None
        except BrokenPipeError:
            pass
        except Exception as err:  # noqa: BLE001
            traceback.print_exc()
            self.session.last_error = str(err)
            try:
                self._send_json({"ok": False, "error": str(err)}, status=500)
            except Exception:  # noqa: BLE001
                pass

    # --- GET routes ---------------------------------------------------------

    def _route_get(self, route, qs):
        static_dir = osp.join(osp.dirname(osp.abspath(__file__)), "static")
        if route == "/":
            return self._send_file(osp.join(static_dir, "index.html"), "text/html; charset=utf-8")
        if route == "/style.css":
            return self._send_file(osp.join(static_dir, "style.css"), "text/css; charset=utf-8")
        if route == "/app.js":
            return self._send_file(osp.join(static_dir, "app.js"), "application/javascript; charset=utf-8")
        if route == "/api/health":
            return self._send_json({"ok": True})
        if route == "/api/status":
            return self._send_json(self.session.status_dict())
        if route == "/api/dirs":
            path = (qs.get("path") or [os.getcwd()])[0]
            return self._send_json({"ok": True, **list_dir(path)})
        if route == "/api/frame":
            advance = (qs.get("advance") or ["0"])[0] in ("1", "true", "True")
            return self._send_json(self.session.display(advance=advance))
        self._send_json({"ok": False, "error": f"No such route: {route}"}, status=404)

    # --- POST routes --------------------------------------------------------

    def _route_post(self, route, body):
        s = self.session
        if route == "/api/config":
            cfg, need_reopen = s.set_config(body)
            return self._send_json({"ok": True, "config": cfg, "need_reopen": need_reopen})
        if route == "/api/open":
            status = s.open(video_path=body.get("path"), use_webcam=body.get("webcam"))
            return self._send_json(status)
        if route == "/api/close":
            s.close()
            return self._send_json({"ok": True})
        if route == "/api/seek":
            return self._send_json(s.seek(body.get("frame", 0)))
        if route == "/api/step":
            return self._send_json(s.step(body.get("n", 1)))
        if route == "/api/play":
            return self._send_json(s.set_play(body.get("playing", True)))
        if route == "/api/reverse":
            return self._send_json(s.set_reverse(body.get("reversed", True)))
        if route == "/api/prompt":
            return self._send_json(s.add_prompt(
                body.get("type"), x=body.get("x"), y=body.get("y"),
                x2=body.get("x2"), y2=body.get("y2"),
            ))
        if route == "/api/set_prompts":
            return self._send_json(s.set_prompts(
                body.get("boxes"), body.get("fg"), body.get("bg"),
            ))
        if route == "/api/select_mask":
            return self._send_json(s.select_mask(body.get("idx", 1)))
        if route == "/api/store_prompt":
            return self._send_json(s.store_prompt(buffer=body.get("buffer")))
        if route == "/api/undo_prompt":
            return self._send_json(s.undo_prompt(buffer=body.get("buffer")))
        if route == "/api/clear_prompts":
            return self._send_json(s.clear_prompts(buffer=body.get("buffer")))
        if route == "/api/clear_history":
            return self._send_json(s.clear_history(buffer=body.get("buffer")))
        if route == "/api/select_buffer":
            return self._send_json(s.select_buffer(body.get("idx", 0)))
        if route == "/api/toggle":
            return self._send_json(s.toggle(body.get("name")))
        if route == "/api/text_prompt":
            return self._send_json(s.set_text_prompt(body.get("text"), buffer=body.get("buffer")))
        if route == "/api/reuse_text_prompt":
            return self._send_json(s.reuse_text_prompt(buffer=body.get("buffer")))
        if route == "/api/save_buffer":
            return self._send_json(s.save_buffer(buffer=body.get("buffer")))
        if route == "/api/clear_buffer":
            return self._send_json(s.clear_buffer(buffer=body.get("buffer")))
        if route == "/api/save_state":
            return self._send_json(s.save_state())
        if route == "/api/crop":
            return self._send_json(s.set_crop(body.get("tlbr")))
        if route == "/api/upload_dir":
            dest = self._api_upload_dir(body.get("name"))
            return self._send_json({"ok": True, "path": dest})
        if route == "/api/upload_file":
            return self._send_json(self._api_upload_file())
        self._send_json({"ok": False, "error": f"No such route: {route}"}, status=404)

    # --- verb entrypoints ----------------------------------------------------

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


# ---------------------------------------------------------------------------------------------------------------------
# %% Main


def _ensure_self_signed(cert_path, key_path):
    """Create (or reuse) a self-signed cert via the openssl CLI so the page can
    be served over HTTPS. Chrome only exposes the fast directory picker
    (showDirectoryPicker) in a secure context; over plain http the
    webkitdirectory fallback is used instead, and that one scans every
    subfolder of the picked folder - which hangs on large trees."""
    import subprocess
    os.makedirs(osp.dirname(cert_path), exist_ok=True)
    if osp.exists(cert_path) and osp.exists(key_path):
        return
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", key_path, "-out", cert_path, "-days", "3650",
        "-subj", "/CN=muggled-sam-webui",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description="Web UI for save_prompts_run_video.py")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to serve on (default: {DEFAULT_PORT})")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                        help=f"Host to bind to (default: {DEFAULT_HOST}, i.e. any IP)")
    parser.add_argument("--https", action="store_true",
                        help="Serve over HTTPS with an auto-generated self-signed cert. "
                             "Needed for Chrome's fast folder picker when connecting "
                             "from another machine over plain http.")
    args = parser.parse_args()

    # Run from the repo root so relative paths (./model_weights, ./tracking_states)
    # behave exactly like in the original script
    os.chdir(_REPO_ROOT)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    scheme = "http"
    if args.https:
        import ssl
        cert_dir = osp.join(_REPO_ROOT, ".webui_ssl")
        cert_path = osp.join(cert_dir, "cert.pem")
        key_path = osp.join(cert_dir, "key.pem")
        _ensure_self_signed(cert_path, key_path)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert_path, key_path)
        server.socket = ssl_ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"

    display_host = "0.0.0.0" if args.host == "0.0.0.0" else args.host
    print(f"Web UI running at {scheme}://{display_host}:{args.port}/  (any IP can connect)")
    if scheme == "https":
        print("  Self-signed cert: browsers show a warning page once - accept it to continue.")
    print(f"  Repo root: {_REPO_ROOT}")
    print(f"  Tracking state saves go to: {STATE_SAVE_DIR}/<video_stem>.pt")
    print(f"  Uploaded video folders go to: {UPLOAD_ROOT}/<folder_name>/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
