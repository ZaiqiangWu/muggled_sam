#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
import os.path as osp
from time import perf_counter
from enum import Enum
from dataclasses import dataclass

import torch
import cv2
import numpy as np

from muggled_sam.make_sam import make_sam_from_state_dict

from muggled_sam.demo_helpers.ui.window import DisplayWindow, KEY
from muggled_sam.demo_helpers.ui.video import (
    ReversibleLoopingVideoReader,
    LoopingVideoPlaybackSlider,
    ValueChangeTracker,
)
from muggled_sam.demo_helpers.ui.layout import HStack, VStack
from muggled_sam.demo_helpers.ui.buttons import ToggleButton, ImmediateButton, RadioConstraint
from muggled_sam.demo_helpers.ui.static import StaticMessageBar
from muggled_sam.demo_helpers.ui.text import ValueBlock, TextBlock
from muggled_sam.demo_helpers.ui.base import force_same_min_width
from muggled_sam.demo_helpers.ui.overlays import DrawPolygonsOverlay
from muggled_sam.demo_helpers.ui.helpers.images import FrameCompositing

from muggled_sam.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI
from muggled_sam.demo_helpers.crop_ui import run_crop_ui

from muggled_sam.demo_helpers.misc import PeriodicVRAMReport, make_device_config, get_default_device_string
from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from muggled_sam.demo_helpers.contours import get_contours_from_mask
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults,SAMVideoBuffer

from muggled_sam.demo_helpers.saving import save_video_frames, get_save_name
from muggled_sam.demo_helpers.ffmpeg import get_default_ffmpeg_command, verify_ffmpeg_path, save_video_stream


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_prompts_path = None
default_display_size = 900
default_base_size = None
default_max_memory_history = 6
default_max_pointer_history = 15
default_num_object_buffers = 4
default_object_score_threshold = 0.0
default_bg_color_hex = "ff00ff00"
default_ffmpeg = get_default_ffmpeg_command()

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to run Segment-Anything-V2 (SAMv2) on a video")
parser.add_argument("-i", "--image_path", default=default_image_path, help="Path to input image")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to SAM model weights")
parser.add_argument(
    "-s",
    "--display_size",
    default=default_display_size,
    type=int,
    help=f"Controls size of displayed results (default: {default_display_size})",
)
parser.add_argument(
    "-d",
    "--device",
    default=default_device,
    type=str,
    help=f"Device to use when running model, such as 'cpu' (default: {default_device})",
)
parser.add_argument(
    "-f32",
    "--use_float32",
    default=False,
    action="store_true",
    help="Use 32-bit floating point model weights. Note: this doubles VRAM usage",
)
parser.add_argument(
    "-ar",
    "--use_aspect_ratio",
    default=True,
    action="store_true",
    help="Process the video at it's original aspect ratio",
)
parser.add_argument(
    "-b",
    "--base_size_px",
    default=default_base_size,
    type=int,
    help="Set image processing size (will use model default if not set)",
)
parser.add_argument(
    "-n",
    "--num_buffers",
    default=default_num_object_buffers,
    type=int,
    help=f"Number of object buffers in the saving/recording UI (default {default_num_object_buffers})",
)
parser.add_argument(
    "--max_memories",
    default=default_max_memory_history,
    type=int,
    help=f"Maximum number of previous-frame memory encodings to store (default: {default_max_memory_history})",
)
parser.add_argument(
    "--max_pointers",
    default=default_max_pointer_history,
    type=int,
    help=f"Maximum number of previous-frame object pointers to store (default: {default_max_pointer_history})",
)
parser.add_argument(
    "--keep_bad_objscores",
    default=False,
    action="store_true",
    help="If set, masks associated with low object-scores will NOT be discarded",
)
parser.add_argument(
    "--keep_history_on_new_prompts",
    default=True,
    action="store_true",
    help="If set, existing history data will not be cleared when adding new prompts",
)
parser.add_argument(
    "--objscore_threshold",
    default=default_object_score_threshold,
    type=float,
    help=f"Threshold below which objects are considered to be 'lost' (default: {default_object_score_threshold})",
)
parser.add_argument(
    "--hide_info",
    default=False,
    action="store_true",
    help="Hide text info elements from UI",
)
parser.add_argument(
    "-cam",
    "--use_webcam",
    default=False,
    action="store_true",
    help="Use a webcam as the video input, instead of a file",
)
parser.add_argument(
    "-nosave",
    "--disable_save",
    default=False,
    action="store_true",
    help="If set, this simplifies the UI by hiding the element associated with saving",
)
parser.add_argument(
    "--ffmpeg",
    nargs="?",
    const=default_ffmpeg,
    default=None,
    type=str,
    help=f"Save as mp4 (using FFmpeg). Can optionally provide a path to executable (default: {default_ffmpeg})",
)
parser.add_argument(
    "--crop",
    default=False,
    action="store_true",
    help="If set, a cropping UI will appear on start-up to allow for the image to be cropped prior to processing",
)
parser.add_argument(
    "-bg",
    "--bg_color_hex",
    default=default_bg_color_hex,
    type=str,
    help=f"Color of segmented regions, given as RGB or RGBA hex code (default: {default_bg_color_hex})",
)

# For convenience
args = parser.parse_args()
enable_saving = not args.disable_save
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
num_obj_buffers = args.num_buffers if enable_saving else 1
max_memory_history = args.max_memories
max_pointer_history = args.max_pointers
discard_on_bad_objscore = not args.keep_bad_objscores
clear_history_on_new_prompts = not args.keep_history_on_new_prompts
object_score_threshold = args.objscore_threshold
show_info = not args.hide_info
use_webcam = args.use_webcam
enable_crop_ui = args.crop
bg_color_hex = args.bg_color_hex
arg_ffmpeg = args.ffmpeg

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_vidpath = history.read("video_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
video_path = ask_for_path_if_missing(arg_image_path, "video", history_vidpath) if not use_webcam else 0
model_path = ask_for_model_path_if_missing(__file__, arg_model_path, history_modelpath)

# Store history for use on reload (but don't save video path when using webcam)
if use_webcam:
    history.store(model_path=model_path)
else:
    history.store(video_path=video_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Check that we can call ffmpeg if flag was given
use_ffmpeg, ffmpeg_path = verify_ffmpeg_path(arg_ffmpeg)
if use_ffmpeg:
    print("", "Found valid FFmpeg path:", f"@ {ffmpeg_path}", sep="\n", flush=True)

# Set up masking/chroma-keying for saving frames (do this early so we fail on bad inputs before loading bigger files!)
mask_color_bgra = FrameCompositing.parse_hex_color(bg_color_hex)
save_masking = FrameCompositing(mask_color_bgra)

# Set up shared image encoder settings (needs to be consistent across image/video frame encodings)
imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": use_square_sizing}

# Get the model name, for reporting
model_name = osp.basename(model_path)

print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
assert sammodel.name in ("samv2", "samv3"), "Only SAMv2/v3 models are supported for video predictions!"
sammodel.to(**device_config_dict)

# Set up access to video
vreader = ReversibleLoopingVideoReader(video_path).release()
video_fps = vreader.get_fps()
sample_frame = vreader.get_sample_frame()
total_frames = vreader.total_frames
if enable_crop_ui:
    print("", "Cropping enabled: Adjust box to select image area for further processing", sep="\n", flush=True)
    _, history_crop_tlbr = history.read("crop_tlbr_norm")
    yx_crop_slice, crop_tlbr_norm = run_crop_ui(sample_frame, display_size_px, history_crop_tlbr)
    sample_frame = sample_frame[yx_crop_slice]
    history.store(crop_tlbr_norm=crop_tlbr_norm)

# Initial model run to make sure everything succeeds
print("", "Encoding image data...", sep="\n", flush=True)
t1 = perf_counter()
encoded_img, token_hw, preencode_img_hw = sammodel.encode_image(sample_frame, **imgenc_config_dict)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
time_taken_ms = round(1000 * (t2 - t1))
print(f"  -> Took {time_taken_ms} ms", flush=True)

# Run model without prompts as sanity check. Also gives initial result values
prompts = ([], [], [])
encoded_prompts = sammodel.encode_prompts(*prompts)
init_mask_preds, _ = sammodel.generate_masks(encoded_img, encoded_prompts, blank_promptless_output=False)
prediction_hw = init_mask_preds.shape[2:]
# mask_uint8 = ((mask_preds[:, 0, :, :] > 0.0) * 255).byte().cpu().numpy().squeeze()

# Provide some feedback about how the model is running
model_device = device_config_dict["device"]
model_dtype = str(device_config_dict["dtype"]).split(".")[-1]
image_hw_str = f"{preencode_img_hw[0]} x {preencode_img_hw[1]}"
token_hw_str = f"{token_hw[0]} x {token_hw[1]}"
print(
    "",
    f"Config ({model_name}):",
    f"  Device: {model_device} ({model_dtype})",
    f"  Resolution HW: {image_hw_str}",
    f"  Tokens HW: {token_hw_str}",
    sep="\n",
    flush=True,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Helper Data types


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

    png_per_frame_dict: dict[int, np.ndarray]
    bytes_per_frame_dict: dict[int, int]
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


def save_segmentation_results(
    video_path: str,
    video_fps: float,
    buffer_index: int,
    save_frames_dict: dict,
    ffmpeg_path: str | None,
    use_ffmpeg: bool,
) -> bool:
    """
    Wrapper around saving tarfiles vs. video (with ffmpeg)
    Returns updated 'use_ffmpeg' which may be toggled false if video encoding fails!
    """

    # Build save pathing
    save_folder, save_idx = get_save_name(video_path, "run_video")
    min_fidx, max_fidx = min(save_frames_dict.keys()), max(save_frames_dict.keys())
    save_name = f"{save_idx}_obj{1+buffer_index}_{min_fidx}_to_{max_fidx}_frames"
    save_path_no_ext = osp.join(save_folder, save_name)

    # Save with ffmpeg
    if use_ffmpeg:
        print("", "Encoding video...", sep="\n", flush=True)
        ok_save, save_path = save_video_stream(ffmpeg_path, save_path_no_ext, video_fps, save_frames_dict)
        if ok_save:
            print(f"@ {save_path}")
        else:
            print("-> Error saving with FFmpeg, will fall back to saving tarfile")
            use_ffmpeg = False

    # Not using 'else' here, because encoding may have failed
    # -> This makes ffmpeg failure fallback to saving pngs
    if not use_ffmpeg:
        print("", f"Saving frame data ({num_frames} frames)...", sep="\n", flush=True)
        save_file_path = save_video_frames(save_path_no_ext, save_frames_dict)
        print(f"@ {save_file_path}")

    return use_ffmpeg

def move_memory_to_device(memory_list, device):
    """
    Move all internal tensors in memory_list to the given device.
    """
    for mem in memory_list:
        # move stored embeddings
        if hasattr(mem, "mem_encs"):
            mem.mem_encs = [t.to(device) for t in mem.mem_encs]

        # move pointers
        if hasattr(mem, "obj_ptrs"):
            mem.obj_ptrs = [t.to(device) for t in mem.obj_ptrs]

        # move mask results if stored as tensors
        if hasattr(mem, "mask_preds"):
            mem.mask_preds = [t.to(device) for t in mem.mask_preds]
# ---------------------------------------------------------------------------------------------------------------------
# %% Set up UI

# Playback control UI for adjusting video position
playback_slider = LoopingVideoPlaybackSlider(vreader, stay_paused_on_change=True)

# Set up shared UI elements & control logic
ui_elems = PromptUI(sample_frame, init_mask_preds, 2)
uictrl = PromptUIControl(ui_elems, mask_color_bgra)

# Add extra polygon drawer for unselected objects
unselected_olay = DrawPolygonsOverlay((50, 5, 130), (25, 0, 60))
ui_elems.overlay_img.add_overlays(unselected_olay)

# Set up text-based reporting UI
vram_text = ValueBlock("VRAM: ", "-", "MB", max_characters=5)
objscore_text = ValueBlock("Score: ", None, max_characters=3)
num_prompts_text = ValueBlock("Prompts: ", "0", max_characters=2)
num_history_text = ValueBlock("History: ", "0", max_characters=2)
frame_idx_text = ValueBlock("Frame: ", "0", max_characters=6)
#force_same_min_width(vram_text, objscore_text)
force_same_min_width(vram_text, objscore_text, frame_idx_text)

# Set up button controls
show_preview_btn = ToggleButton("Preview", default_state=False)
invert_mask_btn = ToggleButton("Invert", default_state=False)
track_btn = ToggleButton("Track", on_color=(30, 140, 30))
reversal_btn = ToggleButton("Reverse", default_state=False, text_scale=0.35)
store_prompt_btn = ImmediateButton("Store Prompt", text_scale=0.35, color=(145, 160, 40))
clear_prompts_btn = ImmediateButton("Clear Prompts", text_scale=0.35, color=(80, 110, 230))
enable_history_btn = ToggleButton("Enable History", default_state=True, text_scale=0.35, on_color=(90, 85, 115))
clear_history_btn = ImmediateButton("Clear History", text_scale=0.35, color=(130, 60, 90))
force_same_min_width(store_prompt_btn, clear_prompts_btn, enable_history_btn, clear_history_btn)


# Create save UI
enable_record_btn = ToggleButton("Enable Recording", default_state=False, on_color=(0, 15, 255), button_height=60)
buffer_btns_list = []
buffer_text_list = []
buffer_elems = []
for objidx in range(num_obj_buffers):
    buffer_btn = ToggleButton(f"Buffer {1+objidx}", button_height=20, text_scale=0.5, on_color=(145, 120, 65))
    buffer_txt = TextBlock(0.0, block_height=25, text_scale=0.35, max_characters=3)
    buffer_elems.extend([HStack(buffer_btn, buffer_txt)])
    buffer_btns_list.append(buffer_btn)
    buffer_text_list.append(buffer_txt)
force_same_min_width(*buffer_btns_list)
buffer_btn_constraint = RadioConstraint(*buffer_btns_list)
buffer_title_text = TextBlock("Buffered Mask Data (MB)", block_height=20, text_scale=0.35)
buffer_save_btn = ImmediateButton("Save Buffer", button_height=30, text_scale=0.5, color=(110, 145, 65))
buffer_clear_btn = ImmediateButton("Clear Buffer", button_height=30, text_scale=0.5, color=(80, 60, 190))
save_sidebar = VStack(enable_record_btn, buffer_title_text, *buffer_elems, buffer_save_btn, buffer_clear_btn)

# Set up info bars
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str, space_equally=True)
footer_msgbar = StaticMessageBar(
    "[tab] Store Prompt",
    "[v/b] Buffers" if enable_saving else "[i] Invert",
    "[space] Play/Pause",
    "[p] Preview",
    text_scale=0.35,
    space_equally=True,
    bar_height=30,
)

# Set up full display layout
disp_layout = VStack(
    header_msgbar if show_info else None,
    HStack(ui_elems.layout, save_sidebar) if enable_saving else ui_elems.layout,
    playback_slider if not use_webcam else None,
    #HStack(vram_text, objscore_text),
    HStack(vram_text, objscore_text, frame_idx_text),
    HStack(num_prompts_text, track_btn, num_history_text),
    HStack(store_prompt_btn, clear_prompts_btn, reversal_btn, enable_history_btn, clear_history_btn),
    footer_msgbar if show_info else None,
).set_debug_name("DisplayLayout")

# Render out an image with a target size, to figure out which side we should limit when rendering
display_image = disp_layout.render(h=display_size_px, w=display_size_px)
render_side = "h" if display_image.shape[1] > display_image.shape[0] else "w"
render_limit_dict = {render_side: display_size_px}
min_display_size_px = disp_layout._rdr.limits.min_h if render_side == "h" else disp_layout._rdr.limits.min_w


# ---------------------------------------------------------------------------------------------------------------------
# %% Video loop
# Set up per-object storage for masking/saving results
objiter = list(range(num_obj_buffers))
maskresults_list = [MaskResults.create(init_mask_preds) for _ in objiter]
savebuffers_list = [SaveBufferData.create() for _ in objiter]
memory_list = [
    SAMVideoObjectResults.create(max_memory_history, max_pointer_history, prompt_history_length=32) for _ in objiter
]


vreader = ReversibleLoopingVideoReader(video_path).release()
vreader.pause(False)

load_path = "saved_tracking_state.pt"
# ✅ Allowlist both classes
from collections import deque
torch.serialization.add_safe_globals([SAMVideoObjectResults, SAMVideoBuffer,deque])

loaded_data = torch.load("saved_tracking_state.pt", map_location="cpu", weights_only=False)

memory_list = loaded_data["memory_list"]

# Move all tensors inside memory_list to GPU
device = device_config_dict["device"]
move_memory_to_device(memory_list, device)

for objidx, mem in loaded_data.items():
    memory_list[objidx] = mem

if not any(mem is not None and mem.check_has_prompts() for mem in memory_list):
    print("No valid prompts found. Exiting.")
    exit(0)



from tqdm import tqdm
pbar = tqdm(total=total_frames)
# Check if ANY object has prompts

# Tracking without UI
prev_real_idx = -1
with torch.inference_mode():
    for is_paused, frame_idx, frame in vreader:
        #print(frame_idx)

        real_frame_idx = frame_idx - 1

        if real_frame_idx < 0:
            continue

        if real_frame_idx < prev_real_idx:
            break

        prev_real_idx = real_frame_idx

        # Encode frame
        encoded_img, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)

        for objidx in objiter:

            if not memory_list[objidx].check_has_prompts():
                continue

            obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                    encoded_img, **memory_list[objidx].to_dict()
                )

            tracked_mask_idx = int(best_mask_idx.squeeze().cpu())
            maskresults_list[objidx].update(mask_preds, tracked_mask_idx, obj_score)

            obj_score = float(obj_score.squeeze().cpu().float().numpy())
            tracked_mask_idx = int(best_mask_idx.squeeze().cpu())

            # Store memory
            memory_list[objidx].store_frame_result(frame_idx, mem_enc, obj_ptr)

            # Save mask
            save_mask = uictrl.create_hires_mask_uint8(mask_preds, tracked_mask_idx, frame.shape[:2])
            save_frame = save_masking.mask_frame(frame, save_mask)

            ok, png = cv2.imencode(".png", save_frame)
            if ok:
                savebuffers_list[objidx].png_per_frame_dict[real_frame_idx] = png

        pbar.update(1)

pbar.close()

# ---------------------------------------------------------------------------------------------------------------------
# %% Final clean up

# Save any buffered frame data
for objidx, savebuffer in enumerate(savebuffers_list):
    png_per_frame_dict = savebuffer.png_per_frame_dict
    num_frames = len(png_per_frame_dict.keys())
    if num_frames > 0:
        use_ffmpeg = save_segmentation_results(
            video_path, video_fps, objidx, png_per_frame_dict, ffmpeg_path, use_ffmpeg
        )
    pass


