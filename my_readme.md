# Instruction

## Install libraries

```
conda create -n sam3 python=3.12
conda activate sam3
```
Install torch and torchvision according to your CUDA version
```
pip install \
    "torch>=2.7,<2.11" \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121
```
```
pip install -r requirements_my.txt
```

## How to segment a video

### Generate prompts interactively
```
python save_prompts_run_video.py --input_video input_video.mp4
```
A window will pop up and then youcan set the points and bbox prompts interactively.
You can use Shift + click to place multiple points.
You can also set text prompt for each object.

After finish setting the prompts, press `q` or `g` to close the user interface, the prompts will be saved as `./saved_tracking_state.pt`.

For SAM3 text prompting, select a Buffer, pause the video, click **Set Text Prompt**,
and type in the video window. Repeating this on later frames for the same Buffer
adds another text-derived tracking prompt frame; it does not replace the earlier one.
To use the same phrase on a later frame without typing it again, select that
Buffer, pause on the new frame, and click **Reuse Text Prompt**.
Text prompts first show up to four detection candidates. Select the desired
candidate in the preview UI, then click **Store Prompt**; only this final step
adds the selected text candidate to the tracking memory.

### Perform video segmentation on a headless PC
You can load the saved prompts and perform segmentation to a video on a headless PC.
```
python load_prompts_run_video.py --prompt_path ./saved_tracking_state.pt --input_video ./videos/spacesuit/spacesuit_01.mp4
```

You can also load the saved prompts and perform segmentation to videos in a director:
```
python load_prompts_run_dir.py --prompt_path ./tracking_states/quilted_jacket.pt --input_dir ./videos/quilted_jacket/
```

### Check the segmentation results
You can execute this command to generate segmentation results for visually checking the results:
```
python check_generated_masks.py
```
