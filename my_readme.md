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

After finish setting the prompts, press `q` or `g` to close the user interface, the prompts will be saved as `./saved_tracking_state.pt`.

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

# TODO task

Enable text prompt like this, and the text prompt can be saved and loaded in above commands.
```
python save_prompts_run_video.py --input_video input_video.mp4 --text_prompt 'garments and hat'
```

`--text_prompt` requires SAM3 weights. It finds matching objects in the first
frame, seeds the video tracker with the highest-scoring matches, and stores the
text alongside the tracking state. `load_prompts_run_video.py` and
`load_prompts_run_dir.py` load that state as usual; the latter delegates to the
former for every video.
