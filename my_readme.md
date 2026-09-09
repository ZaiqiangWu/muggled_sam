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
Changing frames discards unconfirmed candidates, but retains the entered text;
click **Reuse Text Prompt** on the new frame to generate fresh candidates.

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

## Codex 远程 SAM3 工作流（Mac → Tailscale → HPC）

Mac 上的 Codex 通过 `http://100.120.152.79:8765` 直连 SAM3 MCP 服务（MCP URL：`http://100.120.152.79:8765/mcp`）。文件传输规则：

1. 不要使用 SSH / ProxyJump / SCP / SFTP 传输输入或输出文件。
2. 不要把视频 base64 编码进 MCP 参数，不要让视频字节进入 LLM 上下文。
3. Mac 本地视频：`curl -F "file=@/path/to/video.mp4" http://100.120.152.79:8765/upload`。
4. 使用返回 JSON 的 `path`（相对 input-root，如 `<file_id>/video.mp4`）。
5. `segment_video(video_path=<该 path>, ...)`，其余参数见 MCP_README.md。
6. `segment_video` 立即返回 `job_id`（异步执行，GPU 队列串行）：先轮询 `get_job_status` 看 `queued/running/completed/failed` 与帧进度，再 `get_job_result` 取产物路径，`get_segmentation` 查看异常与重试历史；查看全部预览帧后 `submit_visual_review`，必要时 `rerun_segmentation`。
7. 完成后 `package_result(job_id)` 得到 work-root 相对路径 `download`。
8. `curl -o <本地目录>/result.tar.gz "http://100.120.152.79:8765/download/<download>"` 下载到用户指定目录。

已经位于 HPC input-root（共享 Lustre）中的文件无需上传，直接传相对或绝对路径。单个产物用 `GET /files/info/{path}` 查大小、`GET /download/{path}` 下载（支持 Range 断点续传）。详见 MCP_README.md。
