# SAM3 远程 MCP 服务

工作流：`Mac/Ubuntu video → MCP → SAM3 → RGBA masks → anomaly detection → preview → Codex visual inspection → optional retry → RGBA PNG + validation MP4`。

本实现复用 `my_readme.md` 对应代码中的 `make_sam_from_state_dict`、`make_detector_model`、`initialize_from_mask`、`SAMVideoObjectResults`、`step_video_masking` 和 `make_hires_mask_uint8`。不调用会创建 UI 的顶层脚本，不加载外部 prompt pickle。单个文本目标；同一文本匹配多个对象时，通过置信度排序的 `candidate_rank=0..3` 选择实例，然后必须目视确认身份。默认使用与交互脚本相同的非空候选过滤和 2048 square encoding。

## Ubuntu HPC 启动

先按 `my_readme.md` 建立 Python 3.12 / SAM3 环境，按集群 CUDA 实际版本安装兼容的 torch / torchvision，再安装：

```sh
conda activate sam3
pip install -r requirements_my.txt -r requirements_mcp.txt
mkdir -p sam3-input sam3-output
python mcp_server.py --weights ./model_weights/sam3.pt \
  --input-root sam3-input \
  --work-root sam3-output --host 0.0.0.0 --port 8765
```

请在调度器分配的 GPU compute node 内运行，勿占用 login node GPU。默认 CUDA / 项目原有 dtype 配置，可传 `--float32` 或 `--device cpu`；后者极慢。模型与 detector 在启动时加载一次，全部推理由同一个后台线程串行执行；只运行一个服务进程，不要配置多个 Web workers 共用输出目录/GPU。启动需等待权重加载完成。输入根目录可重复指定，`POST /upload` 写入第一个 `--input-root`，`--max-upload-bytes` 控制上传上限（默认 64 GiB）。输出目录必须是 work root 下的新目录。

OpenCV 必须支持 MP4V 编码；生成的是无音轨、恒定 FPS 的检查视频，帧顺序来自 OpenCV 解码，原视频音频/VFR 时间戳不保留。RGBA PNG 保留原尺寸和 RGB，alpha 为最终二值 mask（0/255），空 mask 全透明。

## Mac 连接与文件传输

服务通过 Tailscale 私有网络直连，不需要 SSH 隧道（HTTP 本身无账号认证，访问范围由 Tailscale ACL/防火墙控制）：

```sh
python mcp_server.py --weights ./model_weights/sam3.pt \
  --input-root ./sam3-input --work-root ./sam3-output \
  --host 0.0.0.0 --port 8765
```

Mac 的 MCP URL 为 `http://100.120.152.79:8765/mcp`，两端需能通过 Tailscale 互访。
如需监听所有 IPv4 网卡，使用 `--host 0.0.0.0`，无需额外参数。
服务接受任意 HTTP Host/Origin，已关闭 DNS rebinding 检查，并移除 `--allowed-host` 参数。
默认监听地址仍为 `127.0.0.1`。

将 [examples/codex_mcp.toml](examples/codex_mcp.toml) 合并到 Mac 的 `~/.codex/config.toml`。配置依据 [Codex 官方 MCP 文档](https://developers.openai.com/codex/mcp)；服务使用 [官方 Python MCP SDK v1](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x) 的 Streamable HTTP。依赖限制 `<2`，避免 v2 接口变化。

大文件走与 MCP 同源的 HTTP 数据面（流式传输，数 GB 级）；**不要** SSH/SCP/SFTP，**不要**把视频 base64 编码进 MCP JSON：

- `POST /upload`：`multipart/form-data` 字段名 `file`，流式写入 `<input-root>/<file_id>/<filename>`（临时文件 + 原子 rename），返回 `{"ok":true,"file_id":"...","filename":"...","size":...,"path":"<file_id>/<filename>","sha256":"..."}`。
- `GET /download/{path}`：从 work-root 流式下载，支持 `Range` 断点续传，例 `GET /download/JOB/result.tar.gz`。
- `GET /files/info/{path}?root=input|work`：返回 `{"exists":true,"size":...,"filename":"..."}`。

所有路径 canonicalize 后必须仍位于对应 root 内：`/upload` 只能写 input-root，`/download` 只能读 work-root，`/files/info` 可读 work-root（`root=input` 时读 input-root），`../` 越界返回 403。

Mac 视频上传示例：

```sh
curl --fail --show-error \
  -F "file=@/Users/ME/Videos/input.mp4" \
  http://100.120.152.79:8765/upload
```

把返回 JSON 的 `path`（相对 input-root）直接作为 `segment_video` 的 `video_path`，`input_location` 保持默认 `"ubuntu"`。已经位于 HPC input-root（共享 Lustre）中的文件无需重新上传，直接传相对或绝对路径。`input_location="mac"` 会明确要求先上传。`output_location="mac"` 表示需要打包下载，`output_dir` 始终是 Ubuntu staging 路径，服务不会声称已写入 Mac。

`mcp_transfer.py`（SFTP 辅助脚本）保留为 HTTP 不可达时的可选回退，正常工作流不再需要 SSH。

## 完整调用示例

以下是 Codex 调用 MCP tools 的参数示例，`JOB` 替换为返回的 job_id，`video_path` 用 `POST /upload` 返回的 `path`（或 input-root 内已有文件的相对/绝对路径）：

1. `segment_video`：

```json
{"video_path":"UPLOADED_FILE_ID/input.mp4","text_prompt":"quilted jacket","output_location":"mac","top_k":8,"random_check_frames":4,"max_retry":2,"prompt_frames":[0],"candidate_rank":0}
```

2. 调用 `get_segmentation({"job_id":"JOB"})`，间隔数秒查询，直到 `awaiting_visual_review` 或 `failed`。分割后台运行，HTTP 请求不用等待整个视频；返回任务 ID、状态、各次尝试配置、异常帧与分数、文件路径及重试历史。每一帧的完整指标位于 `analysis.json`。

3. 对本次 attempt 的每个 `preview_frames` 调用 `get_preview({"job_id":"JOB","attempt_index":0,"frame":42})`。返回真实 MCP image，左边原图、右边 overlay，不要求 Mac 读取 Ubuntu 路径。检查衣服遗漏、皮肤/头发/背景误分、身份漂移、消失和边界；必要时下载并查看完整 MP4。服务要求所有预览都已取回，视觉结论仍由 Codex 负责，不能用数值通过代替视觉审核。

4. 若有问题，提交：

```json
{"job_id":"JOB","attempt_index":0,"passed":false,"inspected_frames":[0,42,99],"issues":[{"frame":42,"reason":"mask includes hair"}],"notes":"已检查全部预览，42 帧误分头发"}
```

`inspected_frames` 须替换为**实际查看的全部 preview_frames**，示例帧号不能照搬。然后调用 `rerun_segmentation`：

```json
{"job_id":"JOB","reason":"42 帧开始误分头发，缩小文本范围并从之前的帧重新初始化","text_prompt":"the quilted fabric jacket","prompt_frames":[35,42,50],"frame_range":[35,60],"candidate_rank":0}
```

范围为零起始、两端包含的帧号，必须存在于视频中。范围起始帧自动加入 key frames；范围外直接复制当前最佳 attempt 的 RGBA，不运行模型。区间内重新初始化 tracking；之前 prompt memory 不复用，每个新 key frame 保留当前区间已有 prompt memory、清空滚动历史，与交互脚本一致。局部修正版本是混合来源：`parent_attempt`、`frame_range`、各 attempt 的 config 记录其来源，顶层 final tracking 指最近一次被选中版本的配置。省略范围则重跑完整视频，可增加异常前后的 key frames。每次仍对全视频分析并生成新 MP4/预览，包括区间接缝。

5. 重复查询、查看新预览、提交审核。通过时 `passed=true, issues=[]`，notes 说明检查结论；若有数值报警但属于合理运动/遮挡，应解释这些报警。达到 `max_retry`（不含首次运行）后不再允许重试，返回 `retry_exhausted`、`quality_passed=false`、当前最佳产物及仍存在的问题。当前最佳优先采用视觉通过版本；未通过时按视觉问题数及平均异常分数排序。这是启发式选择，不是语义正确性保证。

6. 调用 `package_result({"job_id":"JOB"})` 得到 `download`（work-root 相对路径，如 `JOB/result.tar.gz`），然后在 Mac 执行：

```sh
curl --fail --show-error \
  -o /Users/ME/Downloads/sam3-result.tar.gz \
  "http://100.120.152.79:8765/download/JOB/result.tar.gz"
mkdir /Users/ME/Downloads/sam3-result
tar -xzf /Users/ME/Downloads/sam3-result.tar.gz -C /Users/ME/Downloads/sam3-result
```

`package_result` 幂等：`result.tar.gz` 仅在产物更新后重建；`-o` 覆盖已有本地文件，需要保留旧结果时先改名。单文件也可用 `/files/info/{path}` 查大小、`/download/{path}` 直接下载（支持 Range 断点续传）。归档包含 `result/rgba/*.png`、`result/overlay.mp4`、`result/analysis.json`、`result/previews/` 和包含完整 retry/review 历史的 `job.json`。Ubuntu 输出无需下载。归档也允许导出未通过结果，返回值明确标出 `quality_passed`。

## QA、故障与限制

数值指标逐帧计算：相邻 mask IoU、相对面积变化、centroid 和 bbox 跳变（归一化到图像对角线）、消失、扩张和碎片数；阈值和分数组合见 `mask_metrics`。预览包含分数最高 top_k、固定随机种子抽取的正常帧、首尾帧和局部重跑接缝。任何版本在视觉审核前 `quality_passed=false`；正常帧不足时仅选择实际可用的正常帧。抽样视觉审核无法保证未查看帧全部正确。

权重加载错误会阻止启动；任务错误有 traceback 日志以及带类型的 error 字段，失败输出不参与最佳选择。每个任务和 attempt 独立，输出不静默覆盖；中断重启后任务标为 failed，可在剩余预算内重试。成功版本保留用于比较/下载，不自动删除用户产物；失败的 RGBA 和临时打包/下载文件会清理。磁盘需求随帧数和重试次数增长，完成后由用户清理任务目录。服务通过 Tailscale 私有网络暴露给单用户可信 Mac，HTTP 层无账号认证。

本机无需权重的验证：

```sh
python -m unittest discover -s tests -v
```

测试使用合成视频和替代 predictor 验证产物、QA、局部保留、重试/审核门槛、MCP tool 注册以及 HTTP 上传/下载/打包数据面。真实 SAM3 精度、CUDA 显存、大文件 HTTP 传输须在部署环境验证。

已在独立临时环境通过 10 项 CPU 测试（Python 3.13、MCP 1.30.0、OpenCV 4.13），包括 Streamable HTTP 握手、MCP image 响应、归档内容以及 /upload、/download（Range/越界/404）、/files/info 与 `package_result` 的 HTTP 下载对接。部署目标仍按原工程使用 Python 3.12；未执行真实 GPU/权重推理或跨机器大文件 HTTP 实传。运行任务期间请保持输入视频不变。
