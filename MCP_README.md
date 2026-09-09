# SAM3 远程 MCP 服务

工作流：`Mac/Ubuntu video → MCP → SAM3 keyframe 候选 → 逐帧视觉审核 → tracking → RGBA masks → anomaly detection → preview → Codex visual inspection → optional retry → RGBA PNG + validation MP4`。

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
- `GET /download/{path}`：从 work-root 流式下载，支持 `Range` 断点续传，例 `GET /download/JOB/result.tar`。
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

1. `segment_video`：提交后立即返回 `job_id`，不阻塞 HTTP 请求；分割在后台异步执行。多任务可以同时提交，但 GPU 推理走同一 worker 队列串行（同一时刻只有一个 SAM3 推理，避免显存抢占）。

```json
{"video_path":"UPLOADED_FILE_ID/input.mp4","text_prompt":"quilted jacket","output_location":"mac","top_k":8,"random_check_frames":4,"max_retry":2,"prompt_frames":[0],"candidate_rank":0}
```

2. 先审核全部 keyframe：轮询 `get_job_status`，当 `internal_status="awaiting_keyframe_review"`（对外 `status="running"`）时，按 `pending_keyframes` 逐帧调用下面两个工具。`attempt_index` 使用当前 attempt 的序号，可从 `get_job_status.attempt` 或 `get_segmentation` 获取。

```text
get_keyframe_preview({"job_id":"JOB","attempt_index":0,"frame":0})
# 先按返回地址 HTTP 下载图片到 Mac /tmp，并查看本地图片，再提交审核：
submit_keyframe_review({"job_id":"JOB","attempt_index":0,"frame":0,"passed":true,"notes":"已查看原图与 overlay，目标身份、覆盖范围和边界正确，无背景误分"})
```

`get_keyframe_preview` 仅返回 JSON 文件信息，不返回 MCP Image、图片字节或 base64。`preferred_preview` 优先指向该帧的 comparison panel；`comparison_panel` 为 panel 文件信息，`candidates` 列出最多四个可选候选（置信度排序 rank 0..3）的 `frame`、`candidate_rank`、`score`、`selected`、work-root-relative `path` 和 `download_url`。`selected_candidate_rank` 表示本次审核对应的候选；查看其他候选不会改变所选 mask，要换候选仍需拒绝并重试。

新预览保存在 `<attempt>/keyframe_previews/frame_00000000_candidate_0.png`，对比图为 `frame_00000000_comparison.png`。每格显示 original/overlay、rank、score，并标出 SELECTED；四个候选采用 2×2 排列，panel 宽度不超过 1920 px。少于四个时仅返回实际可选候选。旧待审核任务可返回已有 `keyframes/00000000.png`，此时 `comparison_panel=null`，不会为了补图重新运行检测。

`download_url` 是相对于 MCP 服务 origin 的 `/download/...` 地址，复用现有下载接口。例如在 Mac 执行（替换服务器、JOB、attempt 和文件名）：

```sh
preview_dir=$(mktemp -d /tmp/sam3-keyframe.XXXXXX)
curl --fail --show-error \
  -o "$preview_dir/frame_00000000_comparison.png" \
  "http://100.120.152.79:8765/download/JOB/attempt_00/keyframe_previews/frame_00000000_comparison.png"
```

然后用视觉工具（如 `view_image`）查看下载后的本地文件；若 panel 中细节不够，再下载所选 candidate 单图。不要读取图片后转为 base64 放进 MCP JSON。获取元数据只记录预览地址已取回，服务不能据此确认客户端下载或实际看过图片；调用方必须完成本地视觉检查并在审核 notes 中说明结论。

必须实际查看每个 keyframe 的原图/overlay，再提交判断；不能依据置信度自动通过。未取回预览、空 notes、重复审核或旧 attempt 审核都会被拒绝。全部通过后才开始 tracking，并直接读取已审核的候选 mask，不重新检测。任何一个拒绝（`passed=false`，notes 说明原因）都会停止本次尝试，进入 `needs_retry` 或 `retry_exhausted`；可用 `rerun_segmentation` 修改文本、候选排名或关键帧，重试仍须重新审核所有 keyframe。首次 keyframe 被拒绝时尚无完整结果可供打包。

候选 mask 保存在 attempt 的 `keyframes/`，预览在 `keyframe_previews/`，候选元数据与审核状态在 `job.json` 中。等待期间释放 GPU worker，其他任务可继续执行；重启后可继续完成等待中的审核。这里的“确认正确”由查看图像的调用方（Codex）负责，服务强制执行审核门槛，不以数值指标代替语义判断。

全部 keyframe 通过后，继续轮询 `get_job_status({"job_id":"JOB"})`，间隔数秒查询：`status` 为 `queued/running/completed/failed`，`running` 时返回 `processed_frames`/`total_frames`/`progress`（0..1，尽力而为：总帧数来自容器元数据、可能为 null，完成时改用精确帧数）。失败时返回带类型的 `error` 和完整 `traceback`。完成后调用 `get_job_result({"job_id":"JOB"})` 获取全部产物路径（work-root 相对，可直接拼 `/download/<path>`），再调用 `get_segmentation({"job_id":"JOB"})` 查看各次尝试配置、异常帧与分数及重试历史。每一帧的完整指标位于 `analysis.json`。

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

6. 调用 `package_result({"job_id":"JOB"})` 立即返回 `status="packaging"`，每隔数秒重复调用查询，期间不会重复启动同一 job 的打包任务。后台完成后返回 `status="ready"`、`archive_path`、`download_url` 和 `download`（work-root 相对路径，如 `JOB/result.tar`），然后在 Mac 执行：

```sh
curl --fail --show-error \
  -o /Users/ME/Downloads/sam3-result.tar \
  "http://100.120.152.79:8765/download/JOB/result.tar"
mkdir /Users/ME/Downloads/sam3-result
tar -xf /Users/ME/Downloads/sam3-result.tar -C /Users/ME/Downloads/sam3-result
```

`package_result` 使用独立后台线程生成无 gzip 压缩的普通 `.tar`，目录扫描和缓存检查也在后台执行，不占用 GPU 推理队列。返回 `status="failed"` 时包含 `error`；下一次调用可重新尝试。取回 ready/failed 后再次调用会启动新的后台缓存检查：若 `result.tar` 不早于全部待打包 artifact 和 job.json，则直接复用，不重新写 tar；否则重建。临时文件为 `.result.tar.<uuid>.partial`，成功后原子替换 `result.tar`，失败时清理临时文件并保留已有归档。`-o` 覆盖已有本地文件，需要保留旧结果时先改名。单文件也可用 `/files/info/{path}` 查大小、`/download/{path}` 直接下载（支持 Range 断点续传）。归档包含 `result/rgba/*.png`、`result/overlay.mp4`、`result/analysis.json`、`result/previews/` 和包含完整 retry/review 历史的 `job.json`。Ubuntu 输出无需下载。归档也允许导出未通过结果，返回值明确标出 `quality_passed`。

## QA、故障与限制

数值指标逐帧计算：相邻 mask IoU、相对面积变化、centroid 和 bbox 跳变（归一化到图像对角线）、消失、扩张和碎片数；阈值和分数组合见 `mask_metrics`。最终视频预览包含分数最高 top_k、固定随机种子抽取的正常帧、首尾帧、所有 keyframe 和局部重跑接缝。任何版本在视觉审核前 `quality_passed=false`；正常帧不足时仅选择实际可用的正常帧。抽样视觉审核无法保证未查看帧全部正确。

权重加载错误会阻止启动；任务错误有 traceback 日志以及带类型的 error 字段，失败输出不参与最佳选择。每个任务和 attempt 独立，输出不静默覆盖；推理中断重启后任务标为 failed（等待 keyframe 审核的任务保留审核状态），可在剩余预算内重试。损坏的 job.json 会在启动时被跳过并告警，不会导致服务启动失败；单个任务失败只写回该任务的 job.json，不影响服务进程和其余任务。成功版本保留用于比较/下载，不自动删除用户产物；失败的 RGBA 和临时打包/下载文件会清理。磁盘需求随帧数和重试次数增长，完成后由用户清理任务目录。服务通过 Tailscale 私有网络暴露给单用户可信 Mac，HTTP 层无账号认证。

本机无需权重的验证：

```sh
python -m unittest discover -s tests -v
```

测试使用合成视频和替代 predictor 验证产物、QA、局部保留、重试/审核门槛、MCP tool 注册以及 HTTP 上传/下载/打包数据面。真实 SAM3 精度、CUDA 显存、大文件 HTTP 传输须在部署环境验证。

已在独立临时环境通过 CPU 测试（Python 3.13、MCP 1.30.0、OpenCV 4.13），包括 Streamable HTTP 握手、MCP image 响应、归档内容、异步 job 状态/进度/失败 traceback/重启恢复，以及 /upload、/download（Range/越界/404）、/files/info 与 `package_result` 的 HTTP 下载对接。部署目标仍按原工程使用 Python 3.12；未执行真实 GPU/权重推理或跨机器大文件 HTTP 实传。运行任务期间请保持输入视频不变。
