# MCP Task

请创建一个 Python MCP 服务脚本 `mcp_server.py`。该服务运行在 Ubuntu HPC 服务器上，并可由运行在 Mac 端的 Codex 通过网络远程调用。

## 目标

根据 `my_readme.md` 中描述的 SAM3 视频分割流程，实现一个 MCP 服务，对输入视频进行基于文本提示词的目标分割，并输出：

- 每一帧的 RGBA PNG 文件，其中 alpha 通道来自最终分割 mask；
- 一个用于人工/视觉模型检查分割质量的 overlay 验证视频 `.mp4`；
- 分割质量检测结果，包括异常帧列表、异常分数以及用于视觉检查的预览图片。

请尽量复用 `my_readme.md` 中已有的代码、模型加载方式、环境配置和推理流程，不要重新设计与现有工程不兼容的 SAM3 pipeline。

## MCP 接口要求

至少提供一个高级 MCP tool，例如：

`segment_video(...)`

建议参数包括：

- `video_path`: 输入视频路径；
- `text_prompt`: 初始文本提示词；
- `output_dir`: 输出目录；
- `top_k`: 需要重点检查的异常帧数量；
- `random_check_frames`: 额外随机抽取用于质量检查的帧数；
- `max_retry`: 自动重新分割的最大次数；
- 与 SAM3 tracking / prompt frame 有关的必要参数。

返回结果至少包括：

- 最终使用的文本提示词；
- 最终使用的 tracking / prompt frames；
- RGBA PNG 输出目录；
- overlay 验证视频路径；
- 异常帧检测结果；
- 用于视觉检查的 preview 图片路径；
- 是否通过最终质量检查；
- 每次 retry 的原因和修改内容。

可以根据工程需要再拆分出：

- `segment_video`
- `analyze_segmentation`
- `rerun_segmentation`
- `transfer_file`

等 MCP tools，但应优先提供一个可以完成完整工作流的高级接口，避免 Codex 必须手动组合大量低级调用。

## 自动质量检查

在返回最终结果之前，必须自动检查视频分割结果。

至少计算以下时间一致性指标：

1. 相邻帧 mask IoU；
2. mask 面积变化；
3. mask bounding box / centroid 的异常跳变；
4. mask 消失、突然扩张或出现大量碎片等情况。

根据这些指标计算每一帧的 anomaly score，并自动选择：

- anomaly score 最高的若干帧；
- 视频中均匀或随机抽取的若干正常帧；

生成 `original frame + mask overlay` 的 preview 图片，供 Codex 的视觉能力进一步检查。

不要只依赖数值指标判断最终结果是否正确。

## Codex 视觉检查与自动修正

MCP 服务本身负责产生异常检测结果和视觉 preview；Codex 负责查看这些 preview 图片并判断分割是否存在明显问题，例如：

- mask 漏掉目标区域；
- mask 分到了头发、皮肤、背景或其他物体；
- tracking 漂移到了错误目标；
- 某些帧 mask 消失；
- 边界或时间连续性明显异常。

如果 Codex 判断结果存在问题，应允许它通过 MCP 再次执行分割，并自动尝试：

- 修改 text prompt；
- 改变或增加 SAM3 prompt / tracking key frames；
- 从异常发生前后的帧重新初始化 tracking；
- 必要时缩小重新处理的帧范围，而不是始终从头处理整个视频。

重复：

`分割 → 数值异常检测 → preview 生成 → Codex 视觉检查 → 调整参数 → 重新分割`

直到：

- 质量检查通过；或
- 达到 `max_retry`。

不要实现无限循环。若达到 `max_retry` 仍存在明显问题，应返回当前最佳结果，并明确报告仍存在问题的帧和原因。

## Mac / Ubuntu 跨机器输入输出

输入视频可能位于：

1. Ubuntu HPC 本地；
2. Mac 本地。

输出文件也可能根据用户要求保存在：

1. Ubuntu；
2. Mac。

需要设计清晰的跨机器文件处理机制。

不要假设 Ubuntu 可以直接访问 `/Users/...` 等 Mac 本地路径，也不要假设 Mac 可以直接访问 Ubuntu 的 `/home/...` 路径。

建议为文件来源和目标增加明确参数，例如：

- `input_location = "ubuntu" | "mac"`
- `output_location = "ubuntu" | "mac"`

或者采用 URI / source abstraction。

如果输入文件在 Mac：

- 设计一个明确的上传/传输机制；
- 不要把大视频编码成 JSON/base64 直接作为 MCP 参数；
- 优先考虑通过 SSH/SCP/SFTP、共享目录或专门的文件传输接口完成。

如果输出目标是 Mac：

- 分割完成后支持把最终 RGBA PNG、验证视频和检查结果传回 Mac；
- 大量 PNG 文件建议打包成 `.tar.gz` 后再传输，除非已有共享目录。

文件传输实现应尽量独立于 SAM3 分割逻辑。

## 网络与 MCP

MCP 服务运行在 Ubuntu HPC 上，Codex 运行在 Mac 上。

因此不要使用只能由客户端本地启动进程的纯 stdio 架构作为唯一方案。

请使用适合跨机器访问的 MCP transport，例如 Streamable HTTP，并：

- 允许配置监听 host 和 port；
- 默认只监听安全网络接口；
- 考虑服务通过 Tailscale / SSH tunnel 使用，而不是直接暴露到公网；
- 给出 Mac 端 Codex 的 MCP 配置示例。

## HPC 注意事项

SAM3 模型应尽量在 MCP Server 启动时只加载一次并常驻 GPU，避免每个 MCP 请求都重新加载模型。

实现时还需要：

- 对并发请求做保护，避免多个任务同时占用同一个 GPU predictor 导致状态冲突；
- 每个任务使用独立工作目录；
- 正确清理临时文件；
- 输出日志；
- 对失败的任务返回清晰错误信息；
- 不要静默覆盖已有输出。

## 最终交付

请至少生成：

1. `mcp_server.py`
2. 如有必要，可增加辅助 Python 文件；
3. `requirements.txt` 或明确的新增依赖说明；
4. Mac 端 Codex MCP 配置示例；
5. Ubuntu HPC 启动命令；
6. 一个完整使用示例；
7. 简短说明整个 workflow：

`Mac/Ubuntu video → MCP → SAM3 → masks → anomaly detection → preview → Codex visual inspection → optional retry → RGBA PNG + validation MP4`

在实现之前先阅读并理解 `my_readme.md`，并以其中现有 SAM3 使用方法为准。