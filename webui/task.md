# Task 1: interface for Prompt Authoring (done)

- 先理解`save_prompts_run_video.py`的作用
- 开发一个网页，其功能以及参数和`save_prompts_run_video.py`完全一致
- 使用端口8765，任何IP都可以访问
- 相关代码文件都放在`webui`中，不要动别的地方
- 和`save_prompts_run_video.py`不一样的地方：
  - 不使用--input_video xxx.mp4指定输入视频，而是添加一个Open按钮交互式选择运行该网页的服务器上的文件
  - 添加一个Save按钮来保存./saved_tracking_state.pt文件
- 不要在该Mac上做任何测试，只管开发

# Task 2: interface for Video Segmentation (todo)

Task 1的Prompt Authoring网页界面开发已经完成，现在需要开发Video Segmentation的网页页面
- 页面顶部能够切换Prompt Authoring和Video Segmentation这两个界面，来回切换的时候每个界面的状态（已加载视频，prompt等）能够保持，不会随着切换而丢失

关于Video Segmentation界面的要求：
- 在这个界面能够提交视频分割任务，提交的任务排队运行，每个视频的分割进度可见
- 视频分割任务分两种：
  - 交互式指定服务器上的视频路径和pt文件路径后执行分割任务，等效于`python load_prompts_run_video.py --prompt_path ./tracking_state.pt --input_video ./video.mp4`
  - 交互式指定服务器上的视频文件夹路径和pt文件路径后执行分割任务，等效于`python load_prompts_run_dir.py --prompt_path ./tracking_state.pt --input_dir ./videos/`

别的要求：
- 相关代码文件都放在`webui`中，不要动别的地方
- 不要在该Mac上做任何测试，只管开发

