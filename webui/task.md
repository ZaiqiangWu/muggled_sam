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

追加要求1：
- 对于每个视频分割任务，执行中的时候显示预估剩余所需时间
- 对于已经完成的分割任务，任务列表要保留，即使服务重启也不能丢失，除非用户主动清理

追加要求2:
该要求与check_generated_masks.py 脚本相关，该脚本主要完成以下任务：
- 从tar文件生成mask帧的rgba的png文件，若有多个object则合并，若只有单个object则只解压
- 生成 mask 预览视频，把mask帧的rgba的png文件合成为"白底 + 分割区域内原图内容"的mp4文件
我的要求是：
- 对已经完成的分割任务的任务条，在右下侧添加3个按钮：Preview, Accept, Delete
  - Preview 按钮按下后，执行生成 mask 预览视频（这里先不要把png文件移动到./videos/<garment>/<mask_name>/，先放在./generated_mask_videos/<mask_name>/），按钮中出现环形进度条；完成后进度条变为视频播放按钮，再次按下Preview后弹出窗口播放生成 mask 预览视频，进度条可以拖动，视频窗口右上角有个叉可以供用户关闭窗口，视频播放完毕后停止，不要自动关闭窗口
  - Accept 按钮按下后，把png文件移动到./videos/<garment>/<mask_name>/
  - Delete按钮按下后，删除相关tar文件，png文件和mask 预览视频mp4文件(若已经被accept移动到./videos/<garment>/<mask_name>/下的png文件不会被删除)

