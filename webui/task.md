# Task

- 先理解`save_prompts_run_video.py`的作用
- 开发一个网页，其功能以及参数和`save_prompts_run_video.py`完全一致
- 使用端口8765，任何IP都可以访问
- 相关代码文件都放在`webui`中，不要动别的地方
- 和`save_prompts_run_video.py`不一样的地方：
  - 不使用--input_video xxx.mp4指定输入视频，而是添加一个Open按钮交互式选择运行该网页的服务器上的文件
  - 添加一个Save按钮来保存./saved_tracking_state.pt文件
- 不要在该Mac上做任何测试，只管开发