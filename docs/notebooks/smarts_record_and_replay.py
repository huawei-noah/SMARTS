# Record and Replay Demo (Jupyter / Jupytext)
# 说明：展示如何记录仿真数据并用项目自带工具回放与可视化。
# 请将占位代码替换为仓库内实际 recorder/replay API。

# %%
# 1) 说明 recorder 使用方法（示例）
# from smarts.tools import recorder
# recorder.start("docs/notebooks/recording_demo")
# # 运行若干 step
# recorder.stop()
#
# 2) 回放（示例）
# !python tools/replay.py --recording docs/notebooks/recording_demo --render
#
# 3) 在 notebook 中显示截图
# from IPython.display import Image, display
# display(Image("docs/notebooks/recording_demo/snapshot.png"))
