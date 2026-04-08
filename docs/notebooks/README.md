# SMARTS Notebooks
本目录包含为项目准备的示例 Jupyter notebooks（也提供 jupytext-friendly 的 .py 版本以便版本控制）。

包含：
- smarts_quickstart.{py,ipynb} — 最小快速上手示例：启动环境并运行短 episode。
- smarts_colab_notes.{py,ipynb} — 在 Colab 上尝试的说明与限制说明。
- smarts_record_and_replay.{py,ipynb} — 演示如何记录仿真并回放/可视化。

如何使用：
- 本地：在项目根目录下运行 `jupyter notebook`，打开 docs/notebooks/*.ipynb。
- 如果使用 jupytext：`jupytext --to notebook docs/notebooks/smarts_quickstart.py`
- 注意：某些示例需要 SUMO 或其他系统依赖，请参考项目 README 中的环境说明。
