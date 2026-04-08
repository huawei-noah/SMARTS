"""
# SMARTS Quickstart Notebook (Jupyter / Jupytext friendly)
# 说明: 最小可运行示例，展示如何启动 SMARTS、运行一个短 episode 并做简单记录/回放提示。
# 保存为 docs/notebooks/smarts_quickstart.py，或使用 jupytext 转为 smarts_quickstart.ipynb。

# %%
# 前置条件
# - Python 3.8/3.9（以项目 README 为准）
# - 项目依赖已安装 (pip install -r requirements.txt)
# - 如需 SUMO：设置 SUMO_HOME 环境变量（示例见下）
import os
import sys
print("Python:", sys.version)
# 在 notebook 中可以临时设置 SUMO_HOME（示例）
# os.environ['SUMO_HOME'] = '/path/to/sumo'

# %%
# 导入并检查 SMARTS（根据仓库实际 API 适配）
try:
    import smarts
    print("SMARTS import OK. version:", getattr(smarts, "__version__", "unknown"))
except Exception as e:
    print("Unable to import SMARTS. Please ensure you're running in the project environment.")
    raise

# %%
# 最小示例：创建 env、跑 5 步、关闭
# 注意：下面的 env 初始化为占位示例，请按仓库 README 替换为正确代码（eg. gym.make(...) 或 smarts.env.SmartsEnv(...)）
try:
    import gym
    env = None
    try:
        env = gym.make("smarts-v0")  # <-- 替换为真实 env name
    except Exception:
        # 尝试仓库推荐的创建方式（占位）
        from smarts.env import SmartsEnv  # 替换/删除按实际 API
        # env = SmartsEnv(scenarios=[...])
        env = None

    if env:
        obs = env.reset()
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"step {step}: reward={reward}")
            if done:
                break
        env.close()
        print("Ran a short episode")
    else:
        print("No env created. Please update the env creation code per README.")
except Exception as ex:
    print("Error running example:", ex)

# %%
# 记录与回放（High level）
# 仓库里通常有 tools/recorder 或 scripts/replay.py，示例伪代码：
# from smarts.tools import recorder
# recorder.start("examples/recording1")
# run_simulation(...)
# recorder.stop()
# 回放:
# !python tools/replay.py --recording examples/recording1 --render

# %%
# 故障排查提示（写入 notebook 文档）
# - 如果遇到 TabError：请确保全部使用 spaces（PEP8 建议使用 4 spaces）
# - 如果出现 SUMO/依赖错误：检查 SUMO_HOME、pip install -r requirements.txt
# - 如果无法 import：确定已 activate 虚拟环境
"""