from gym.envs.registration import register

register(
    id="ultra-v0", entry_point="ultra.env.ultra_env:UltraEnv",
)

# register(
#     id="task0-v0",
#     entry_point="ultra.env.task0_env:Task0Env",
# )

# register(
#     id="task1",
#     entry_point="ultra.env.ultra_env:UltraEnv",
# )
#
# register(
#     id="task2",
#     entry_point="ultra.env.ultra_env:Task2Env",
# )
