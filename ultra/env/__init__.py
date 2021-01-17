from gym.envs.registration import register

register(
    id="ultra-v0", entry_point="ultra.env.ultra_env:UltraEnv",
)
