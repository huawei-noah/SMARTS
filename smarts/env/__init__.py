from gym.envs.registration import register


register(
    id="hiway-v0", entry_point="smarts.env.hiway_env:HiWayEnv",
)
