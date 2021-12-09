import gym
import numpy as np
from stable_baselines3 import PPO
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
import env.rgb_image as smarts_rgb_image
import env.single_agent as smarts_single_agent
import env.adapter as adapter
import env.action as action

from stable_baselines3.common.env_checker import check_env

scenarios = ['scenarios/loop']

vehicle_interface = smarts_agent_interface.AgentInterface(
    max_episode_steps=300,
    rgb=smarts_agent_interface.RGB(
        width=64,
        height=64,
        resolution=1,
    ),
    action=getattr(
        smarts_controllers.ActionSpaceType,
        "Continuous",
    ),
    done_criteria=smarts_agent_interface.DoneCriteria(
        collision=True,
        off_road=True,
        off_route=False,
        on_shoulder=False,
        wrong_way=False,
        not_moving=False,
    ),
)

agent_specs = {
    "agent": smarts_agent.AgentSpec(
        interface=vehicle_interface,
        agent_builder=None,
        reward_adapter=adapter.reward_adapter,
        info_adapter=adapter.info_adapter,
    )
}

env = smarts_hiway_env.HiWayEnv(
    scenarios=scenarios,
    agent_specs=agent_specs,
    headless=True,
    visdom=False,
    seed=42,
    sim_name="env",
)

env = smarts_rgb_image.RGBImage(env=env, num_stack=1)
env = action.Action(env=env)
env = smarts_single_agent.SingleAgent(env)
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)

obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()