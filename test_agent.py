import gym
from smarts.zoo import registry
import sys

test_agent = registry.make_agent("zoo.policies:competition_agent-v0")
test_agent_spec = registry.make("zoo.policies:competition_agent-v0")

shared_configs = dict(
    action_space="TargetPose",
    img_meters=64,
    img_pixels=256,
    headless=True,
    sumo_headless=True,
)

test_env_path = "smarts.env:multi-scenario-v0"
test_senario = "1_to_2lane_left_turn_c"
test_env = gym.make(test_env_path, scenario=test_senario, **shared_configs)
test_env = test_agent_spec.adapt_env(test_env)

for _ in range(10):
    observations = test_env.reset()
    actions = test_agent.act(observations)
    print(actions)
    observations, rewards, dones, infos = test_env.step(actions)

test_env.close()
