import unittest, ray
import numpy as np
from ultra.baselines.ppo.policy import PPOPolicy
import gym
from ultra.env.agent_spec import UltraAgentSpec
from smarts.core.controllers import ActionSpaceType

AGENT_ID = "001"
seed = 2


class AdapterTest(unittest.TestCase):
    def test_observation_features(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()

            observations = env.reset()
            env.close()
            return observations

        ray.init(ignore_reinit_error=True)
        observations = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue("speed" in observations[AGENT_ID]["state"])
        self.assertTrue("relative_goal_position" in observations[AGENT_ID]["state"])
        self.assertTrue("steering" in observations[AGENT_ID]["state"])
        self.assertTrue("angle_error" in observations[AGENT_ID]["state"])
        self.assertTrue("social_vehicles" in observations[AGENT_ID]["state"])
        self.assertTrue("road_speed" in observations[AGENT_ID]["state"])
        self.assertTrue("start" in observations[AGENT_ID]["state"])
        self.assertTrue("goal" in observations[AGENT_ID]["state"])
        self.assertTrue("heading" in observations[AGENT_ID]["state"])
        self.assertTrue("goal_path" in observations[AGENT_ID]["state"])
        self.assertTrue("ego_position" in observations[AGENT_ID]["state"])
        self.assertTrue("waypoint_paths" in observations[AGENT_ID]["state"])

        self.assertTrue("position" in observations[AGENT_ID]["ego"])
        self.assertTrue("speed" in observations[AGENT_ID]["ego"])
        self.assertTrue("steering" in observations[AGENT_ID]["ego"])
        self.assertTrue("heading" in observations[AGENT_ID]["ego"])
        self.assertTrue("dist_center" in observations[AGENT_ID]["ego"])
        self.assertTrue("start" in observations[AGENT_ID]["ego"])
        self.assertTrue("goal" in observations[AGENT_ID]["ego"])
        self.assertTrue("path" in observations[AGENT_ID]["ego"])
        self.assertTrue("closest_wp" in observations[AGENT_ID]["ego"])
        self.assertTrue("events" in observations[AGENT_ID]["ego"])

    def test_rewards_adapter(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()
            observations = env.reset()
            state = observations[AGENT_ID]["state"]
            action = agent.act(state, explore=True)
            observations, rewards, dones, infos = env.step({AGENT_ID: action})
            env.close()
            return rewards

        ray.init(ignore_reinit_error=True)
        rewards = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue("ego_social_safety_reward" in rewards[AGENT_ID]["log"])
        self.assertTrue("ego_num_violations" in rewards[AGENT_ID]["log"])
        self.assertTrue("social_num_violations" in rewards[AGENT_ID]["log"])
        self.assertTrue("goal_dist" in rewards[AGENT_ID]["log"])
        self.assertTrue("linear_jerk" in rewards[AGENT_ID]["log"])
        self.assertTrue("angular_jerk" in rewards[AGENT_ID]["log"])

    def test_rewards_returns(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()
            observations = env.reset()
            state = observations[AGENT_ID]["state"]
            action = agent.act(state, explore=True)
            observations, rewards, dones, infos = env.step({AGENT_ID: action})
            env.close()
            return rewards

        ray.init(ignore_reinit_error=True)
        rewards = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue(isinstance(rewards[AGENT_ID], dict))


def prepare_test_env_agent(headless=True):
    timestep_sec = 0.1
    # [throttle, brake, steering]
    policy_class = PPOPolicy
    spec = UltraAgentSpec(
        action_type=ActionSpaceType.Continuous,
        policy_class=policy_class,
        max_episode_steps=10,
    )
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=("00", "easy"),
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )
    agent = spec.build_agent()
    return agent, env
