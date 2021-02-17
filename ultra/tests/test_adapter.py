# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import unittest, ray
import numpy as np
import gym
from ultra.baselines.agent_spec import BaselineAgentSpec
from ultra.baselines.adapter import BaselineAdapter
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import make

AGENT_ID = "001"
seed = 2

adapter = BaselineAdapter()


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
        print(observations[AGENT_ID])
        self.assertTrue("speed" in observations[AGENT_ID])
        self.assertTrue("relative_goal_position" in observations[AGENT_ID])
        self.assertTrue("steering" in observations[AGENT_ID])
        self.assertTrue("angle_error" in observations[AGENT_ID])
        self.assertTrue("social_vehicles" in observations[AGENT_ID])
        self.assertTrue("road_speed" in observations[AGENT_ID])
        self.assertTrue("start" in observations[AGENT_ID])
        self.assertTrue("goal" in observations[AGENT_ID])
        self.assertTrue("heading" in observations[AGENT_ID])
        self.assertTrue("goal_path" in observations[AGENT_ID])
        self.assertTrue("ego_position" in observations[AGENT_ID])
        self.assertTrue("waypoint_paths" in observations[AGENT_ID])

    def test_rewards_adapter(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()
            observations = env.reset()
            state = observations[AGENT_ID]
            action = agent.act(state, explore=True)
            observations, rewards, dones, infos = env.step({AGENT_ID: action})
            env.close()
            return rewards

        ray.init(ignore_reinit_error=True)
        rewards = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(rewards[AGENT_ID], float)

    def test_rewards_returns(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()
            observations = env.reset()
            state = observations[AGENT_ID]
            action = agent.act(state, explore=True)
            observations, rewards, dones, infos = env.step({AGENT_ID: action})
            env.close()
            return rewards

        ray.init(ignore_reinit_error=True)
        rewards = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue(isinstance(rewards, dict))
        self.assertTrue(isinstance(rewards[AGENT_ID], float))


def prepare_test_env_agent(headless=True):
    timestep_sec = 0.1
    # [throttle, brake, steering]
    policy_class = "ultra.baselines.sac:sac-v0"
    spec = make(locator=policy_class)
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
