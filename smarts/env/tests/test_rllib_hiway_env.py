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
from pathlib import Path

import gym
import numpy as np
import psutil
import pytest
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.file import make_dir_in_smarts_log_dir
from smarts.env.rllib_hiway_env import RLlibHiWayEnv

AGENT_ID = "Agent-007"
INFO_EXTRA_KEY = "__test_extra__"


@pytest.fixture
def rllib_agent():
    def observation_adapter(env_observation):
        ego = env_observation.ego_vehicle_state
        waypoint_paths = env_observation.waypoint_paths
        wps = [path[0] for path in waypoint_paths]

        # distance of vehicle from center of lane
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
        lane_hwidth = closest_wp.lane_width * 0.5
        norm_dist_from_center = signed_dist_from_center / lane_hwidth

        return {
            "distance_from_center": np.array([norm_dist_from_center]),
            "angle_error": np.array([closest_wp.relative_heading(ego.heading)]),
            "speed": np.array([ego.speed]),
            "steering": np.array([ego.steering]),
        }

    def reward_adapter(env_obs, env_reward):
        return env_reward

    def action_adapter(model_action):
        throttle, brake, steering = model_action
        return np.array([throttle, brake, steering])

    def info_adapter(env_obs, env_reward, env_info):
        env_info[INFO_EXTRA_KEY] = "blah"
        return env_info

    # This action space should match the input to the action_adapter(..) function below.
    ACTION_SPACE = gym.spaces.Box(
        low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
    )

    # This observation space should match the output of observation_adapter(..) below
    OBSERVATION_SPACE = gym.spaces.Dict(
        {
            "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        }
    )

    return {
        "agent_spec": AgentSpec(
            interface=AgentInterface.from_type(
                AgentType.Standard,
                # We use a low number of steps here since this is a test
                max_episode_steps=10,
            ),
            observation_adapter=observation_adapter,
            reward_adapter=reward_adapter,
            action_adapter=action_adapter,
            info_adapter=info_adapter,
        ),
        "observation_space": OBSERVATION_SPACE,
        "action_space": ACTION_SPACE,
    }


def test_rllib_hiway_env(rllib_agent):
    def on_episode_end(info):
        episode = info["episode"]
        agent_info = episode.last_info_for(AGENT_ID)

        assert INFO_EXTRA_KEY in agent_info, "Failed to apply info adapter."

    rllib_model_name = "FullyConnectedNetwork"
    ModelCatalog.register_custom_model(rllib_model_name, FullyConnectedNetwork)

    rllib_policies = {
        "policy": (
            None,
            rllib_agent["observation_space"],
            rllib_agent["action_space"],
            {"model": {"custom_model": rllib_model_name}},
        )
    }

    # XXX: We should be able to simply provide "scenarios/loop"?
    scenario_path = Path(__file__).parent / "../../../scenarios/loop"
    tune_confg = {
        "env": RLlibHiWayEnv,
        "env_config": {
            "scenarios": [str(scenario_path.absolute())],
            "seed": 42,
            "headless": True,
            "agent_specs": {AGENT_ID: rllib_agent["agent_spec"]},
        },
        "callbacks": {"on_episode_end": on_episode_end},
        "multiagent": {
            "policies": rllib_policies,
            "policy_mapping_fn": lambda _: "policy",
        },
        "log_level": "WARN",
        "num_workers": 1,
    }

    # Test tune with the number of physical cpus with a minimum of 2 cpus
    num_cpus = max(2, psutil.cpu_count(logical=False) - 1)
    ray.init(num_cpus=num_cpus, num_gpus=0)
    analysis = tune.run(
        "PPO",
        name="RLlibHiWayEnv test",
        # terminate as soon as possible (this will run one training iteration)
        stop={"training_iteration": 1},
        max_failures=0,  # On failures, exit immediately
        local_dir=make_dir_in_smarts_log_dir("smarts_rllib_smoke_test"),
        config=tune_confg,
    )

    # trial status will be ERROR if there are any issues with the environment
    assert (
        analysis.get_best_trial("episode_reward_mean", mode="max").status
        == "TERMINATED"
    )
