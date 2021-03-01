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
import gym
import pytest

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes

AGENT_ID = "Agent-007"

MAX_EPISODES = 3

OBSERVATION_EXPECTED = "distance_from_center"
REWARD_EXPECTED = 3.14159
INFO_EXTRA_KEY = "__test_extra__"
ACTION_TO_BE_ADAPTED = "KEEP_LANE"  # expected to be adapted to lower case


@pytest.fixture
def agent_spec():
    def observation_adapter(env_observation):
        ego = env_observation.ego_vehicle_state
        waypoint_paths = env_observation.waypoint_paths
        wps = [path[0] for path in waypoint_paths]

        # distance of vehicle from center of lane
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
        lane_hwidth = closest_wp.lane_width * 0.5
        norm_dist_from_center = signed_dist_from_center / lane_hwidth

        return {OBSERVATION_EXPECTED: norm_dist_from_center}

    def reward_adapter(env_obs, env_reward):
        # reward is currently the delta in distance travelled by this agent.
        # We want to make sure that this is infact a delta and not total distance
        # travelled since this bug has appeared a few times.
        #
        # The way to verify this is by making sure the reward does not grow without bounds.
        assert -3 < env_reward < 3

        # Return a constant reward to test reward adapter call.
        return REWARD_EXPECTED

    def info_adapter(env_obs, env_reward, env_info):
        env_info[INFO_EXTRA_KEY] = "blah"
        return env_info

    def action_adapter(model_action):
        # We convert the action command to the required lower case.
        return model_action.lower()

    return AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=100),
        agent_builder=lambda: Agent.from_function(lambda _: ACTION_TO_BE_ADAPTED),
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
        info_adapter=info_adapter,
    )


@pytest.fixture
def env(agent_spec):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/loop"],
        agent_specs={AGENT_ID: agent_spec},
        headless=True,
        visdom=False,
        timestep_sec=0.01,
    )

    yield env
    env.close()


def test_hiway_env(env, agent_spec):
    for episode in episodes(n=MAX_EPISODES):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            obs = observations[AGENT_ID]
            observations, rewards, dones, infos = env.step({AGENT_ID: agent.act(obs)})
            episode.record_step(observations, rewards, dones, infos)

            assert (
                OBSERVATION_EXPECTED in observations[AGENT_ID]
            ), "Failed to apply observation adapter"

            assert (
                REWARD_EXPECTED == rewards[AGENT_ID]
            ), "Failed to apply reward adapter"

            assert INFO_EXTRA_KEY in infos[AGENT_ID], "Failed to apply info adapter"

    assert episode.index == (
        MAX_EPISODES - 1
    ), "Simulation must cycle through to the final episode."
