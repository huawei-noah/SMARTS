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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import gymnasium as gym

# Reference: https://stackoverflow.com/a/53978543/2783780
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.utils.core_logging import diff_unpackable
from smarts.core.utils.episodes import episodes
from smarts.zoo.agent_spec import AgentSpec


def agent_spec(max_steps_per_episode):
    return AgentSpec(
        interface=AgentInterface(
            drivable_area_grid_map=True,
            occupancy_grid_map=True,
            top_down_rgb=True,
            lidar_point_cloud=True,
            waypoint_paths=True,
            max_episode_steps=max_steps_per_episode,
            debug=True,
            neighborhood_vehicle_states=True,
            action=ActionSpaceType.Lane,
        ),
        agent_builder=lambda: Agent.from_function(lambda _: 0),
    )


def run(agent_spec, callback, scenarios, episode_count, capture_step):
    AGENT_ID = "Agent-007"
    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=[scenarios],
        agent_interfaces={AGENT_ID: agent_spec.interface},
        headless=True,
        fixed_timestep_sec=0.1,
        seed=42,
    )
    i = 0
    for episode in episodes(n=episode_count):
        agent = agent_spec.build_agent()
        observations, _ = env.reset()

        episode.record_scenario(env.scenario_log)

        terminateds = {"__all__": False}
        while not terminateds["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, terminateds, truncateds, infos = env.step(
                {AGENT_ID: agent_action}
            )

            episode.record_step(observations, rewards, terminateds, truncateds, infos)

            if i % capture_step == 0:
                callback(
                    rewards, agent_obs, terminateds, truncateds, int(i / capture_step)
                )
            i += 1

    env.close()


def vehicle_state_check(vs_now, vs_prev):
    assert all([now == prev for (now, prev) in zip(vs_now.position, vs_prev.position)])
    assert vs_now.speed == vs_prev.speed
    assert vs_now.heading == vs_prev.heading
    assert vs_now.bounding_box == vs_prev.bounding_box
    assert vs_now.lane_id == vs_prev.lane_id
    assert vs_now.lane_index == vs_prev.lane_index


def determinism(agent_spec, scenarios, episode_count, capture_step):
    rewards_capture = []
    terminateds_capture = []
    truncateds_capture = []
    observations_capture = []

    def capture_callback(rewards, agent_obs, terminateds, truncateds, index):
        rewards_capture.append(rewards)
        terminateds_capture.append(terminateds)
        truncateds_capture.append(truncateds)
        observations_capture.append(agent_obs)

    def check_callback(rewards, agent_obs, terminateds, truncateds, index):
        assert len(rewards_capture) > index - 1
        orig_agent_obs = observations_capture[index]

        assert rewards_capture[index] == rewards

        # fmt: off
        assert len(terminateds) == len(terminateds_capture[index])
        assert len(truncateds) == len(truncateds_capture[index])
        assert all([ds == ds2 for (ds, ds2) in zip(terminateds, terminateds_capture[index])])
        assert all([ds == ds2 for (ds, ds2) in zip(truncateds, truncateds_capture[index])])
        # fmt: on
        assert diff_unpackable(agent_obs, orig_agent_obs) == ""

    run(agent_spec, capture_callback, scenarios, episode_count, capture_step)
    run(agent_spec, check_callback, scenarios, episode_count, capture_step)


def test_short_determinism():
    max_steps_per_episode = 20
    episode_count = 3
    capture_step = 5
    scenarios = "scenarios/sumo/intersections/2lane"
    determinism(
        agent_spec(max_steps_per_episode), scenarios, episode_count, capture_step
    )


def test_long_determinism():
    max_steps_per_episode = 55000
    episode_count = 1
    capture_step = 13750
    scenarios = "scenarios/sumo/intersections/2lane"
    determinism(
        agent_spec(max_steps_per_episode), scenarios, episode_count, capture_step
    )
