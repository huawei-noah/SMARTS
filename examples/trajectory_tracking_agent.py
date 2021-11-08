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
import logging

import gym

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes

from .argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class TrackingAgent(Agent):
    def act(self, obs):
        lane_index = 0
        num_trajectory_points = min([10, len(obs.waypoint_paths[lane_index])])
        # Desired speed is in m/s
        desired_speed = 50 / 3.6
        trajectory = [
            [
                obs.waypoint_paths[lane_index][i].pos[0]
                for i in range(num_trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_index][i].pos[1]
                for i in range(num_trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_index][i].heading
                for i in range(num_trajectory_points)
            ],
            [desired_speed for i in range(num_trajectory_points)],
        ]
        return trajectory


def main(scenarios, sim_name, headless, num_episodes, seed):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Tracker, max_episode_steps=None),
        agent_builder=TrackingAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        fixed_timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        # envision_record_data_replay_path="./data_replay",
    )

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("trajectory-tracking-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
