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
"""
This examples runs the OpEn Agent, a classical, MPC based agent using [OpEn](https://alphaville.github.io/optimization-engine/).

NOTE: You will need to install Rust to run this example. See https://rustup.rs/ for instructions.
"""

import importlib
import logging
import math

import gym

from smarts.core.utils.episodes import episodes

try:
    open_agent = importlib.import_module("open_agent")
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"Ensure that the open-agent has been installed with `pip install open-agent"
    )

from .argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


def main(scenarios, sim_name, headless, num_episodes, seed):
    open_agent_spec = open_agent.entrypoint(debug=False, aggressiveness=3)
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: open_agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        fixed_timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        # envision_record_data_replay_path="./data_replay",
    )

    for episode in episodes(n=num_episodes):
        agent = open_agent_spec.build_agent()

        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

        del agent

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("OpEn-trajectory-optimizer-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
