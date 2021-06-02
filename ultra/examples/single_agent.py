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
from argument_parser import argument_parser

from smarts.zoo.registry import make

from ultra.utils.common import agent_pool_value
from ultra.utils.episode import episodes


def run(args):
    """This example trains a single ULTRA baseline agent

    Args:
        args (argparse.Namespace): Arguments provided through the command line
    """
    # Default value of AGENT_ID
    AGENT_ID = "ego-agent"

    # Extract policy class from policy argument
    policy_class = agent_pool_value(args.policy, "policy_class")
    # Extract etag from policy class
    etag = policy_class.split(":")[-1]

    # Generate agent spec
    spec = make(locator=policy_class, max_episode_steps=args.max_episode_steps)

    # Build agent from agent specs
    agent = spec.build_agent()

    # Create an ULTRA environment
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=(args.task, args.level),
        headless=args.headless,
        timestep_sec=0.1,
        seed=args.seed,
    )

    total_step = 0
    for episode in episodes(args.episodes, log_dir=args.log_dir):
        observations = env.reset()
        episode.reset()
        dones = {"__all__": False}

        while not dones["__all__"]:
            actions = {AGENT_ID: agent.act(observations[AGENT_ID])}
            next_observations, rewards, dones, infos = env.step(actions)

            agent.step(
                state=observations[AGENT_ID],
                action=actions[AGENT_ID],
                reward=rewards[AGENT_ID],
                next_state=next_observations[AGENT_ID],
                done=dones[AGENT_ID],
                info=infos[AGENT_ID],
            )

            # Record the data from this episode.
            episode.record_step(
                agent_ids_to_record=[AGENT_ID],
                infos=infos,
                rewards=rewards,
                total_step=total_step,
            )
            total_step += 1

            observations = next_observations

        episode.record_episode()
        episode.record_tensorboard()

    env.close()


if __name__ == "__main__":
    parser = argument_parser("single-agent")
    args = parser.parse_args()
    run(args)
