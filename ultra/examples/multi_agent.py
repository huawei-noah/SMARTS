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

from ultra.train import build_agent
from ultra.utils.common import agent_pool_value
from ultra.utils.episode import episodes


def run(args):
    """This example trains multi ULTRA baseline agents

    Args:
        args (argparse.Namespace): Arguments provided through the command line
    """

    # Obtain the policy class strings for each specified policy.
    policy_classes = [
        agent_pool_value(agent_name, "policy_class")
        for agent_name in args.policy.split(",")
    ]

    # Obtain the policy class IDs from the arguments.
    policy_ids = args.policy_ids.split(",") if args.policy_ids else None

    agent_ids, agent_classes, agent_specs, agents, etag = build_agent(
        policy_classes, policy_ids, args.max_episode_steps
    )

    # Create an ULTRA environment
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs=agent_specs,
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
            actions = {
                agent_id: agents[agent_id].act(observation, explore=True)
                for agent_id, observation in observations.items()
            }
            next_observations, rewards, dones, infos = env.step(actions)

            active_agent_ids = observations.keys() & next_observations.keys()
            loss_outputs = {
                agent_id: agents[agent_id].step(
                    state=observations[agent_id],
                    action=actions[agent_id],
                    reward=rewards[agent_id],
                    next_state=next_observations[agent_id],
                    done=dones[agent_id],
                    info=infos[agent_id],
                )
                for agent_id in active_agent_ids
            }

            # Record the data from this episode.
            episode.record_step(
                agent_ids_to_record=active_agent_ids,
                infos=infos,
                rewards=rewards,
                total_step=total_step,
            )
            total_step += 1

        episode.record_episode()
        episode.record_tensorboard()

    env.close()


if __name__ == "__main__":
    parser = argument_parser("multi-agent")
    args = parser.parse_args()
    run(args)
