# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
# The author of this file is: https://github.com/mg2015started

# The following test was modified from examples/ray_multi_instance.py

import argparse
import logging
import warnings

import gym
import numpy as np
import tensorflow as tf
from ac_network import ActorNetwork, CriticNetwork
from adapters import (
    action_adapter,
    cross_interface,
    get_aux_info,
    observation_adapter,
    reward_adapter,
)
from config import HyperParameters
from soc_mt_ac_network import SocMtActorNetwork, SocMtCriticNetwork

from smarts.core.agent import AgentSpec
from smarts.core.utils.episodes import episodes

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"
WITH_SOC_MT = True


def init_tensorflow():
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    return configProto


def test(test_scenarios, sim_name, headless, num_episodes, seed):
    config = HyperParameters()
    configProto = init_tensorflow()
    # init env
    agent_spec = AgentSpec(
        # you can custom AgentInterface to control what obs information you need and the action type
        interface=cross_interface,
        # agent_builder=actor,
        # you can custom your observation adapter, reward adapter, info adapter, action adapter and so on.
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=test_scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        timestep_sec=0.1,
        seed=seed,
    )
    # init nets structure
    if WITH_SOC_MT:
        model_name = "Soc_Mt_TD3Network"
        actor = SocMtActorNetwork(name="actor")
        critic_1 = SocMtCriticNetwork(name="critic_1")
        critic_2 = SocMtCriticNetwork(name="critic_2")
    else:
        model_name = "TD3Network"
        actor = ActorNetwork(name="actor")
        critic_1 = CriticNetwork(name="critic_1")
        critic_2 = CriticNetwork(name="critic_2")
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session(config=configProto) as sess:
        # load network
        saver = tf.compat.v1.train.import_meta_graph(
            "models/" + model_name + ".ckpt" + ".meta"
        )
        saver.restore(sess, "models/" + model_name + ".ckpt")
        if saver is None:
            print("did not load")

        # init testing params
        test_num = 100
        test_ep = 0
        # results record
        success = 0
        failure = 0
        passed_case = 0

        collision = 0
        trouble_collision = 0
        time_exceed = 0
        episode_time_record = []

        # start testing
        for episode in episodes(n=num_episodes):
            episode_reward = 0
            env_steps = 0  # step in one episode
            observations = env.reset()  # states of all vehs
            state = observations[AGENT_ID]  # ego state
            episode.record_scenario(env.scenario_log)
            dones = {"__all__": False}
            while not dones["__all__"]:
                action = actor.get_action_noise(sess, state, rate=-1)
                observations, rewards, dones, infos = env.step(
                    {AGENT_ID: action}
                )  # states of all vehs in next step

                # ego state in next step
                state = observations[AGENT_ID]
                if WITH_SOC_MT:
                    reward = rewards[AGENT_ID]
                else:
                    reward = np.sum(reward)
                done = dones[AGENT_ID]
                info = infos[AGENT_ID]
                aux_info = get_aux_info(infos[AGENT_ID]["env_obs"])
                episode.record_step(observations, rewards, dones, infos)
                if WITH_SOC_MT:
                    episode_reward += np.sum(reward)
                else:
                    episode_reward += reward
                env_steps += 1

                if done:
                    test_ep += 1
                    # record result
                    if aux_info == "collision":
                        collision += 1
                        failure += 1
                    elif aux_info == "trouble_collision":
                        trouble_collision += 1
                        passed_case += 1
                    elif aux_info == "time_exceed":
                        time_exceed += 1
                        failure += 1
                    else:
                        # get episode time
                        episode_time_record.append(env_steps * 0.1)
                        success += 1
                    # print
                    print(
                        episode.index,
                        "EPISODE ended",
                        "TOTAL REWARD {:.4f}".format(episode_reward),
                        "Result:",
                        aux_info,
                    )
                    print("total step of this episode: ", env_steps)
                    episode_reward = 0
                    env_steps = 0
                    observations = env.reset()  # states of all vehs
                    state = observations[AGENT_ID]  # ego state
        env.close()

        print("-*" * 15, " result ", "-*" * 15)
        print("success: ", success, "/", test_num)
        print("collision: ", collision, "/", test_num)
        print("time_exceed: ", time_exceed, "/", test_num)
        print("passed_case: ", passed_case, "/", test_num)
        print("average time: ", np.mean(episode_time_record))


def main(
    test_scenarios,
    sim_name,
    headless,
    num_episodes,
    seed,
):
    test(
        test_scenarios,
        sim_name,
        headless,
        num_episodes,
        seed,
    )


def default_argument_parser(program: str):
    """This factory method returns a vanilla `argparse.ArgumentParser` with the
    minimum subset of arguments that should be supported.

    You can extend it with more `parser.add_argument(...)` calls or obtain the
    arguments via `parser.parse_args()`.
    """
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "scenarios",
        help="A list of scenarios. Each element can be either the scenario to run "
        "(see scenarios/ for some samples you can use) OR a directory of scenarios "
        "to sample from.",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--sim-name",
        help="a string that gives this simulation a name.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--headless", help="Run the simulation in headless mode.", action="store_true"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sumo-port", help="Run SUMO with a specified port.", type=int, default=None
    )
    parser.add_argument(
        "--episodes",
        help="The number of episodes to run the simulation for.",
        type=int,
        default=100,
    )
    return parser


if __name__ == "__main__":
    parser = default_argument_parser("pytorch-example")
    args = parser.parse_args()

    main(
        test_scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
