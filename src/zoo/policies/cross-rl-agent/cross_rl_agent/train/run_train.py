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
# The following was modified from examples/ray_multi_instance.py

import argparse
import logging
import os
import pickle
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
from prioritized_replay import Buffer
from soc_mt_ac_network import SocMtActorNetwork, SocMtCriticNetwork

from smarts.core.agent import AgentSpec
from smarts.core.utils.episodes import episodes
from utils import get_split_batch

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


def init_tensorflow():
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    return configProto


def train(
    training_scenarios,
    sim_name,
    headless,
    num_episodes,
    seed,
    without_soc_mt,
    session_dir,
):
    WITH_SOC_MT = without_soc_mt
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
        scenarios=training_scenarios,
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
    # tensorflow summary for tensorboard visualization
    writer = tf.compat.v1.summary.FileWriter("summary")
    # losses
    tf.compat.v1.summary.scalar("Loss", critic_1.loss)
    tf.compat.v1.summary.scalar("Hubor_loss", critic_1.loss_2)
    tf.compat.v1.summary.histogram("ISWeights", critic_1.ISWeights)
    write_op = tf.compat.v1.summary.merge_all()
    saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    # init memory buffer
    buffer = Buffer(config.buffer_size, config.pretrain_length)
    if config.load_buffer:  # !!!the capacity of the buffer is limited with buffer file
        buffer = buffer.load_buffer(config.buffer_load_path)
        print("BUFFER: Buffer Loaded")
    else:
        buffer.fill_buffer(env, AGENT_ID)
        print("BUFFER: Buffer Filled")
        buffer.save_buffer(config.buffer_save_path, buffer)
    print("BUFFER: Buffer initialize")

    with tf.compat.v1.Session(config=configProto) as sess:
        # init nets params
        sess.run(tf.compat.v1.global_variables_initializer())
        writer.add_graph(sess.graph)
        # update params of the target network
        actor.update_target(sess)
        critic_1.update_target(sess)
        critic_2.update_target(sess)

        # Reinforcement Learning loop
        print("Training Starts...")
        # experiment results
        recent_rewards = []  # rewards from recent 100 episodes
        avarage_rewards = []  # avareage reward of recent 100 episodes
        recent_success = []
        recent_success_rate = []
        EPSILON = 1

        for episode in episodes(n=num_episodes):
            env_steps = 0
            # save the model from time to time
            if config.model_save_frequency:
                if episode.index % config.model_save_frequency == 0:
                    save_path = saver.save(sess, f"{session_dir}/{model_name}.ckpt")
                    print("latest model saved")
                if episode.index % config.model_save_frequency_no_paste == 0:
                    saver.save(
                        sess,
                        f"{session_dir}/{model_name}_{str(episode.index)}.ckpt",
                    )
                    print("model saved")

            # initialize
            EPSILON = (config.noised_episodes - episode.index) / config.noised_episodes
            episode_reward = 0

            observations = env.reset()  # states of all vehs
            state = observations[AGENT_ID]  # ego state
            episode.record_scenario(env.scenario_log)
            dones = {"__all__": False}
            while not dones["__all__"]:
                action_noise = actor.get_action_noise(sess, state, rate=EPSILON)
                observations, rewards, dones, infos = env.step(
                    {AGENT_ID: action_noise}
                )  # states of all vehs in next step

                # ego state in next step
                next_state = observations[AGENT_ID]
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

                # store the experience
                experience = state, action_noise, reward, next_state, done
                # print(state)
                buffer.store(experience)

                ## Model training STARTS
                if env_steps % config.train_frequency == 0:
                    # "Delayed" Policy Updates
                    policy_delayed = 2
                    for _ in range(policy_delayed):
                        # First we need a mini-batch with experiences (s, a, r, s', done)
                        tree_idx, batch, ISWeights_mb = buffer.sample(config.batch_size)
                        s_mb, a_mb, r_mb, next_s_mb, dones_mb = get_split_batch(batch)
                        task_mb = s_mb[:, -config.task_size :]
                        next_task_mb = next_s_mb[:, -config.task_size :]

                        # Get q_target values for next_state from the critic_target
                        if WITH_SOC_MT:
                            a_target_next_state = actor.get_action_target(
                                sess, next_s_mb
                            )  # with Target Policy Smoothing
                            q_target_next_state_1 = critic_1.get_q_value_target(
                                sess, next_s_mb, a_target_next_state
                            )
                            q_target_next_state_1 = (
                                q_target_next_state_1 * next_task_mb
                            )  # multi task q value
                            q_target_next_state_2 = critic_2.get_q_value_target(
                                sess, next_s_mb, a_target_next_state
                            )
                            q_target_next_state_2 = (
                                q_target_next_state_2 * next_task_mb
                            )  # multi task q value
                            q_target_next_state = np.minimum(
                                q_target_next_state_1, q_target_next_state_2
                            )
                        else:
                            a_target_next_state = actor.get_action_target(
                                sess, next_s_mb
                            )  # with Target Policy Smoothing
                            q_target_next_state_1 = critic_1.get_q_value_target(
                                sess, next_s_mb, a_target_next_state
                            )
                            q_target_next_state_2 = critic_2.get_q_value_target(
                                sess, next_s_mb, a_target_next_state
                            )
                            q_target_next_state = np.minimum(
                                q_target_next_state_1, q_target_next_state_2
                            )

                        # Set Q_target = r if the episode ends at s+1, otherwise Q_target = r + gamma * Qtarget(s',a')
                        target_Qs_batch = []
                        for i in range(0, len(dones_mb)):
                            terminal = dones_mb[i]
                            # if we are in a terminal state. only equals reward
                            if terminal:
                                target_Qs_batch.append((r_mb[i] * task_mb[i]))
                            else:
                                # take the Q taregt for action a'
                                target = (
                                    r_mb[i] * task_mb[i]
                                    + config.gamma * q_target_next_state[i]
                                )
                                target_Qs_batch.append(target)
                        targets_mb = np.array([each for each in target_Qs_batch])

                        # critic train
                        if len(a_mb.shape) > 2:
                            a_mb = np.squeeze(a_mb, axis=1)
                        loss, absolute_errors = critic_1.train(
                            sess, s_mb, a_mb, targets_mb, ISWeights_mb
                        )
                        loss_2, absolute_errors_2 = critic_2.train(
                            sess, s_mb, a_mb, targets_mb, ISWeights_mb
                        )
                    # actor train
                    a_for_grad = actor.get_action(sess, s_mb)
                    a_gradients = critic_1.get_gradients(sess, s_mb, a_for_grad)
                    # print(a_gradients)
                    actor.train(sess, s_mb, a_gradients[0])
                    # target train
                    actor.update_target(sess)
                    critic_1.update_target(sess)
                    critic_2.update_target(sess)

                    # update replay memory priorities
                    if WITH_SOC_MT:
                        absolute_errors = np.sum(absolute_errors, axis=1)
                    buffer.batch_update(tree_idx, absolute_errors)
                    ## Model training ENDS

                if done:
                    # visualize reward data
                    recent_rewards.append(episode_reward)
                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)
                    avarage_rewards.append(np.mean(recent_rewards))
                    avarage_rewards_data = np.array(avarage_rewards)
                    d = {"avarage_rewards": avarage_rewards_data}
                    with open(
                        os.path.join("results", "reward_data" + ".pkl"), "wb"
                    ) as f:
                        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
                    # visualize success rate data
                    if aux_info == "success":
                        recent_success.append(1)
                    else:
                        recent_success.append(0)
                    if len(recent_success) > 100:
                        recent_success.pop(0)
                    avarage_success_rate = recent_success.count(1) / len(recent_success)
                    recent_success_rate.append(avarage_success_rate)
                    recent_success_rate_data = np.array(recent_success_rate)
                    d = {"recent_success_rates": recent_success_rate_data}
                    with open(
                        os.path.join("results", "success_rate_data" + ".pkl"), "wb"
                    ) as f:
                        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
                    # print results on the terminal
                    print("Episode total reward:", episode_reward)
                    print("Episode time:", env_steps * 0.1)
                    print("Success rate:", avarage_success_rate)
                    print(episode.index, "episode finished.")
                    buffer.measure_utilization()
                    print("---" * 15)
                    break
                else:
                    state = next_state
                    env_steps += 1
        env.close()


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
        default=5000,
    )
    return parser


if __name__ == "__main__":
    parser = default_argument_parser("pytorch-example")
    parser.add_argument(
        "--without-soc-mt", help="Enable social mt.", action="store_true"
    )
    parser.add_argument(
        "--session-dir",
        help="The save directory for the model.",
        type=str,
        default="model/",
    )
    args = parser.parse_args()

    train(
        training_scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
        without_soc_mt=args.without_soc_mt,
        session_dir=args.session_dir,
    )
