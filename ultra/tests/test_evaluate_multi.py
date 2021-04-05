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
import glob
import os
import pickle
import re
import shutil
import sys
import unittest

import dill
import gym
import ray

from smarts.core.controllers import ActionSpaceType
from ultra.baselines.agent_spec import BaselineAgentSpec
from ultra.baselines.sac.sac.policy import SACPolicy
from ultra.evaluate import evaluate, evaluation_check
from ultra.utils.episode import episodes

seed = 2
AGENT_ID = "001"


class MultiAgentEvaluateTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/multi_agent_evaluate_test/"

    @classmethod
    def setUpClass(cls):
        multiagent_path = os.path.join(
            MultiAgentEvaluateTest.OUTPUT_DIRECTORY, "multiagent_test_models/"
        )
        multiagent_generate_command = (
            "python ultra/scenarios/interface.py generate "
            "--task 00-multiagent --level eval_test --root-dir tests/scenarios "
            "--save-dir tests/task/eval_test_multiagent/"
        )
        multiagent_train_command = (
            "python ultra/train.py "
            "--task 00-multiagent --level eval_test --policy sac,dqn,ppo --headless True --episodes 1 "
            f"--eval-rate 1 --eval-episodes 1 --max-episode-steps 2 --log-dir {multiagent_path}"
        )

        # Generate the scenarios.
        os.system(multiagent_generate_command)

        # Remove existing models
        if os.path.exists(multiagent_path):
            shutil.rmtree(multiagent_path)

        # Generate models before evaluation tests
        if not os.path.exists(multiagent_path):
            os.system(multiagent_train_command)

    def test_a_folders(self):
        multiagent_path = os.path.join(
            MultiAgentEvaluateTest.OUTPUT_DIRECTORY, "multiagent_test_models/"
        )
        if not os.path.exists(multiagent_path):
            self.assertTrue(False)

        multiagent_path = glob.glob(
            os.path.join(
                MultiAgentEvaluateTest.OUTPUT_DIRECTORY,
                "multiagent_test_models/*/models",
            )
        )[0]
        if len(os.listdir(multiagent_path)) < 2:
            self.assertTrue(False)

        multiagent_path = "tests/task/eval_test_multiagent"
        if len(os.listdir(path)) <= 2:
            self.assertTrue(False)

    def test_evaluation_check_multiagent(self):
        log_dir = os.path.join(
            MultiAgentEvaluateTest.OUTPUT_DIRECTORY,
            "output_eval_check_multiagent_logs/",
        )
        # ray.init(ignore_reinit_error=True)
        ray.shutdown()
        ray.init()
        try:
            run_experiment(
                scenario_info=("00-multiagent", "eval_test"),
                num_agents=3,
                log_dir=log_dir,
            )
            self.assertTrue(True)
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if not os.listdir(log_dir):
            raise "Evaluation failed to generate new experiment folder"
            self.assertTrue(False)
        else:
            shutil.rmtree(log_dir)

    def test_evaluate_cli_multiagent(self):
        log_dir = os.path.join(
            MultiAgentEvaluateTest.OUTPUT_DIRECTORY, "output_eval_cli_multiagent_logs/"
        )
        experiment_dir = glob.glob(
            os.path.join(
                MultiAgentEvaluateTest.OUTPUT_DIRECTORY, "multiagent_test_models/*"
            )
        )[0]
        models = " ".join(glob.glob(os.path.join(experiment_dir, "models/000/")))
        evaluate_command = (
            f"python ultra/evaluate.py "
            f"--task 00-multiagent --level eval_test --models {models} --experiment-dir {experiment_dir} "
            f"--episodes 1 --max-episode-steps 2 --log-dir {log_dir} --headless True"
        )

        ray.shutdown()
        try:
            os.system(evaluate_command)
            self.assertTrue(True)
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if not os.listdir(log_dir):
            raise "Evaluation failed to generate new experiment folder"
            self.assertTrue(False)
        else:
            shutil.rmtree(log_dir)

    # This test performs evaluation on multiple agents, but the test map
    # that is created can only support one agent. Skip this for now until
    # we can specify a map to use that supports multiple agents.
    @unittest.skip
    def test_evaluate_multiagent(self):
        seed = 2
        models_directory = glob.glob(
            os.path.join(
                MultiAgentEvaluateTest.OUTPUT_DIRECTORY,
                "multiagent_test_models/*/models/",
            )
        )[0]
        log_dir = os.path.join(
            MultiAgentEvaluateTest.OUTPUT_DIRECTORY, "output_eval_multiagent_logs/"
        )

        with open(
            os.path.join(models_directory, "../agent_metadata.pkl"), "rb"
        ) as metadata_file:
            agent_metadata = pickle.load(metadata_file)

        agent_ids = agent_metadata["agent_ids"]
        policy_classes = agent_metadata["agent_classes"]
        checkpoint_directories = {
            agent_id: sorted(
                glob.glob(os.path.join(models_directory, agent_id, "*")),
                key=lambda x: int(x.split("/")[-1]),
            )
            for agent_id in agent_ids
        }

        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        try:
            evaluate.remote(
                experiment_dir=None,
                agent_ids=agent_ids,
                policy_classes=policy_classes,
                seed=seed,
                checkpoint_dirs=checkpoint_directories,
                scenario_info=("00-multiagent", "eval_test"),
                num_episodes=1,
                max_episode_steps=2,
                timestep_sec=0.1,
                headless=True,
                log_dir=log_dir,
                grade_mode=False,
            )
            self.assertTrue(True)
        except Exception as err:
            print(err)
            self.assertTrue(False)

    # @classmethod
    # def tearDownClass(cls):
    #     os.system("ray stop")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(MultiAgentEvaluateTest.OUTPUT_DIRECTORY):
            shutil.rmtree(MultiAgentEvaluateTest.OUTPUT_DIRECTORY)
        if os.path.exists("tests/task/eval_test_multiagent/"):
            shutil.rmtree("tests/task/eval_test_multiagent/")


def run_experiment(scenario_info, num_agents, log_dir, headless=True):
    agent_ids = ["0" * max(0, 3 - len(str(i))) + str(i) for i in range(num_agents)]
    agent_classes = {agent_id: "ultra.baselines.sac:sac-v0" for agent_id in agent_ids}
    agent_specs = {
        agent_id: BaselineAgentSpec(
            action_type=ActionSpaceType.Continuous,
            policy_class=SACPolicy,
            max_episode_steps=2,
        )
        for agent_id in agent_ids
    }

    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs=agent_specs,
        scenario_info=scenario_info,
        headless=headless,
        timestep_sec=0.1,
        seed=seed,
    )

    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    total_step = 0
    episode_count = 0
    old_episode = None
    etag = ":".join([policy_class.split(":")[-1] for policy_class in agent_classes])
    finished = False

    for episode in episodes(1, etag=etag, log_dir=log_dir):
        observations = env.reset()
        dones = {"__all__": False}
        infos = None
        episode.reset()
        experiment_dir = episode.experiment_dir

        # Save relevant agent metadata.
        if not os.path.exists(f"{experiment_dir}/agent_metadata.pkl"):
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            with open(f"{experiment_dir}/agent_metadata.pkl", "wb") as metadata_file:
                dill.dump(
                    {
                        "agent_ids": agent_ids,
                        "agent_classes": agent_classes,
                        "agent_specs": agent_specs,
                    },
                    metadata_file,
                    pickle.HIGHEST_PROTOCOL,
                )

        while not dones["__all__"]:
            # Break if any of the agent's step counts is 1000000 or greater.
            if any([episode.get_itr(agent_id) >= 1000000 for agent_id in agents]):
                finished = True
                break

            # Request and perform actions on each agent that received an observation.
            actions = {
                agent_id: agents[agent_id].act(observation, explore=True)
                for agent_id, observation in observations.items()
            }
            next_observations, rewards, dones, infos = env.step(actions)

            # Active agents are those that receive observations in this step and the next
            # step. Step each active agent (obtaining their network loss if applicable).
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
                loss_outputs=loss_outputs,
            )

            # Update variables for the next step.
            total_step += 1
            observations = next_observations

        episode.record_episode(old_episode, 1)
        old_episode = episode

        if (episode_count + 1) % 1 == 0:
            episode.record_tensorboard()
            old_episode = None

        # Perform the evaluation check.
        evaluation_check(
            agents=agents,
            agent_ids=agent_ids,
            episode=episode,
            eval_rate=1,
            eval_episodes=1,
            max_episode_steps=2,
            policy_classes=agent_classes,
            scenario_info=scenario_info,
            timestep_sec=0.1,
            headless=True,
            log_dir=log_dir,
            episode_count=episode_count,
            grade_mode=False,
            agent_coordinator=None,
        )

        episode_count += 1

        if finished:
            break

    env.close()
