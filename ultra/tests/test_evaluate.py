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
import glob
import os
import pickle
import re
import shutil
import time
import unittest

import dill
import gym
import ray

from ultra.baselines.agent_spec import BaselineAgentSpec
from ultra.baselines.sac.sac.policy import SACPolicy
from ultra.evaluate import collect_evaluations, evaluate, evaluation_check
from ultra.utils.episode import episodes

seed = 2
AGENT_ID = "001"


class EvaluateTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/evaluate_test/"

    @classmethod
    def setUpClass(cls):
        path = os.path.join(EvaluateTest.OUTPUT_DIRECTORY, "sac_test_models/")
        multiagent_path = os.path.join(
            EvaluateTest.OUTPUT_DIRECTORY, "multiagent_test_models/"
        )
        generate_command = (
            "python ultra/scenarios/interface.py generate "
            "--task 00 --level eval_test --root-dir tests/scenarios "
            " --save-dir tests/task/eval_test/"
        )
        multiagent_generate_command = (
            "python ultra/scenarios/interface.py generate "
            "--task 00-multiagent --level eval_test --root-dir tests/scenarios "
            "--save-dir tests/task/eval_test_multiagent/"
        )
        train_command = (
            "python ultra/train.py "
            "--task 00 --level eval_test --policy sac --headless --episodes 1 "
            f"--eval-rate 1 --eval-episodes 1 --max-episode-steps 2 --log-dir {path}"
        )
        multiagent_train_command = (
            "python ultra/train.py "
            "--task 00-multiagent --level eval_test --policy sac,dqn,ppo --headless --episodes 1 "
            f"--eval-rate 1 --eval-episodes 1 --max-episode-steps 2 --log-dir {multiagent_path}"
        )

        # Generate the scenarios.
        os.system(generate_command)
        os.system(multiagent_generate_command)

        # Remove existing models
        if os.path.exists(path):
            shutil.rmtree(path)
        if os.path.exists(multiagent_path):
            shutil.rmtree(multiagent_path)

        # Generate models before evaluation tests
        if not os.path.exists(path):
            os.system(train_command)
        if not os.path.exists(multiagent_path):
            os.system(multiagent_train_command)

    def test_a_folders(self):
        path = os.path.join(EvaluateTest.OUTPUT_DIRECTORY, "sac_test_models/")
        if not os.path.exists(path):
            self.assertTrue(False)

        path = glob.glob(
            os.path.join(EvaluateTest.OUTPUT_DIRECTORY, "sac_test_models/*/models")
        )[0]
        if len(os.listdir(path)) == 0:
            self.assertTrue(False)

        path = "tests/task/eval_test"
        if len(os.listdir(path)) <= 2:
            self.assertTrue(False)

        multiagent_path = os.path.join(
            EvaluateTest.OUTPUT_DIRECTORY, "multiagent_test_models/"
        )
        if not os.path.exists(path):
            self.assertTrue(False)

        multiagent_path = glob.glob(
            os.path.join(
                EvaluateTest.OUTPUT_DIRECTORY, "multiagent_test_models/*/models"
            )
        )[0]
        if len(os.listdir(multiagent_path)) < 2:
            self.assertTrue(False)

        multiagent_path = "tests/task/eval_test_multiagent"
        if len(os.listdir(path)) <= 2:
            self.assertTrue(False)

    def test_evaluation_check(self):
        log_dir = os.path.join(EvaluateTest.OUTPUT_DIRECTORY, "output_eval_check_logs/")
        # ray.init(ignore_reinit_error=True)
        ray.shutdown()
        ray.init()
        try:
            run_experiment(
                scenario_info=("00", "eval_test"), num_agents=1, log_dir=log_dir
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

    def test_evaluation_check_multiagent(self):
        log_dir = os.path.join(
            EvaluateTest.OUTPUT_DIRECTORY, "output_eval_check_multiagent_logs/"
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

    def test_evaluate_cli(self):
        log_dir = os.path.join(EvaluateTest.OUTPUT_DIRECTORY, "output_eval_cli_logs/")
        experiment_dir = glob.glob(
            os.path.join(EvaluateTest.OUTPUT_DIRECTORY, "sac_test_models/*")
        )[0]
        evaluate_command = (
            f"python ultra/evaluate.py "
            f"--task 00 --level eval_test --experiment-dir {experiment_dir} "
            f"--episodes 1 --max-episode-steps 2 --log-dir {log_dir} --headless"
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

    def test_evaluate_cli_multiagent(self):
        log_dir = os.path.join(
            EvaluateTest.OUTPUT_DIRECTORY, "output_eval_cli_multiagent_logs/"
        )
        experiment_dir = glob.glob(
            os.path.join(EvaluateTest.OUTPUT_DIRECTORY, "multiagent_test_models/*")
        )[0]
        evaluate_command = (
            f"python ultra/evaluate.py "
            f"--task 00-multiagent --level eval_test --agents 000 --experiment-dir {experiment_dir} "
            f"--episodes 1 --max-episode-steps 2 --log-dir {log_dir} --headless"
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

    def test_evaluate_agent(self):
        seed = 2
        models_directory = glob.glob(
            os.path.join(EvaluateTest.OUTPUT_DIRECTORY, "sac_test_models/*/models/")
        )[0]
        log_dir = os.path.join(EvaluateTest.OUTPUT_DIRECTORY, "output_eval_agent_logs/")

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
                scenario_info=("00", "eval_test"),
                num_episodes=1,
                max_episode_steps=2,
                timestep_sec=0.1,
                headless=True,
                log_dir=log_dir,
            )
            self.assertTrue(True)
        except Exception as err:
            print(err)
            self.assertTrue(False)

    # This test performs evaluation on multiple agents, but the test map
    # that is created can only support one agent. Skip this for now until
    # we can specify a map to use that supports multiple agents.
    @unittest.skip("Test map does not yet support multiple agents.")
    def test_evaluate_multiagent(self):
        seed = 2
        models_directory = glob.glob(
            os.path.join(
                EvaluateTest.OUTPUT_DIRECTORY, "multiagent_test_models/*/models/"
            )
        )[0]
        log_dir = os.path.join(
            EvaluateTest.OUTPUT_DIRECTORY, "output_eval_multiagent_logs/"
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
            )
            self.assertTrue(True)
        except Exception as err:
            print(err)
            self.assertTrue(False)

    def test_record_evaluation_at_proper_episode_indices(self):
        """Due to parallelization, there might arise a situation where the episode
        object at the beginning of an evaluation would not match the episode
        object when recording to tensorboard. This test ensures that the evaluation data
        (for both test and train scenarios) is recorded at the proper episode index.
        """
        AGENT_ID = "000"
        log_dir = os.path.join(
            EvaluateTest.OUTPUT_DIRECTORY, "output_eval_episode_check_log/"
        )

        # Arbitary values for evaluation rate and number of training episodes
        eval_rate = 4
        num_episodes = 20

        train_command = (
            "python ultra/train.py "
            f"--task 00 --level eval_test --policy sac --headless --episodes {num_episodes} "
            f"--eval-rate {eval_rate} --eval-episodes 2 --max-episode-steps 2 --log-dir {log_dir}"
        )

        if not os.path.exists(log_dir):
            os.system(train_command)

        with open(
            os.path.join(
                log_dir, os.listdir(log_dir)[0], "pkls/Evaluation/results.pkl"
            ),
            "rb",
        ) as handle:
            evaluation_results = dill.load(handle)

        # Check if the episode indices are divisible by the evaluation rate. If they
        # do, then the evaluation data is properly saved under the results.pkl
        # and also correctly added to the tensorboard
        for index in evaluation_results[AGENT_ID].keys():
            self.assertEqual((index) % eval_rate, 0)

        with open(
            os.path.join(
                log_dir, os.listdir(log_dir)[0], "pkls/Evaluation_Training/results.pkl"
            ),
            "rb",
        ) as handle:
            evaluation_training_results = dill.load(handle)

        # Check if the episode indices are divisible by the evaluation rate. If they
        # do, then the evaluation training data is properly saved under the results.pkl
        # and also correctly added to the tensorboard
        for index in evaluation_training_results[AGENT_ID].keys():
            self.assertEqual((index) % eval_rate, 0)

    def test_extract_policy_from_path(self):
        paths = [
            "from.ultra.baselines.sac:sac-v0",
            "hello.ultra.ppo:ppo-v1",
            "ultra.custom:custom",
            "a.sb.ultra.c.d.e.sac:sac-v99",
            "a.b.c.d.e.ultra.custom_agent.policy:MBPO-v2",
        ]

        def extract(path):
            m = re.search(
                "ultra(.)*([a-zA-Z0-9_]*.)+([a-zA-Z0-9_])+:[a-zA-Z0-9_]+((-)*[a-zA-Z0-9_]*)*",
                path,
            )

            try:
                policy_class = m.group(0)  # pytype: disable=attribute-error
            except AttributeError as e:
                self.assertTrue(False)

        for path in paths:
            extract(path)

    # @classmethod
    # def tearDownClass(cls):
    #     os.system("ray stop")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(EvaluateTest.OUTPUT_DIRECTORY):
            shutil.rmtree(EvaluateTest.OUTPUT_DIRECTORY)
        if os.path.exists("tests/task/eval_test/"):
            shutil.rmtree("tests/task/eval_test/")
        if os.path.exists("tests/task/eval_test_multiagent/"):
            shutil.rmtree("tests/task/eval_test_multiagent/")


def run_experiment(scenario_info, num_agents, log_dir, headless=True):
    agent_ids = ["0" * max(0, 3 - len(str(i))) + str(i) for i in range(num_agents)]
    agent_classes = {agent_id: "ultra.baselines.sac:sac-v0" for agent_id in agent_ids}
    agent_specs = {
        agent_id: BaselineAgentSpec(policy_class=SACPolicy, max_episode_steps=2)
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
    etag = ":".join([policy_class.split(":")[-1] for policy_class in agent_classes])
    evaluation_task_ids = dict()

    for episode in episodes(1, etag=etag, log_dir=log_dir):
        observations = env.reset()
        dones = {"__all__": False}
        infos = None
        episode.reset()
        experiment_dir = episode.experiment_dir

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
            evaluation_check(
                agents=agents,
                agent_ids=agent_ids,
                episode=episode,
                eval_rate=10,
                eval_episodes=1,
                max_episode_steps=2,
                policy_classes=agent_classes,
                scenario_info=scenario_info,
                evaluation_task_ids=evaluation_task_ids,
                timestep_sec=0.1,
                headless=True,
                log_dir=log_dir,
            )
            collect_evaluations(evaluation_task_ids=evaluation_task_ids)

            actions = {
                agent_id: agents[agent_id].act(observation, explore=True)
                for agent_id, observation in observations.items()
            }
            next_observations, rewards, dones, infos = env.step(actions)

            active_agent_ids = observations.keys() & next_observations.keys()
            # pytype: disable=attribute-error
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
            # pytype: enable=attribute-error

            episode.record_step(
                agent_ids_to_record=active_agent_ids,
                infos=infos,
                rewards=rewards,
                total_step=total_step,
                loss_outputs=loss_outputs,
            )

            total_step += 1
            observations = next_observations

    # Wait on the remaining evaluations to finish.
    while collect_evaluations(evaluation_task_ids):
        time.sleep(0.1)

    env.close()
