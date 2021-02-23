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


class EvaluateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.system(
            "python ultra/scenarios/interface.py generate --task 00 --level eval_test --root-dir tests/scenarios --save-dir tests/task/eval_test/eval"
        )

        path = "tests/sac_test_models"
        if os.path.exists(path):
            shutil.rmtree(path)

        if not os.path.exists(path):
            # Generate models before evaluation tests
            os.system(
                "python ultra/train.py --task 00 --level eval_test --policy sac --headless True --episodes 1 --eval-rate 1 --eval-episodes 1 --max-episode-steps 2 --log-dir tests/sac_test_models"
            )

    def test_a_folders(self):
        path = "tests/sac_test_models"
        if not os.path.exists(path):
            self.assertTrue(False)

        path = glob.glob("tests/sac_test_models/*/models")[0]
        if len(os.listdir(path)) == 0:
            self.assertTrue(False)

        path = "tests/task/eval_test"
        if len(os.listdir(path)) <= 2:
            self.assertTrue(False)

    def test_evaluation_check(self):
        log_dir = "tests/output_eval_check_logs"
        # @ray.remote(max_calls=1, num_gpus=0)
        def run_experiment():
            total_step = 0
            agent, env, spec = prepare_test_env_agent(headless=True)
            timestep_sec = env.timestep_sec
            policy_class = "ultra.baselines.sac:sac-v0"
            log_dir = "tests/output_eval_check_logs"

            for episode in episodes(1, etag=policy_class, log_dir=log_dir):
                observations = env.reset()
                state = observations[AGENT_ID]
                dones, infos = {"__all__": False}, None
                episode.reset()
                experiment_dir = episode.experiment_dir

                if not os.path.exists(f"{experiment_dir}/spec.pkl"):
                    if not os.path.exists(experiment_dir):
                        os.makedirs(experiment_dir)
                    with open(f"{experiment_dir}/spec.pkl", "wb") as spec_output:
                        dill.dump(spec, spec_output, pickle.HIGHEST_PROTOCOL)

                while not dones["__all__"]:
                    evaluation_check(
                        agent=agent,
                        agent_id=AGENT_ID,
                        episode=episode,
                        eval_rate=10,
                        eval_episodes=1,
                        max_episode_steps=2,
                        policy_class=policy_class,
                        scenario_info=("00", "eval_test"),
                        timestep_sec=0.1,
                        headless=True,
                        log_dir=log_dir,
                    )
                    action = agent.act(state, explore=True)
                    observations, rewards, dones, infos = env.step({AGENT_ID: action})
                    next_state = observations[AGENT_ID]

                    # retrieve some relavant information from reward processor
                    # observations[AGENT_ID]["ego"].update(rewards[AGENT_ID]["log"])
                    loss_output = agent.step(
                        state=state,
                        action=action,
                        reward=rewards[AGENT_ID],
                        next_state=next_state,
                        done=dones[AGENT_ID],
                    )
                    episode.record_step(
                        agent_id=AGENT_ID,
                        infos=infos,
                        rewards=rewards,
                        total_step=total_step,
                        loss_output=loss_output,
                    )
                    total_step += 1
                    state = next_state

            env.close()

        # ray.init(ignore_reinit_error=True)
        ray.shutdown()
        ray.init()
        try:
            run_experiment()
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
        log_dir = "tests/output_eval_cli_logs/"
        model_dir = glob.glob("tests/sac_test_models/*/models")[0]
        ray.shutdown()
        try:
            os.system(
                f"python ultra/evaluate.py --task 00 --level eval_test --models {model_dir} --episodes 1 --max-episode-steps 2 --log-dir {log_dir} --headless True"
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

    def test_evaluate_agent(self):
        seed = 2
        model = glob.glob("tests/sac_test_models/*/models/0")[0]
        log_dir = "tests/output_eval_agent_logs/"
        policy_class = "ultra.baselines.sac:sac-v0"

        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        try:
            evaluate.remote(
                experiment_dir=None,
                agent_id="AGENT_001",
                policy_class=policy_class,
                seed=seed,
                itr_count=0,
                checkpoint_dir=model,
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
                policy_class = m.group(0)
            except AttributeError as e:
                self.assertTrue(False)

        for path in paths:
            extract(path)

    # @classmethod
    # def tearDownClass(cls):
    #     os.system("ray stop")


def prepare_test_env_agent(headless=True):
    timestep_sec = 0.1
    spec = BaselineAgentSpec(
        action_type=ActionSpaceType.Continuous,
        policy_class=SACPolicy,
        max_episode_steps=2,
    )
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=("00", "eval_test"),
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )
    agent = spec.build_agent()
    return agent, env, spec
