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
import unittest, ray, glob, gym, os
from ultra.evaluate import evaluate, evaluation_check
from ultra.utils.episode import episodes
from ultra.baselines.agent_spec import BaselineAgentSpec
from smarts.zoo.registry import make

seed = 2
policy_class = "ultra.baselines.ppo:ppo-v0"
AGENT_ID = "001"


class EvaluateTest(unittest.TestCase):
    # def test_evaluation_check(self):
    #     # @ray.remote(max_calls=1, num_gpus=0)
    #     def run_experiment():
    #         total_step = 0
    #         agent, env = prepare_test_env_agent(headless=False)
    #         timestep_sec = env.timestep_sec
    #         for episode in episodes(1, etag="Train"):
    #             observations = env.reset()
    #             state = observations[AGENT_ID]
    #             dones, infos = {"__all__": False}, None
    #             episode.reset()
    #             while not dones["__all__"]:
    #                 evaluation_check(
    #                     agent=agent,
    #                     agent_id=AGENT_ID,
    #                     episode=episode,
    #                     eval_rate=10,
    #                     eval_episodes=1,
    #                     policy_class=policy_class,
    #                     scenario_info=("00", "easy"),
    #                     timestep_sec=0.1,
    #                     headless=True,
    #                 )
    #                 action = agent.act(state, explore=True)
    #                 observations, rewards, dones, infos = env.step({AGENT_ID: action})
    #                 next_state = observations[AGENT_ID]

    #                 # retrieve some relavant information from reward processor
    #                 # observations[AGENT_ID]["ego"].update(rewards[AGENT_ID]["log"])
    #                 loss_output = agent.step(
    #                     state=state,
    #                     action=action,
    #                     reward=rewards[AGENT_ID],
    #                     next_state=next_state,
    #                     done=dones[AGENT_ID],
    #                 )
    #                 episode.record_step(
    #                     agent_id=AGENT_ID,
    #                     infos=infos,
    #                     rewards=rewards,
    #                     total_step=total_step,
    #                     loss_output=loss_output,
    #                 )
    #                 total_step += 1
    #                 state = next_state

    #         env.close()

    #     # ray.init(ignore_reinit_error=True)
    #     try:
    #         ray.init()
    #         run_experiment()
    #         self.assertTrue(True)
    #     except Exception as err:
    #         print(err)
    #         self.assertTrue(False)

    def test_evaluate_cli(self):
        try:
            os.system(
                "python ultra/evaluate.py --task 00 --level easy --policy ppo --models ultra/tests/ppo_models/models --episodes 1"
            )
            self.assertTrue(True)
        except Exception as err:
            print(err)
            self.assertTrue(False)

    def test_evaluate_agent(self):
        seed = 2
        model = glob.glob("ultra/tests/ppo_models/models/*")[0]

        ray.init(ignore_reinit_error=True)
        try:
            evaluate.remote(
                experiment_dir=None,
                agent_id="AGENT_001",
                policy_class=policy_class,
                seed=seed,
                itr_count=0,
                checkpoint_dir=model,
                scenario_info=("00", "easy"),
                num_episodes=2,
                timestep_sec=0.1,
                headless=True,
            )
            self.assertTrue(True)
        except Exception as err:
            print(err)
            self.assertTrue(False)


def prepare_test_env_agent(headless=True):
    timestep_sec = 0.1
    # [throttle, brake, steering]
    policy_class = "ultra.baselines.ppo:ppo-v0"
    spec = make(locator=policy_class)
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=("00", "easy"),
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )
    agent = spec.build_agent()
    return agent, env
