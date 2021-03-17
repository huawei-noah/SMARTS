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
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.sac as sac
import ray.rllib.agents.ddpg as ddpg


class RllibAgent:
    def __init__(self, agent_name, env, config, logger_creator):
        assert agent_name in [
            "td3",
            "ddpg",
            "ppo",
        ], "Some policies are not currently supported (dqn,sac)"  # dqn and sac not currently supported
        self.agent_name = agent_name

        if self.agent_name == "ppo":
            self.trainer = ppo.PPOTrainer(
                env=env,
                config=config,
                logger_creator=logger_creator,
            )
        elif self.agent_name == "ddpg":
            self.trainer = ddpg.DDPGTrainer(
                env=env,
                config=config,
                logger_creator=logger_creator,
            )
        elif self.agent_name == "td3":
            self.trainer = ddpg.TD3Trainer(
                env=env,
                config=config,
                logger_creator=logger_creator,
            )
        # elif self.agent_name=='dqn':
        #     trainer = dqn.DQNTrainer(
        #         env=RLlibUltraEnv,
        #         config=tune_config,
        #         logger_creator=log_creator(log_dir),
        #     )

    def train(self):
        return self.trainer.train()

    def log_evaluation_metrics(self, results):
        return self.trainer.log_result(results)

    @staticmethod
    def rllib_default_config(agent_name):
        assert agent_name in [
            "td3",
            "ddpg",
            "ppo",
        ], "Some policies are not currently supported (dqn,sac)"  # dqn and sac not currently supported

        if agent_name == "ppo":
            return ppo.DEFAULT_CONFIG.copy()
        elif agent_name == "ddpg":
            return ddpg.DEFAULT_CONFIG.copy()
        elif agent_name == "td3":
            return td3.DEFAULT_CONFIG.copy()
