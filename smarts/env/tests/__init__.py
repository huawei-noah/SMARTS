# MIT License
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
from ray import tune


class atdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Trainable(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.score = 0
        self.env = config["env"](
            config=atdict(**config["env_config"], worker_index=0, vector_index=1)
        )
        self.action = config["multiagent"]["policies"]["policy"][2].sample()
        self.agent_ids = list(config["env_config"]["agent_specs"].keys())
        self.env.reset()

    def step(self):  # This is called iteratively.
        self.env.step({aid: self.action for aid in self.agent_ids})
        self.score += 1
        return {"score": self.score}

    def reset_config(self):
        self.env.reset()
