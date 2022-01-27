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


class HyperParameters(object):
    """Hyperparameters for RL agent"""

    def __init__(self):
        # Env parameters
        self.ego_feature_num = 4
        self.npc_num = 5
        self.npc_feature_num = 5

        self.state_size = self.ego_feature_num + self.npc_num * self.npc_feature_num
        self.mask_size = self.npc_num + 1
        self.task_size = 4

        self.all_state_size = self.state_size + self.mask_size + self.task_size
        self.action_size = 2

        # Training parameters
        self.noised_episodes = 2500  # 2500
        self.max_steps = 500  # 400
        self.batch_size = 256  # 256
        self.train_frequency = 2

        # Soft update
        self.tau = 1e-3

        # Q LEARNING hyperparameters
        self.lra = 2e-5
        self.lrc = 1e-4
        self.gamma = 0.99  # Discounting rate
        self.pretrain_length = 2500  # Number of experiences stored in the Memory when initialized for the first time --INTIALLY 100k
        self.buffer_size = (
            100000  # Number of experiences the Memory can keep  --INTIALLY 100k
        )
        self.load_buffer = (
            True  # If True load memory, otherwise fill the memory with new data
        )
        self.buffer_load_path = "memory_buffer/memory.pkl"
        self.buffer_save_path = "memory_buffer/memory.pkl"

        # model saving
        self.model_save_frequency = 10
        self.model_save_frequency_no_paste = 50
