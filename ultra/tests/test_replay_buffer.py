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
import unittest

import numpy as np

import ultra.adapters as adapters
from ultra.baselines.common.replay_buffer import ReplayBuffer


def _print(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)


class ReplayBufferTest(unittest.TestCase):
    def test_vector_replay_buffer(self):
        pass

    def test_image_replay_buffer(self):
        # TRANSITIONS = 64
        # STACK_SIZE = 3
        # ACTION_SIZE = 3
        # IMAGE_WIDTH = 64
        # IMAGE_HEIGHT = 64
        # BUFFER_SIZE = 64
        # BATCH_SIZE = 32

        TRANSITIONS = 4
        STACK_SIZE = 2
        ACTION_SIZE = 3
        IMAGE_WIDTH = 4
        IMAGE_HEIGHT = 4
        BUFFER_SIZE = 4
        BATCH_SIZE = 2

        states = np.random.uniform(low=0.0, high=1.0, size=(TRANSITIONS, STACK_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)
        next_states = np.concatenate((states[1:], np.random.uniform(low=0.0, high=1.0, size=(1, STACK_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)))
        actions = np.random.uniform(low=-1.0, high=1.0, size=(TRANSITIONS, ACTION_SIZE)).astype(np.float32)
        previous_actions = np.random.uniform(low=-1.0, high=1.0, size=(TRANSITIONS,)).astype(np.float32)
        rewards = np.random.uniform(low=-10.0, high=10.0, size=(TRANSITIONS,)).astype(np.float32)
        dones = np.random.choice([True, False], size=(TRANSITIONS,))

        _print("states:", states, verbose=True)
        _print("next_states:", next_states, verbose=True)
        _print("actions:", actions)
        _print("previous_actions:", previous_actions)
        _print("rewards:", rewards)
        _print("dones:", dones)

        replay_buffer = ReplayBuffer(
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            observation_type=adapters.AdapterType.DefaultObservationImage,
            device_name="cpu",
        )

        for state, next_state, action, previous_action, reward, done in zip(states, next_states, actions, previous_actions, rewards, dones):
            _print("state:", state)
            _print("next_state:", next_state)
            _print("action:", action)
            _print("previous_action:", previous_action)
            _print("reward:", reward)
            _print("done:", done)

            replay_buffer.add(
                state=state,
                next_state=next_state,
                action=action,
                prev_action=previous_action,
                reward=reward,
                done=done,
            )

        sample = replay_buffer.sample()

        _print(">>> Asserting.")

        for state, action, reward, next_state, done, _ in zip(*sample):
            # TODO: Compare based on state.
            state = state.numpy()
            action = action.numpy()
            reward = reward.numpy()[0]
            next_state = next_state.numpy()
            done = True if done.numpy()[0] else False

            index_of_state = None
            for index, original_state in enumerate(states):
                if np.array_equal(original_state, state):
                    index_of_state = index
            _print("index_of_state:", index_of_state)

            _print("state:", state)
            _print("next_state:", next_state)
            _print("action:", action)
            _print("reward:", reward)
            _print("done:", done)

            self.assertTrue(state in states)
            self.assertTrue(next_state in next_states)
            self.assertTrue(action in actions)
            self.assertTrue(reward in rewards)
            self.assertTrue(done in dones)

            self.assertTrue(np.array_equal(state, states[index_of_state]))
            self.assertTrue(np.array_equal(next_state, next_states[index_of_state]))
            self.assertTrue(np.array_equal(action, actions[index_of_state]))
            self.assertTrue(reward == rewards[index_of_state])
            self.assertTrue(done == dones[index_of_state])
