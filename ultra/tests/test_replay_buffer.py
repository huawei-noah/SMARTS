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


class ReplayBufferTest(unittest.TestCase):
    def test_image_replay_buffer(self):
        TRANSITIONS = 1024  # The number of transitions to save in the replay buffer.
        STACK_SIZE = 4  # The stack size of the images.
        ACTION_SIZE = 3  # The size of each action.
        IMAGE_WIDTH = 64  # The width of each image.
        IMAGE_HEIGHT = 64  # The height of each image.
        BUFFER_SIZE = 1024  # The size of the replay buffer.
        BATCH_SIZE = 128  # Batch size of each sample from the replay buffer.
        NUM_SAMPLES = 10  # Number of times to sample from the replay buffer.

        replay_buffer = ReplayBuffer(
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            observation_type=adapters.AdapterType.DefaultObservationImage,
            device_name="cpu",
        )

        (
            states,
            next_states,
            previous_actions,
            actions,
            rewards,
            dones,
        ) = generate_image_transitions(
            TRANSITIONS, STACK_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, ACTION_SIZE
        )

        for state, next_state, action, previous_action, reward, done in zip(
            states, next_states, actions, previous_actions, rewards, dones
        ):
            replay_buffer.add(
                state=state,
                next_state=next_state,
                action=action,
                prev_action=previous_action,
                reward=reward,
                done=done,
            )

        for _ in range(NUM_SAMPLES):
            sample = replay_buffer.sample()

            for state, action, reward, next_state, done, _ in zip(*sample):
                state = state.numpy()
                action = action.numpy()
                reward = reward.numpy()[0]
                next_state = next_state.numpy()
                done = True if done.numpy()[0] else False

                index_of_state = None
                for index, original_state in enumerate(states):
                    if np.array_equal(original_state, state):
                        index_of_state = index
                        break

                self.assertIn(state, states)
                self.assertIn(next_state, next_states)
                self.assertIn(action, actions)
                self.assertIn(reward, rewards)
                self.assertIn(done, dones)
                self.assertTrue(np.array_equal(state, states[index_of_state]))
                self.assertTrue(np.array_equal(next_state, next_states[index_of_state]))
                self.assertTrue(np.array_equal(action, actions[index_of_state]))
                self.assertEqual(reward, rewards[index_of_state])
                self.assertEqual(done, dones[index_of_state])

    # def test_vector_replay_buffer(self):
    #     # TRANSITIONS = 64
    #     # LOW_DIM_STATES_SIZE = 47
    #     # SOCIAL_CAPACITY = 10
    #     # SOCIAL_FEATURES = 4
    #     # ACTION_SIZE = 3
    #     # BUFFER_SIZE = 64
    #     # BATCH_SIZE = 32
    #     # NUM_SAMPLES = 1

    #     TRANSITIONS = 4
    #     LOW_DIM_STATES_SIZE = 47
    #     SOCIAL_CAPACITY = 10
    #     SOCIAL_FEATURES = 4
    #     ACTION_SIZE = 3
    #     BUFFER_SIZE = 4
    #     BATCH_SIZE = 2
    #     NUM_SAMPLES = 1

    #     states = np.asarray(
    #         [
    #             generate_vector_state(
    #                 LOW_DIM_STATES_SIZE, SOCIAL_CAPACITY, SOCIAL_FEATURES
    #             )
    #             for _ in range(TRANSITIONS)
    #         ]
    #     )
    #     next_states = np.concatenate(
    #         (
    #             states[1:],
    #             np.asarray(
    #                 [
    #                     generate_vector_state(
    #                         LOW_DIM_STATES_SIZE, SOCIAL_CAPACITY, SOCIAL_FEATURES
    #                     )
    #                 ]
    #             ),
    #         )
    #     )
    #     previous_actions = np.random.uniform(
    #         low=-1.0, high=1.0, size=(TRANSITIONS, ACTION_SIZE)
    #     ).astype(np.float32)
    #     actions = np.concatenate(
    #         (
    #             previous_actions[1:],
    #             np.random.uniform(low=-1.0, high=1.0, size=(1, ACTION_SIZE)).astype(
    #                 np.float32
    #             ),
    #         )
    #     )
    #     rewards = np.random.uniform(low=-10.0, high=10.0, size=(TRANSITIONS,)).astype(
    #         np.float32
    #     )
    #     dones = np.random.choice([True, False], size=(TRANSITIONS,))

    #     replay_buffer = ReplayBuffer(
    #         buffer_size=BUFFER_SIZE,
    #         batch_size=BATCH_SIZE,
    #         observation_type=adapters.AdapterType.DefaultObservationVector,
    #         device_name="cpu",
    #     )

    #     for state, next_state, action, previous_action, reward, done in zip(
    #         states, next_states, actions, previous_actions, rewards, dones
    #     ):
    #         print("state:", state)
    #         print("next_state:", next_state)
    #         # print("len(state):", len(state))
    #         # print("state[\"low_dim_states\"].shape:", state["low_dim_states"].shape)
    #         # print("state[\"social_vehicles\"].shape:", state["social_vehicles"].shape)
    #         # print("len(next_state):", len(next_state))
    #         # print("next_state[\"low_dim_states\"].shape:", next_state["low_dim_states"].shape)
    #         # print("next_state[\"social_vehicles\"].shape:", next_state["social_vehicles"].shape)

    #         replay_buffer.add(
    #             state=state,
    #             next_state=next_state,
    #             action=action,
    #             prev_action=previous_action,
    #             reward=reward,
    #             done=done,
    #         )

    #     for _ in range(NUM_SAMPLES):
    #         # print("sampling")
    #         sample = replay_buffer.sample()
    #         # # print("sample:", sample)
    #         # print("type(sample):", type(sample))
    #         # for element in sample:
    #         #     print(">>>>>>>> element:", element)

    #         (
    #             states_from_sample,
    #             actions,
    #             rewards,
    #             next_states_from_sample,
    #             dones,
    #             _,
    #         ) = sample
    #         states_low_dim_states = states_from_sample["low_dim_states"]
    #         states_social_vehicles = states_from_sample["social_vehicles"]
    #         next_states_low_dim_states = next_states_from_sample["low_dim_states"]
    #         next_states_social_vehicles = next_states_from_sample["social_vehicles"]

    #         for (
    #             state_low_dim_states,
    #             state_social_vehicles,
    #             action,
    #             reward,
    #             next_state_low_dim_states,
    #             next_state_social_vehicles,
    #             done,
    #         ) in zip(
    #             states_low_dim_states,
    #             states_social_vehicles,
    #             actions,
    #             rewards,
    #             next_states_low_dim_states,
    #             next_states_social_vehicles,
    #             dones,
    #         ):
    #             state_low_dim_states = np.asarray(state_low_dim_states.numpy()).astype(
    #                 np.float32
    #             )
    #             state_social_vehicles = np.asarray(
    #                 state_social_vehicles.numpy()
    #             ).astype(np.float32)
    #             action = action.numpy()
    #             reward = reward.numpy()[0]
    #             next_state_low_dim_states = np.asarray(
    #                 next_state_low_dim_states.numpy()
    #             ).astype(np.float32)
    #             next_state_social_vehicles = np.asarray(
    #                 next_state_social_vehicles.numpy()
    #             ).astype(np.float32)
    #             done = True if done.numpy()[0] else False

    #             index_of_state = None
    #             for index, original_state in enumerate(states):
    #                 print("os:", original_state["low_dim_states"])
    #                 print("sld:", state_low_dim_states)
    #                 print("oss:", original_state["social_vehicles"])
    #                 print("ssv:", state_social_vehicles)
    #                 print(
    #                     "osl == sld?",
    #                     np.array_equal(
    #                         original_state["low_dim_states"], state_low_dim_states
    #                     ),
    #                 )
    #                 print(
    #                     "oss == ssv?",
    #                     np.array_equal(
    #                         original_state["social_vehicles"], state_social_vehicles
    #                     ),
    #                 )
    #                 if np.array_equal(
    #                     original_state["low_dim_states"], state_low_dim_states
    #                 ) and np.array_equal(
    #                     original_state["social_vehicles"], state_social_vehicles
    #                 ):
    #                     index_of_state = index
    #                     break

    #             # print("states:", states)
    #             # print("states ios:", states[index_of_state])
    #             # print("index:", index_of_state)
    #             self.assertTrue(
    #                 np.array_equal(
    #                     state_low_dim_states, states[index_of_state]["low_dim_states"]
    #                 )
    #             )
    #             self.assertTrue(
    #                 np.array_equal(
    #                     state_social_vehicles, states[index_of_state]["social_vehicles"]
    #                 )
    #             )
    #             self.assertTrue(
    #                 np.array_equal(
    #                     next_state_low_dim_states,
    #                     next_states[index_of_state]["low_dim_states"],
    #                 )
    #             )
    #             self.assertTrue(
    #                 np.array_equal(
    #                     next_state_social_vehicles,
    #                     next_states[index_of_state]["social_vehicles"],
    #                 )
    #             )
    #             self.assertTrue(np.array_equal(action, actions[index_of_state]))
    #             self.assertEqual(reward, rewards[index_of_state])
    #             self.assertEqual(done, dones[index_of_state])

    #             # pass
    #             # print("state:", state)
    #             # print("next_state:", next_state)
    #             # print("action:", action)
    #             # print("reward:", reward)
    #             # print("done:", done)
    #             # print("-----------------------------------------")


def generate_image_transitions(
    num_transitions, stack_size, image_height, image_width, action_size
):
    states = np.random.uniform(
        low=0.0, high=1.0, size=(num_transitions, stack_size, image_height, image_width)
    ).astype(np.float32)
    next_states = np.concatenate(
        (
            states[1:],
            np.random.uniform(
                low=0.0, high=1.0, size=(1, stack_size, image_height, image_width)
            ).astype(np.float32),
        )
    )
    previous_actions = np.random.uniform(
        low=-1.0, high=1.0, size=(num_transitions, action_size)
    ).astype(np.float32)
    actions = np.concatenate(
        (
            previous_actions[1:],
            np.random.uniform(low=-1.0, high=1.0, size=(1, action_size)).astype(
                np.float32
            ),
        )
    )
    rewards = np.random.uniform(low=-10.0, high=10.0, size=(num_transitions,)).astype(
        np.float32
    )
    dones = np.random.choice([True, False], size=(num_transitions,))

    return states, next_states, previous_actions, actions, rewards, dones


# def generate_vector_state(low_dim_states_size, social_capacity, social_features):
#     state = {
#         "low_dim_states": np.random.uniform(
#             low=-1.0, high=1.0, size=(low_dim_states_size,)
#         ).astype(np.float32),
#         "social_vehicles": np.random.uniform(
#             low=-1.0, high=1.0, size=(social_capacity, social_features)
#         ).astype(np.float32),
#     }
#     return state
