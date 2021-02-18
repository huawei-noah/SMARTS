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
from collections import deque
from typing import Sequence

import gym
import numpy as np
from ray import logger
from ray.rllib.models import Preprocessor
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from scipy.spatial import distance

from benchmark.common import ActionAdapter, cal_obs
from benchmark.wrappers.rllib.wrapper import Wrapper


def _get_preprocessor(space: gym.spaces.Space):
    if isinstance(space, gym.spaces.Tuple):
        preprocessor = TupleStackingPreprocessor
    else:
        preprocessor = get_preprocessor(space)
    return preprocessor


class TupleStackingPreprocessor(Preprocessor):
    @override(Preprocessor)
    def _init_shape(self, obs_space: gym.Space, options: dict):
        assert isinstance(self._obs_space, gym.spaces.Tuple)
        size = None
        self.preprocessors = []
        for i in range(len(self._obs_space.spaces)):
            space = self._obs_space.spaces[i]
            logger.debug("Creating sub-preprocessor for {}".format(space))
            preprocessor = _get_preprocessor(space)(space, self._options)
            self.preprocessors.append(preprocessor)
            if size is not None:
                assert size == preprocessor.size
            else:
                size = preprocessor.size
        return len(self._obs_space.spaces), size

    @override(Preprocessor)
    def transform(self, observation):
        self.check_shape(observation)
        array = np.zeros(self.shape[0] * self.shape[1])
        self.write(observation, array, 0)
        array.reshape(self.shape)
        return array

    @override(Preprocessor)
    def write(self, observation, array, offset):
        assert len(observation) == len(self.preprocessors), observation
        for o, p in zip(observation, self.preprocessors):
            p.write(o, array, offset)
            offset += p.size


class FrameStack(Wrapper):
    """ By default, this wrapper will stack 3 consecutive frames as an agent observation"""

    def __init__(self, config):
        super(FrameStack, self).__init__(config)
        config = config["custom_config"]
        self.num_stack = config["num_stack"]

        self.observation_adapter = config["observation_adapter"]
        self.action_adapter = config["action_adapter"]
        self.info_adapter = config["info_adapter"]
        self.reward_adapter = config["reward_adapter"]

        self.frames = dict.fromkeys(self._agent_keys, deque(maxlen=self.num_stack))

    @staticmethod
    def get_observation_space(observation_space, wrapper_config):
        frame_num = wrapper_config["num_stack"]
        if isinstance(observation_space, gym.spaces.Box):
            return gym.spaces.Tuple([observation_space] * frame_num)
        elif isinstance(observation_space, gym.spaces.Dict):
            # inner_spaces = {}
            # for k, space in observation_space.spaces.items():
            #     inner_spaces[k] = FrameStack.get_observation_space(space, wrapper_config)
            # dict_space = gym.spaces.Dict(spaces)
            return gym.spaces.Tuple([observation_space] * frame_num)
        else:
            raise TypeError(
                f"Unexpected observation space type: {type(observation_space)}"
            )

    @staticmethod
    def get_action_space(action_space, wrapper_config=None):
        return action_space

    @staticmethod
    def get_observation_adapter(
        observation_space, feature_configs, wrapper_config=None
    ):
        def func(env_obs_seq):
            assert isinstance(env_obs_seq, Sequence)
            observation = cal_obs(env_obs_seq, observation_space, feature_configs)
            return observation

        return func

    @staticmethod
    def get_action_adapter(action_type, action_space, wrapper_config=None):
        return ActionAdapter.from_type(action_type)

    @staticmethod
    def stack_frames(frames):
        proto = frames[0]

        if isinstance(proto, dict):
            res = dict()
            for key in proto.keys():
                res[key] = np.stack([frame[key] for frame in frames], axis=0)
        elif isinstance(proto, np.ndarray):
            res = np.stack(frames, axis=0)
        else:
            raise NotImplementedError

        return res

    @staticmethod
    def get_preprocessor():
        return TupleStackingPreprocessor

    def _get_observations(self, raw_frames):
        """Update frame stack with given single frames,
        then return nested array with given agent ids
        """

        for k, frame in raw_frames.items():
            self.frames[k].append(frame)

        agent_ids = list(raw_frames.keys())
        observations = dict.fromkeys(agent_ids)

        for k in agent_ids:
            observation = list(self.frames[k])
            observation = self.observation_adapter(observation)
            observations[k] = observation

        return observations

    def _get_rewards(self, env_observations, env_rewards):
        agent_ids = list(env_rewards.keys())
        rewards = dict.fromkeys(agent_ids, None)

        for k in agent_ids:
            rewards[k] = self.reward_adapter(list(self.frames[k]), env_rewards[k])
        return rewards

    def _get_infos(self, env_obs, rewards, infos):
        if self.info_adapter is None:
            return infos

        res = {}
        agent_ids = list(env_obs.keys())
        for k in agent_ids:
            res[k] = self.info_adapter(env_obs[k], rewards[k], infos[k])
        return res

    def step(self, agent_actions):
        agent_actions = {
            agent_id: self.action_adapter(action)
            for agent_id, action in agent_actions.items()
        }
        env_observations, env_rewards, dones, infos = super(FrameStack, self).step(
            agent_actions
        )

        observations = self._get_observations(env_observations)
        rewards = self._get_rewards(env_observations, env_rewards)
        infos = self._get_infos(env_observations, env_rewards, infos)
        self._update_last_observation(self.frames)

        return observations, rewards, dones, infos

    def reset(self):
        observations = super(FrameStack, self).reset()
        for k, observation in observations.items():
            _ = [self.frames[k].append(observation) for _ in range(self.num_stack)]
        self._update_last_observation(self.frames)
        return self._get_observations(observations)

    @staticmethod
    def get_reward_adapter(observation_adapter):
        def func(env_obs_seq, env_reward):
            penalty, bonus = 0.0, 0.0
            obs_seq = observation_adapter(env_obs_seq)

            # ======== Penalty: too close to neighbor vehicles
            # if the mean ttc or mean speed or mean dist is higher than before, get penalty
            # otherwise, get bonus
            last_env_obs = env_obs_seq[-1]
            neighbor_features_np = np.asarray([e.get("neighbor") for e in obs_seq])
            if neighbor_features_np is not None:
                new_neighbor_feature_np = neighbor_features_np[-1].reshape((-1, 5))
                mean_dist = np.mean(new_neighbor_feature_np[:, 0])
                mean_ttc = np.mean(new_neighbor_feature_np[:, 2])

                last_neighbor_feature_np = neighbor_features_np[-2].reshape((-1, 5))
                mean_dist2 = np.mean(last_neighbor_feature_np[:, 0])
                # mean_speed2 = np.mean(last_neighbor_feature[:, 1])
                mean_ttc2 = np.mean(last_neighbor_feature_np[:, 2])
                penalty += (
                    0.03 * (mean_dist - mean_dist2)
                    # - 0.01 * (mean_speed - mean_speed2)
                    + 0.01 * (mean_ttc - mean_ttc2)
                )

            # ======== Penalty: distance to goal =========
            goal = last_env_obs.ego_vehicle_state.mission.goal
            ego_2d_position = last_env_obs.ego_vehicle_state.position[:2]
            if hasattr(goal, "position"):
                goal_position = goal.position
            else:
                goal_position = ego_2d_position
            goal_dist = distance.euclidean(ego_2d_position, goal_position)
            penalty += -0.01 * goal_dist

            old_obs = env_obs_seq[-2]
            old_goal = old_obs.ego_vehicle_state.mission.goal
            old_ego_2d_position = old_obs.ego_vehicle_state.position[:2]
            if hasattr(old_goal, "position"):
                old_goal_position = old_goal.position
            else:
                old_goal_position = old_ego_2d_position
            old_goal_dist = distance.euclidean(old_ego_2d_position, old_goal_position)
            penalty += 0.1 * (old_goal_dist - goal_dist)  # 0.05

            # ======== Penalty: distance to the center
            distance_to_center_np = np.asarray(
                [e["distance_to_center"] for e in obs_seq]
            )
            diff_dist_to_center_penalty = np.abs(distance_to_center_np[-2]) - np.abs(
                distance_to_center_np[-1]
            )
            penalty += 0.01 * diff_dist_to_center_penalty[0]

            # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
            ego_events = last_env_obs.events
            # ::collision
            penalty += -50.0 if len(ego_events.collisions) > 0 else 0.0
            # ::off road
            penalty += -50.0 if ego_events.off_road else 0.0
            # ::reach goal
            if ego_events.reached_goal:
                bonus += 20.0

            # ::reached max_episode_step
            if ego_events.reached_max_episode_steps:
                penalty += -0.5
            else:
                bonus += 0.5

            # ======== Penalty: heading error penalty
            # if obs.get("heading_errors", None):
            #     heading_errors = obs["heading_errors"][-1]
            #     penalty_heading_errors = -0.03 * heading_errors[:2]
            #
            #     heading_errors2 = obs["heading_errors"][-2]
            #     penalty_heading_errors += -0.01 * (heading_errors[:2] - heading_errors2[:2])
            #     penalty += np.mean(penalty_heading_errors)

            # ======== Penalty: penalise sharp turns done at high speeds =======
            if last_env_obs.ego_vehicle_state.speed > 60:
                steering_penalty = -pow(
                    (last_env_obs.ego_vehicle_state.speed - 60)
                    / 20
                    * last_env_obs.ego_vehicle_state.steering
                    / 4,
                    2,
                )
            else:
                steering_penalty = 0
            penalty += 0.1 * steering_penalty

            # ========= Bonus: environment reward (distance travelled) ==========
            bonus += 0.05 * env_reward
            return bonus + penalty

        return func
