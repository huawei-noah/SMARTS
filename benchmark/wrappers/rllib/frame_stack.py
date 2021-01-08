from collections import deque
from typing import Sequence

import numpy as np
import gym

from scipy.spatial import distance
from benchmark.common import cal_obs, ActionAdapter
from benchmark.wrappers.rllib.wrapper import Wrapper


class FrameStack(Wrapper):
    """ By default, this wrapper will stack 3 consecutive frames as an agent observation"""

    def __init__(self, config):
        super(FrameStack, self).__init__(config)
        self.num_stack = config["num_stack"]

        config = config["custom_config"]
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
            spaces = {}
            for k, space in observation_space.spaces.items():
                spaces[k] = FrameStack.get_observation_space(space, wrapper_config)
            return gym.spaces.Dict(spaces)
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
            observation = dict(
                map(lambda kv: (kv[0], np.stack(kv[1])), observation.items())
            )
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

    def step(self, agent_actions):
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
            obs_np = observation_adapter(env_obs_seq)

            # ======== Penalty: too close to neighbor vehicles
            # if the mean ttc or mean speed or mean dist is higher than before, get penalty
            # otherwise, get bonus
            last_env_obs = env_obs_seq[-1]
            neighbor_features_np = obs_np.get("neighbor", None)
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
            diff_dist_to_center_penalty = np.abs(
                obs_np["distance_to_center"][-2]
            ) - np.abs(obs_np["distance_to_center"][-1])
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
