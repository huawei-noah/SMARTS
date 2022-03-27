import random as python_random
from itertools import starmap

import numpy as np
import pickle

from rlkit.data_management.replay_buffer import AgentReplayBuffer


class AgentSimpleReplayBuffer(AgentReplayBuffer):
    """
    THE MAX LENGTH OF AN EPISODE SHOULD BE STRICTLY SMALLER THAN THE max_replay_buffer_size
    OTHERWISE THERE IS A BUG IN TERMINATE_EPISODE

    # It's a bit memory inefficient to save the observations twice,
    # but it makes the code *much* easier since you no longer have to
    # worry about termination conditions.
    """

    def __init__(
        self, max_replay_buffer_size, observation_dim, action_dim, random_seed=1995
    ):
        self._np_rand_state = np.random.RandomState(random_seed)

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size

        if isinstance(observation_dim, tuple):
            dims = [d for d in observation_dim]
            dims = [max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims)
            self._next_obs = np.zeros(dims)
        elif isinstance(observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([max_replay_buffer_size] + list(dims))
                else:
                    dims = (max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims)
                self._next_obs[key] = np.zeros(dims)
        else:
            # else observation_dim is an integer
            self._observations = np.zeros((max_replay_buffer_size, observation_dim))
            self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))

        self._actions = np.zeros((max_replay_buffer_size, action_dim))

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        self._timeouts = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        self._absorbing = np.zeros((max_replay_buffer_size, 2))
        self._top = 0
        self._size = 0
        self._trajs = 0

        # keeping track of trajectory boundaries
        # assumption is trajectory lengths are AT MOST the length of the entire replay buffer
        self._cur_start = 0
        self._traj_endpoints = {}  # start->end means [start, end)

    def _np_randint(self, *args, **kwargs):
        rets = self._np_rand_state.randint(*args, **kwargs)
        return rets

    def _np_choice(self, *args, **kwargs):
        rets = self._np_rand_state.choice(*args, **kwargs)
        return rets

    def clear(self):
        if isinstance(self._observation_dim, tuple):
            dims = [d for d in self._observation_dim]
            dims = [self._max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims)
            self._next_obs = np.zeros(dims)
        elif isinstance(self._observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in self._observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([self._max_replay_buffer_size] + list(dims))
                else:
                    dims = (self._max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims)
                self._next_obs[key] = np.zeros(dims)
        else:
            # else observation_dim is an integer
            self._observations = np.zeros(
                (self._max_replay_buffer_size, self._observation_dim)
            )
            self._next_obs = np.zeros(
                (self._max_replay_buffer_size, self._observation_dim)
            )

        self._actions = np.zeros((self._max_replay_buffer_size, self._action_dim))

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((self._max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((self._max_replay_buffer_size, 1), dtype="uint8")
        self._timeouts = np.zeros((self._max_replay_buffer_size, 1), dtype="uint8")
        self._top = 0
        self._size = 0
        self._trajs = 0

        # keeping track of trajectory boundaries
        # assumption is trajectory lengths are AT MOST the length of the entire replay buffer
        self._cur_start = 0
        self._traj_endpoints = {}  # start->end means [start, end)

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        timeout=False,
        **kwargs,
    ):
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._timeouts[self._top] = timeout

        if terminal:
            next_start = (self._top + 1) % self._max_replay_buffer_size
            self._traj_endpoints[self._cur_start] = next_start
            self._cur_start = next_start

        if isinstance(self._observations, dict):
            for key, obs in observation.items():
                self._observations[key][self._top] = obs
            for key, obs in next_observation.items():
                self._next_obs[key][self._top] = obs
        else:
            self._observations[self._top] = observation
            self._next_obs[self._top] = next_observation
        self._advance()

    def save_data(self, save_name):
        save_dict = {
            "observations": self._observations[: self._top],
            "actions": self._actions[: self._top],
            "next_observations": self._next_obs[: self._top],
            "terminals": self._terminals[: self._top],
            "timeouts": self._timeouts[: self._top],
            "rewards": self._rewards[: self._top],
            "agent_infos": [None] * len(self._observations[: self._top]),
            "env_infos": [None] * len(self._observations[: self._top]),
        }

        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)

    def terminate_episode(self):
        if self._cur_start != self._top:
            # if they are equal it means that the previous state was terminal
            # and was handled so there is no need to handle it again
            # THERE WILL BE A BUG HERE IS max_replay_buffer_size
            # IS NOT STRICTLY LARGER THAN MAX EPISODE LENGTH
            self._traj_endpoints[self._cur_start] = self._top
            self._cur_start = self._top

    def add_path(self, path):
        for (ob, action, reward, next_ob, terminal,) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
        ):
            self.add_sample(
                observation=ob,
                action=action,
                reward=reward,
                next_observation=next_ob,
                terminal=terminal,
            )

        self.terminate_episode()
        self._trajs += 1

    def get_traj_num(self):
        return self._trajs

    def _advance(self):
        if self._top in self._traj_endpoints:
            # this means that the step in the replay buffer
            # that we just overwrote was the start of a some
            # trajectory, so now the full trajectory is no longer
            # there and we should remove it
            del self._traj_endpoints[self._top]
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_all_trajs(
        self,
        keys=None,
        samples_per_traj=None,
    ):
        # samples_per_traj of None mean use all of the samples
        starts = list(self._traj_endpoints.keys())
        ends = map(lambda k: self._traj_endpoints[k], starts)

        if samples_per_traj is None:
            return list(
                starmap(lambda s, e: self._get_segment(s, e, keys), zip(starts, ends))
            )
        else:
            return list(
                starmap(
                    lambda s, e: self._get_samples_from_traj(
                        s, e, samples_per_traj, keys
                    ),
                    zip(starts, ends),
                )
            )

    def random_batch(
        self, batch_size, keys=None, multi_step=False, step_num=1, **kwargs
    ):
        indices = self._np_randint(0, self._size, batch_size)
        if multi_step:
            indices = self._np_randint(0, self._size - step_num, batch_size)

        return self._get_batch_using_indices(
            indices, keys=keys, multi_step=multi_step, step_num=step_num, **kwargs
        )

    def _get_batch_using_indices(
        self, indices, keys=None, multi_step=False, step_num=1
    ):
        if keys is None:
            keys = set(
                [
                    "observations",
                    "actions",
                    "rewards",
                    "terminals",
                    "next_observations",
                ]
            )
        if isinstance(self._observations, dict):
            obs_to_return = {}
            next_obs_to_return = {}
            for k in self._observations:
                if "observations" in keys:
                    obs_to_return[k] = self._observations[k][indices]
                if "next_observations" in keys:
                    next_obs_to_return[k] = self._next_obs[k][indices]
        else:
            obs_to_return = self._observations[indices]
            next_obs_to_return = self._next_obs[indices]

        ret_dict = {}
        if "observations" in keys:
            ret_dict["observations"] = obs_to_return
        if "actions" in keys:
            ret_dict["actions"] = self._actions[indices]
        if "rewards" in keys:
            ret_dict["rewards"] = self._rewards[indices]
        if "terminals" in keys:
            ret_dict["terminals"] = self._terminals[indices]
        if "next_observations" in keys:
            ret_dict["next_observations"] = next_obs_to_return

        if multi_step:
            next_next_obs_return = [None] * step_num
            for i in np.arange(1, step_num + 1):
                if isinstance(self._observations, dict):
                    next_next_obs_return[i - 1] = {}
                    for k in self._observations:
                        next_next_obs_return[i - 1][k] = self._next_obs[k][
                            (indices + i) % self._max_replay_buffer_size
                        ]
                else:
                    next_next_obs_return[i - 1] = self._next_obs[
                        (indices + i) % self._max_replay_buffer_size
                    ]

                for j, indice in enumerate(indices):
                    source_list = list(range(indice + 1, indice + i + 1))
                    target_list = list(self._traj_endpoints.values())
                    res = set(source_list) & set(target_list)
                    if (
                        len(res) > 0
                    ):  # there is a number in range(indice, indice+i+1) are a traj endpoint, this should be the last state of the traj
                        next_next_obs_return[i - 1][j] = self._next_obs[
                            list(res)[0] - 1
                        ]

                ret_dict["next{}_observations".format(i)] = next_next_obs_return[i - 1]

        # print(step_num, ret_dict.keys())
        return ret_dict

    def _get_segment(self, start, end, keys=None):
        if start < end or end == 0:
            if end == 0:
                end = self._max_replay_buffer_size
            return self._get_batch_using_indices(range(start, end), keys=keys)

        inds = list(range(start, self._max_replay_buffer_size)) + list(range(0, end))
        return self._get_batch_using_indices(inds, keys=keys)

    def _get_samples_from_traj(self, start, end, samples_per_traj, keys=None):
        # subsample a trajectory
        if start < end or end == 0:
            if end == 0:
                end = self._max_replay_buffer_size
            inds = range(start, end)
        else:
            inds = list(range(start, self._max_replay_buffer_size)) + list(
                range(0, end)
            )
        inds = self._np_choice(
            inds, size=samples_per_traj, replace=len(inds) < samples_per_traj
        )
        return self._get_batch_using_indices(inds, keys=keys)

    def sample_trajs(self, num_trajs, keys=None, samples_per_traj=None):
        # samples_per_traj of None mean use all of the samples
        keys_list = list(self._traj_endpoints.keys())
        starts = self._np_choice(
            keys_list, size=num_trajs, replace=len(keys_list) < num_trajs
        )
        ends = map(lambda k: self._traj_endpoints[k], starts)

        if samples_per_traj is None:
            return list(
                starmap(lambda s, e: self._get_segment(s, e, keys), zip(starts, ends))
            )
        else:
            return list(
                starmap(
                    lambda s, e: self._get_samples_from_traj(
                        s, e, samples_per_traj, keys
                    ),
                    zip(starts, ends),
                )
            )

    def num_steps_can_sample(self):
        return self._size


class SimpleReplayBufferDict(dict):
    def __init__(self, max_size, obs_dim, act_dim, seed):
        super().__init__()
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._py_rand_state = python_random.Random(seed)

    def __missing__(self, k):
        rand_seed = self._py_rand_state.randint(0, 10000)
        self[k] = AgentSimpleReplayBuffer(
            self.max_size, self.obs_dim, self.act_dim, rand_seed
        )
        return self[k]


class AgentMetaSimpleReplayBuffer:
    def __init__(
        self, max_rb_size_per_task, observation_dim, action_dim, random_seed=2001
    ):
        self._py_rand_state = python_random.Random(random_seed)

        self._obs_dim = observation_dim
        self._act_dim = action_dim
        self._max_rb_size_per_task = max_rb_size_per_task
        self.task_replay_buffers = SimpleReplayBufferDict(
            max_rb_size_per_task,
            observation_dim,
            action_dim,
            self._py_rand_state.randint(0, 10000),
        )

    def add_path(self, path, task_identifier):
        self.task_replay_buffers[task_identifier].add_path(path)

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        task_identifier,
        **kwargs,
    ):
        self.task_replay_buffers[task_identifier].add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )

    def terminate_episode(self, task_identifier):
        self.task_replay_buffers[task_identifier].terminate_episode()

    def sample_trajs(
        self,
        task_identifiers,
        num_trajs_per_task,
        samples_per_traj=None,
        keys=None,
    ):
        # if task_identifiers is None:
        # sample_params = list(sample(self.task_replay_buffers.keys(), num_tasks))
        batch_list = [
            self.task_replay_buffers[p].sample_trajs(
                num_trajs_per_task, keys=keys, samples_per_traj=samples_per_traj
            )
            for p in task_identifiers
        ]
        return batch_list, sample_params

    def sample_trajs_from_task(
        self, task_identifier, num_trajs, keyes=None, samples_per_traj=None
    ):
        return self.task_replay_buffers[task_identifier].sample_trajs(
            num_trajs, keys=keys, samples_per_traj=samples_per_traj
        )

    def sample_random_batch_from_task(self, task_identifier, num_samples, keys=None):
        return self.task_replay_buffers[task_identifier].random_batch(
            num_samples, keys=keys
        )

    def random_batch(self, task_identifiers, batch_size_per_task, keys=None):
        batch_list = [
            self.task_replay_buffers[p].random_batch(batch_size_per_task, keys=keys)
            for p in task_identifiers
        ]
        return batch_list

    def num_steps_can_sample(self):
        return sum(
            map(lambda rb: rb.num_steps_can_sample(), self.task_replay_buffers.values())
        )


def concat_nested_dicts(d1, d2):
    # two dicts that have the exact same nesting structure
    # and contain leaf values that are numpy arrays of the same
    # shape except for the first dimensions
    return {
        k: np.concatenate((d1[k], d2[k]), axis=0)
        if not isinstance(d1[k], dict)
        else concat_nested_dicts(d1[k], d2[k])
        for k in d1
    }
