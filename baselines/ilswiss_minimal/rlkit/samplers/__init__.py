from typing import Dict
import numpy as np

from rlkit.data_management.path_builder import PathBuilder


class PathSampler:
    def __init__(
        self,
        env,
        vec_env,
        policy_n,
        policy_mapping_dict,
        num_steps,
        max_path_length,
        car_num,
        no_terminal=False,
        render=False,
        render_kwargs={},
    ):
        """
        When obtain_samples is called, the path sampler will generates the
        minimum number of rollouts such that at least num_steps timesteps
        have been sampled
        """
        self.env = env
        self.vec_env = vec_env
        self.env_num = vec_env.env_num
        self.wait_num = vec_env.wait_num
        self.car_num = car_num
        self.policy_n = policy_n
        self.policy_mapping_dict = policy_mapping_dict
        self.num_steps = num_steps
        self.max_path_length = max_path_length
        self.no_terminal = no_terminal
        self.render = render
        self.render_kwargs = render_kwargs
        self.agent_ids = self.env.agent_ids
        self.n_agents = self.env.n_agents

        self.observations_n = self.vec_env.reset()
        self.actions_n = np.array(
            [
                {
                    a_id: self.env.action_space_n[a_id].sample()
                    for a_id in self.agent_ids
                }
                for _ in range(self.env_num)
            ]
        )
        self._ready_env_ids = np.arange(self.env_num)
        self.path_builders = [PathBuilder(self.agent_ids) for _ in range(self.env_num)]

    def obtain_samples(self, num_steps=None, pred_obs=False):
        paths = []
        total_steps = 0
        if num_steps is None:
            num_steps = self.num_steps

        finished_env_ids = []
        env_finished_car_num = np.zeros(self.env_num)
        while True:
            self.actions_n[self._ready_env_ids] = self._get_action_and_info(
                self.observations_n[self._ready_env_ids],
            )

            (
                next_observations_n,
                rewards_n,
                terminals_n,
                env_infos_n,
            ) = self.vec_env.step(
                self.actions_n[self._ready_env_ids].copy(), id=self._ready_env_ids
            )
            self._ready_env_ids = np.array([i["env_id"] for i in env_infos_n])

            for (
                observation_n,
                action_n,
                reward_n,
                next_observation_n,
                terminal_n,
                env_info_n,
                env_id,
            ) in zip(
                self.observations_n[self._ready_env_ids],
                self.actions_n[self._ready_env_ids],
                rewards_n,
                next_observations_n,
                terminals_n,
                env_infos_n,
                self._ready_env_ids,
            ):
                for a_id in self.agent_ids:
                    self.path_builders[env_id][a_id].add_all(
                        observations=observation_n[a_id],
                        actions=action_n[a_id],
                        rewards=reward_n[a_id],
                        next_observations=next_observation_n[a_id],
                        terminals=terminal_n[a_id],
                        env_infos=env_info_n[a_id],
                    )

            self.observations_n[self._ready_env_ids] = next_observations_n

            terminals_all = [
                np.all(list(terminal.values())) for terminal in terminals_n
            ]
            for env_id, terminal in zip(self._ready_env_ids, terminals_all):
                if terminal or len(self.path_builders[env_id]) >= self.max_path_length:
                    paths.append(self.path_builders[env_id])
                    total_steps += len(self.path_builders[env_id])
                    self.path_builders[env_id] = PathBuilder(self.agent_ids)
                    env_finished_car_num[env_id] += 1
                    if not terminal or not self.vec_env.auto_reset:
                        self.observations_n[env_id] = self.vec_env.reset(id=env_id)[0]
                    if env_finished_car_num[env_id] == self.car_num[env_id]:
                        finished_env_ids.append(env_id)

            self._ready_env_ids = np.array(
                [x for x in self._ready_env_ids if x not in finished_env_ids]
            )

            if len(finished_env_ids) == self.env_num:
                assert len(self._ready_env_ids) == 0
                break

        self._ready_env_ids = np.arange(self.env_num)

        return paths

    def _get_action_and_info(self, observations_n):
        """
        Get an action to take in the environment.
        :param observation_n:
        :return:
        """
        action_n = [{} for _ in range(len(observations_n))]
        for agent_id in self.agent_ids:
            policy_id = self.policy_mapping_dict[agent_id]
            _observations = []
            _idxes = []
            for idx, observation_n in enumerate(observations_n):
                if agent_id in observation_n:
                    _observations.append(observation_n[agent_id])
                    _idxes.append(idx)
            _actions = self.policy_n[policy_id].get_actions(
                np.stack(_observations, axis=0)
            )
            for idx, action in zip(_idxes, _actions):
                action_n[idx][agent_id] = action
        return action_n


class MultiagentPathSampler:
    def __init__(
        self,
        env,
        vec_env,
        policy_n,
        policy_mapping_dict,
        num_steps,
        max_path_length,
        car_num,
        no_terminal=False,
        render=False,
        render_kwargs={},
    ):
        """
        When obtain_samples is called, the path sampler will generates the
        minimum number of rollouts such that at least num_steps timesteps
        have been sampled
        """
        self.env = env
        self.vec_env = vec_env
        self.env_num = vec_env.env_num
        self.wait_num = vec_env.wait_num
        self.car_num = car_num
        self.policy_n = policy_n
        self.policy_mapping_dict = policy_mapping_dict
        self.num_steps = num_steps
        self.max_path_length = max_path_length
        self.no_terminal = no_terminal
        self.render = render
        self.render_kwargs = render_kwargs
        self.agent_ids = self.env.agent_ids
        self.n_agents = self.env.n_agents

        self.observations_n = self.vec_env.reset()
        self.actions_n = np.array(
            [
                {
                    a_id: self.env.action_space_n[a_id].sample()
                    for a_id in self.agent_ids
                }
                for _ in range(self.env_num)
            ]
        )
        self._ready_env_ids = np.arange(self.env_num)
        self.path_builders = [PathBuilder(self.agent_ids) for _ in range(self.env_num)]

    def obtain_samples(self, num_steps=None, pred_obs=False):
        paths = []
        total_steps = 0
        if num_steps is None:
            num_steps = self.num_steps

        finished_env_ids = []
        env_finished_car_num = np.zeros(self.env_num)
        self.n_agents = self.env.n_agents
        _terminals_all = np.zeros((self.env_num), dtype=int)
        while True:
            self.actions_n[self._ready_env_ids] = self._get_action_and_info(
                self.observations_n[self._ready_env_ids],
            )

            (
                next_observations_n,
                rewards_n,
                terminals_n,
                env_infos_n,
            ) = self.vec_env.step(
                self.actions_n[self._ready_env_ids].copy(), id=self._ready_env_ids
            )
            self._ready_env_ids = np.array([i["env_id"] for i in env_infos_n])

            for (
                observation_n,
                action_n,
                reward_n,
                next_observation_n,
                terminal_n,
                env_info_n,
                env_id,
            ) in zip(
                self.observations_n[self._ready_env_ids],
                self.actions_n[self._ready_env_ids],
                rewards_n,
                next_observations_n,
                terminals_n,
                env_infos_n,
                self._ready_env_ids,
            ):
                for a_id in observation_n.keys():
                    if a_id not in next_observation_n or a_id not in reward_n:
                        continue
                    self.path_builders[env_id][a_id].add_all(
                        observations=observation_n[a_id],
                        actions=action_n[a_id],
                        rewards=reward_n[a_id],
                        next_observations=next_observation_n[a_id],
                        terminals=terminal_n[a_id],
                        env_infos=env_info_n[a_id],
                    )

            self.observations_n[self._ready_env_ids] = next_observations_n

            step_terminals = [
                np.sum(np.array(list(terminal.values()), dtype="int"))
                for terminal in terminals_n
            ]
            _terminals_all[self._ready_env_ids] = (
                _terminals_all[self._ready_env_ids] + step_terminals
            )

            # if np.any(_terminals_all > self.n_agents):
            #     pdb.set_trace()

            for env_id, terminal in zip(
                self._ready_env_ids, _terminals_all[self._ready_env_ids]
            ):
                if (
                    terminal == self.n_agents
                    or len(self.path_builders[env_id]) >= self.max_path_length
                ):
                    paths.append(self.path_builders[env_id])
                    total_steps += len(self.path_builders[env_id])
                    self.path_builders[env_id] = PathBuilder(self.agent_ids)
                    env_finished_car_num[env_id] += self.n_agents
                    if terminal != self.n_agents or not self.vec_env.auto_reset:
                        _terminals_all[env_id] = 0
                        self.observations_n[env_id] = self.vec_env.reset(id=env_id)[0]
                    if (
                        env_finished_car_num[env_id]
                        > self.car_num[env_id] - self.n_agents
                    ):
                        finished_env_ids.append(env_id)

            self._ready_env_ids = np.array(
                [x for x in self._ready_env_ids if x not in finished_env_ids]
            )

            if len(finished_env_ids) == self.env_num:
                assert len(self._ready_env_ids) == 0
                break

        self._ready_env_ids = np.arange(self.env_num)

        return paths

    def _get_action_and_info(self, observations_n):
        """
        Get an action to take in the environment.
        :param observation_n:
        :return:
        """
        action_n = [{} for _ in range(len(observations_n))]
        for agent_id in self.agent_ids:
            policy_id = self.policy_mapping_dict[agent_id]
            _observations = []
            _idxes = []
            for idx, observation_n in enumerate(observations_n):
                if agent_id in observation_n:
                    _observations.append(observation_n[agent_id])
                    _idxes.append(idx)
            if len(_observations) == 0:
                continue
            _actions = self.policy_n[policy_id].get_actions(
                np.stack(_observations, axis=0)
            )
            for idx, action in zip(_idxes, _actions):
                action_n[idx][agent_id] = action
        return action_n
