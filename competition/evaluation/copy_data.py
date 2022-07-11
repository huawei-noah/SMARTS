import copy
from typing import Any, Dict, Iterable, Tuple

import gym


class DataStore:
    def __init__(self):
        self._data = None
        self._agent_names = None

    def __call__(self, **kwargs):
        self._data = copy.deepcopy(dict(**kwargs))

    @property
    def data(self):
        return self._data

    @property
    def agent_names(self):
        return self._agent_names

    @agent_names.setter
    def agent_names(self, names: Iterable[str]):
        self._agent_names = copy.deepcopy(names)


class CopyData(gym.Wrapper):
    def __init__(self, env: gym.Env, datastore: DataStore):
        super(CopyData, self).__init__(env)
        self._datastore = datastore
        self._datastore.agent_names = env.agent_specs.keys()

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Steps the environment and makes a copy of info. The info copy is a private attribute and
        cannot be acccessed from outside.

        Args:
            action (Dict[str, Any]): Action for each agent.

        Returns:
            Tuple[ Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, Any]] ]:
                Observation, reward, done, and info, for each agent is returned.
        """
        obs, rewards, dones, infos = self.env.step(action)
        self._datastore(infos=infos, dones=dones)
        return obs, rewards, dones, infos
