from typing import Any, Dict, Tuple

import gym
import copy

class DataStore:
    def __init__(self):
        self._data = None

    def __call__(self, data):
        self._data = copy.deepcopy(data)

    def get(self):
        return self._data


class CopyInfo(gym.Wrapper):
    def __init__(self, env: gym.Env, datastore:DataStore):
        super(CopyInfo, self).__init__(env)
        self._datastore=datastore

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
        obs, reward, done, info = self.env.step(action)
        self._datastore(info)
        return obs, reward, done, info