import numpy as np


class PathBuilder:
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.agent_path_builders = {a_id: AgentPathBuilder() for a_id in agent_ids}

    def keys(self):
        return self.agent_ids

    def __len__(self):
        return len(list(self.agent_path_builders.values())[0])

    def __getitem__(self, agent_id):
        return self.agent_path_builders[agent_id]

    def get_all_agent_dict(self, key):
        return {a_id: self.agent_path_builders[a_id][key] for a_id in self.agent_ids}


class AgentPathBuilder(dict):
    """
    Usage:
    ```
    path_builder = PathBuilder()
    path.add_sample(
        observations=1,
        actions=2,
        next_observations=3,
        ...
    )
    path.add_sample(
        observations=4,
        actions=5,
        next_observations=6,
        ...
    )

    path = path_builder.get_all_stacked()

    path['observations']
    # output: [1, 4]
    path['actions']
    # output: [2, 5]
    ```

    Note that the key should be "actions" and not "action" since the
    resulting dictionary will have those keys.
    """

    def __init__(self):
        super().__init__()
        self._path_length = 0

    def add_all(self, **key_to_value):
        for k, v in key_to_value.items():
            if k not in self:
                self[k] = [v]
            else:
                self[k].append(v)
        self._path_length += 1

    def get_all_stacked(self):
        raise NotImplementedError("Does not handle dict obs")
        # output_dict = dict()
        # for k, v in self.items():
        #     output_dict[k] = stack_list(v)
        # return output_dict

    def get_stacked(self, key):
        v = self.__getitem__(key)
        if isinstance(v[0], dict):
            raise NotImplementedError()
        return np.array(v)

    def __len__(self):
        return self._path_length


def stack_list(lst):
    if isinstance(lst[0], dict):
        return lst
    else:
        return np.array(lst)
