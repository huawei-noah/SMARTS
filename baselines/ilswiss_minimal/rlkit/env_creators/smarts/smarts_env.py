from smarts_imitation import create_env

from rlkit.env_creators.base_env import BaseEnv


class SmartsEnv(BaseEnv):
    def __init__(self, vehicle_ids=None, **configs):
        super().__init__(**configs)

        # create underlying smarts simulator
        env_kwargs = configs["env_kwargs"]
        scenario_name = configs["scenario_name"]
        self._env = create_env(scenario_name, vehicle_ids=vehicle_ids, **env_kwargs)

        self.n_agents = self._env.n_agents
        self.agent_ids = self._env.agent_ids

        self.observation_space_n = dict(
            zip(
                self.agent_ids,
                [self._env.observation_space for _ in range(self._env.n_agents)],
            )
        )
        self.action_space_n = dict(
            zip(
                self.agent_ids,
                [self._env.action_space for _ in range(self._env.n_agents)],
            )
        )

    def __getattr__(self, attrname):
        if "_env" not in vars(self):
            raise AttributeError
        return getattr(self._env, attrname)

    def seed(self, seed):
        return self._env.seed(seed)

    def reset(self):
        return self._env.reset()

    def step(self, action_n):
        return self._env.step(action_n)

    def render(self, **kwargs):
        return self._env.render(**kwargs)
