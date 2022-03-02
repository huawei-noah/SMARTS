import gym

from smarts.env.wrappers import format_obs


class Observation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["rgb"]

    def observation(self, obs: format_obs.StdObs):
        """Adapts the wrapped environment's observation.

        Note: Users should not directly call this method.
        """
        wrapped_obs = obs.rgb
        return wrapped_obs
