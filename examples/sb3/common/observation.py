import gym


class Observation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["rgb"]

        print("ddddddddddddddddddddddd", self.observation_space)
        import sys
        print("exiting ==============================")
        sys.exit(2)

    def observation(self, obs):
        """Adapts the wrapped environment's observation.

        Note: Users should not directly call this method.
        """
        wrapped_obs = obs.rgb
        return wrapped_obs
