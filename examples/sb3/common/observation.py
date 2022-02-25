import gym


class Observation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        print("--------------------------------")
        print(env.observation_space)
        print(env.observation_space.keys())
        print("--------------------------------")
        self.observation_space = gym.spaces.Dict(
            {
                agent_id: space["rgb"] for agent_id, space in env.observation_space.items() 
            }
        )
    
    def observation(self, obs):
        """Adapts the wrapped environment's observation.

        Note: Users should not directly call this method.
        """
        wrapped_obs = {}
        for agent_id, agent_obs in obs.items():
            wrapped_obs.update({agent_id: agent_obs.rgb})

        return wrapped_obs