from ultra.baselines.agent_spec import BaselineAgentSpec


class RLlibAgent:
    def __init__(self, action_type, policy_class):
        self._spec = BaselineAgentSpec(
            action_type=action_type, policy_class=policy_class
        )

    @property
    def spec(self):
        return self._spec

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "speed": gym.spaces.Box(low=0, high=1e10, shape=(1,)),
                "relative_goal_position": gym.spaces.Box(
                    low=-1e10, high=1e10, shape=(1,)
                ),
                "distance_from_center": gym.spaces.Box(
                    low=-1e10, high=1e10, shape=(1,)
                ),
                "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
                "social_vehicles": gym.spaces.Box(low=0, high=1e10, shape=(1000,)),
                "road_speed": gym.spaces.Box(low=0, high=1e10, shape=(1,)),
                # ----------
                # don't normalize the following:
                "start": gym.spaces.Box(low=0, high=1e10, shape=(2,)),
                "goal": gym.spaces.Box(low=0, high=1e10, shape=(2,)),
                "heading": gym.spaces.Box(low=0, high=1e10, shape=(1,)),
                "goal_path": gym.spaces.Box(low=0, high=1e10, shape=(1000,)),
                "ego_position": gym.spaces.Box(low=0, high=1e10, shape=(2,)),
                "waypoint_paths": gym.spaces.Box(low=0, high=1e10, shape=(1000,)),
                "events": gym.spaces.Box(low=0, high=100, shape=(10,)),
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
