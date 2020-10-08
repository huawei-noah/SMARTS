import gym

DEFAULT_STATE_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        "speed_of_closest": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "proximity": gym.spaces.Box(low=-1e10, high=1e10, shape=(6,)),
        "headings_of_cars": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)


def default_state_adapter(state):
    return state
