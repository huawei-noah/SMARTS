import numpy as np


# ==============================================================================
# -- Functions -----------------------------------------------------------------
# ==============================================================================
def get_split_batch(batch):
    """memory.sample() returns a batch of experiences, but we want an array
    for each element in the memory (s, a, r, s', done)"""
    states_mb = np.array([each[0][0] for each in batch])
    # print(states_mb.shape)
    actions_mb = np.array([each[0][1] for each in batch])
    # print(actions_mb.shape)
    rewards_mb = np.array([each[0][2] for each in batch])
    # print(rewards_mb.shape)
    next_states_mb = np.array([each[0][3] for each in batch])
    # print(next_states_mb.shape)
    dones_mb = np.array([each[0][4] for each in batch])

    return states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb


def OU(action, mu=0, theta=0.15, sigma=0.3):
    # noise = np.ones(action_dim) * mu
    noise = theta * (mu - action) + sigma * np.random.randn(1)
    # noise = noise + d_noise
    return noise


def calculate_angle(ego_location, goal_location, ego_direction):
    # calculate vector direction
    goal_location = np.array(goal_location)
    ego_location = np.array(ego_location)
    goal_vector = goal_location - ego_location
    L_g_vector = np.sqrt(goal_vector.dot(goal_vector))
    ego_vector = np.array(
        [np.cos(ego_direction * np.pi / 180), np.sin(ego_direction * np.pi / 180)]
    )
    L_e_vector = np.sqrt(ego_vector.dot(ego_vector))
    cos_angle = goal_vector.dot(ego_vector) / (L_g_vector * L_e_vector)
    angle = (np.arccos(cos_angle)) * 180 / np.pi
    if np.cross(goal_vector, ego_vector) > 0:
        angle = -angle
    return angle


def calculate_distance(location_a, location_b):
    """ calculate distance between a and b"""
    return np.linalg.norm(location_a - location_b)
