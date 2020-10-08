import numpy as np
from .observation_adapter import default_obs_adapter


def default_reward_adapter(env_obs, env_reward):
    obs = default_obs_adapter(env_obs)
    center_penalty = -np.abs(obs["distance_from_center"])

    # penalize flip occurences (taking into account that the vehicle spawns in the air)
    flip_penalty = 0
    if (
        env_obs.ego_vehicle_state.speed >= 25
        and env_obs.ego_vehicle_state.position[2] > 0.85
    ):
        flip_penalty = -2 * env_obs.ego_vehicle_state.speed

    # penalise sharp turns done at high speeds
    steering_penalty = 0
    if env_obs.ego_vehicle_state.speed > 60:
        steering_penalty = -pow(
            (env_obs.ego_vehicle_state.speed - 60)
            / 20
            * (env_obs.ego_vehicle_state.steering)
            * 45
            / 4,
            2,
        )

    # penalize close proximity to other cars
    crash_penalty = -5 if bool(obs["ego_will_crash"]) else 0

    total_reward = np.sum([1.0 * env_reward,])
    total_penalty = np.sum([0.1 * center_penalty, 1 * steering_penalty, crash_penalty])

    if flip_penalty != 0:
        return float((-total_reward + total_penalty) / 200.0)
    else:
        return float((total_reward + total_penalty) / 200.0)
