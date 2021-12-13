def info_adapter(obs, reward, info):
    return info


def reward_adapter(obs, env_reward):
    ego = obs.ego_vehicle_state
    reward = 0

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 200
        return float(reward)

    # Penalty for colliding
    if len(obs.events.collisions) > 0:
        reward -= 200
        return float(reward)

    # Distance based reward
    reward += env_reward

    return float(reward)
