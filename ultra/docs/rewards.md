# Rewards

## Environment Rewards

The raw rewards from the SMARTS environment is given by a calculation within SMARTS. SMARTS uses the distance travelled along the mission route per time step as the raw reward signal.

## Reward Adapters

A reward adapter takes a raw environment observation and a scalar-valued raw environment reward, and produces another scalar that combines the raw reward with reward-affecting environment observations.

For example, the reward adapter used by the baseline agents (see `ultra/baselines/adapter.py`) adds the raw reward with scaled observations to indicate a sense of performance. See the paraphrased version of the baseline agents' reward adapter below:
```python
ego_step_reward = 0.02 * min(speed_fraction, 1) * np.cos(angle_error)
...
ego_collision_reward = -1.0 if ego_collision else 0.0
ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
ego_wrong_way = -0.02 if ego_events.wrong_way else 0.0
...
ego_reached_goal = 1.0 if ego_events.reached_goal else 0.0
...
env_reward /= 100
...

rewards = [
    ...
    ego_collision_reward,
    ego_off_road_reward,
    ego_off_route_reward,
    ego_wrong_way,
    ...
    ego_reached_goal,
    ego_step_reward,
    env_reward,
    ...
]
return sum(rewards)
```
Where 'bad' ego observations (such as collisions and turning the wrong way) subtract from the total reward, and 'good' ego observations (such as reaching the goal) add to the total reward. Additionally, notice that the raw environment reward, `env_reward`, is proportionally added to the adapted reward.