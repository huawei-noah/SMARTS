from typing import Tuple

import gymnasium as gym
import numpy as np

from smarts.core.colors import SceneColors


class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env, crop: Tuple[int, int, int, int]):
        """Constructor for the Reward wrapper."""
        super().__init__(env)
        self._leader_color = np.array(SceneColors.SocialAgent.value[0:3]) * 255
        self._total_dist = {}
        self._crop = crop

    def reset(self, *, seed=None, options=None):
        self._total_dist = {}
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Adapts the wrapped environment's step.

        Note: Users should not directly call this method.
        """

        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        for agent_id, agent_obs in obs.items():
            # Accumulate total distance travelled
            self._total_dist[agent_id] = (
                self._total_dist.get(agent_id, 0) + agent_obs["distance_travelled"]
            )

            # If agent is done
            if terminated[agent_id] == True:
                if agent_obs["events"]["reached_goal"]:
                    print(f"{agent_id}: Hooray! Reached goal.")
                    raise Exception(
                        f"{agent_id}: Goal has been leaked to the ego agent!"
                    )
                elif agent_obs["events"]["reached_max_episode_steps"]:
                    print(f"{agent_id}: Reached max episode steps.")
                elif (
                    agent_obs["events"]["collisions"]
                    | agent_obs["events"]["off_road"]
                    | agent_obs["events"]["wrong_way"]
                ):
                    pass
                elif agent_obs["events"]["interest_done"]:
                    print(f"{agent_id}: Interest done triggered.")
                else:
                    print("Events: ", agent_obs["events"])
                    raise Exception("Episode ended for unknown reason.")

                print(
                    f"{agent_id}: Steps = {agent_obs['steps_completed']} "
                    f"{agent_id}: Dist = {self._total_dist[agent_id]:.2f}"
                )

        return obs, wrapped_reward, terminated, truncated, info

    def _reward(self, obs, env_reward):
        reward = {agent_id: np.float64(0) for agent_id in obs.keys()}

        leader = None
        max_agent_steps_completed = 0
        for agent_id, agent_obs in obs.items():
            interest_vehicles = _get_interest_vehicles(obs=agent_obs)
            max_agent_steps_completed = max(
                max_agent_steps_completed, agent_obs["steps_completed"]
            )
            if interest_vehicles:
                leader = interest_vehicles[0]
                break

        if leader == None and max_agent_steps_completed == 1:
            raise Exception("Leader is not present on ego's first step.")

        for agent_id, agent_obs in obs.items():
            # Penalty for colliding
            if agent_obs["events"]["collisions"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Collided.")
                continue

            # Penalty for driving off road
            if agent_obs["events"]["off_road"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went off road.")
                continue

            # Penalty for driving on wrong way
            if obs[agent_id]["events"]["wrong_way"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went wrong way.")
                continue

            # Check if leader is in front and within the rgb observation
            if leader:
                rgb = agent_obs["top_down_rgb"]
                h, w, d = rgb.shape
                rgb_masked = rgb[0 : h // 2, :, :]
                rgb_cropped = rgb_masked[
                    self._crop[2] :, self._crop[0] : w - self._crop[1], :
                ]
                leader_in_rgb = (
                    (rgb_cropped == self._leader_color.reshape((1, 1, 3)))
                    .all(axis=-1)
                    .any()
                )

            # Rewards specific to vehicle following task
            if leader and leader_in_rgb:
                # Get agent's waypoints
                waypoints = agent_obs["waypoint_paths"]["position"]

                # Find the nearest waypoint paths to the ego
                ego_pos = agent_obs["ego_vehicle_state"]["position"]
                dist = np.linalg.norm(waypoints[:, 0, :] - ego_pos, axis=-1)
                ego_wp_inds = np.where(dist == dist.min())[0]

                # Find the nearest waypoint index, if any, to the leader
                leader_wp_ind, leader_ind = _nearest_waypoint(
                    waypoints,
                    np.array([leader["position"]]),
                )

                # Reward for being in the same lane as the leader
                if (leader_ind is not None) and (leader_wp_ind[0] in ego_wp_inds):
                    reward[agent_id] += np.float64(3)

            else:
                reward[agent_id] -= np.float64(0.3)

        return reward


def _get_interest_vehicles(obs):
    keys = ["interest", "id", "heading", "lane_index", "position", "speed"]
    interest_vehicles = [
        neighbor
        for neighbor in zip(
            obs["neighborhood_vehicle_states"]["interest"],
            obs["neighborhood_vehicle_states"]["id"],
            obs["neighborhood_vehicle_states"]["heading"],
            obs["neighborhood_vehicle_states"]["lane_index"],
            obs["neighborhood_vehicle_states"]["position"],
            obs["neighborhood_vehicle_states"]["speed"],
        )
        if neighbor[0] == 1
    ]
    interest_vehicles_dict = [
        dict(zip(keys, interest_vehicle)) for interest_vehicle in interest_vehicles
    ]
    return interest_vehicles_dict


def _nearest_waypoint(matrix: np.ndarray, points: np.ndarray, radius: float = 1):
    """
    Returns
        (i) the `matrix` index of the nearest waypoint to the ego, which has a nearby `point`.
        (ii) the `points` index which is nearby the nearest waypoint to the ego.

    Nearby is defined as a point within `radius` of a waypoint.

    Args:
        matrix (np.ndarray): Waypoints matrix.
        points (np.ndarray): Points matrix.
        radius (float, optional): Nearby radius. Defaults to 2.

    Returns:
        Tuple[(int, int), Optional[int]] : `matrix` index of shape (a,b) and scalar `point` index.
    """
    cur_point_index = ((np.intp(1e10), np.intp(1e10)), None)

    if points.shape == (0,):
        return cur_point_index

    assert len(matrix.shape) == 3
    assert matrix.shape[2] == 3
    assert len(points.shape) == 2
    assert points.shape[1] == 3

    points_expanded = np.expand_dims(points, (1, 2))
    diff = matrix - points_expanded
    dist = np.linalg.norm(diff, axis=-1)
    for ii in range(points.shape[0]):
        index = np.argmin(dist[ii])
        index_unravel = np.unravel_index(index, dist[ii].shape)
        min_dist = dist[ii][index_unravel]
        if min_dist <= radius and index_unravel[1] < cur_point_index[0][1]:
            cur_point_index = (index_unravel, ii)

    return cur_point_index
