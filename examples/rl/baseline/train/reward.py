import gymnasium as gym
import numpy as np

from smarts.core.colors import SceneColors


class Reward(gym.Wrapper):
    def __init__(self, env):
        """Constructor for the Reward wrapper."""
        super().__init__(env)
        self._half_pi = np.pi / 2
        self._two_pi = 2 * np.pi
        self._leader_color = np.array(SceneColors.SocialAgent.value[0:3]) * 255
        self._total_dist = {}

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
            self._total_dist[agent_id] = self._total_dist.get(agent_id,0) + agent_obs['distance_travelled']

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
                    # | agent_obs["events"]["off_route"]
                    # | agent_obs["events"]["on_shoulder"]
                    # | agent_obs["events"]["wrong_way"]
                ):
                    pass
                elif agent_obs["events"]["agents_alive_done"]:
                    print(f"{agent_id}: Agents alive done triggered.")
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

        leader_name = "Leader-007"
        leader = None
        for agent_id, agent_obs in obs.items():
            neighbor_vehicles = _get_neighbor_vehicles(
                obs=agent_obs, neighbor_name=leader_name
            )
            if neighbor_vehicles:
                leader = neighbor_vehicles[0]
                break

        for agent_id, agent_obs in obs.items():
            # Penalty for colliding
            if agent_obs["events"]["collisions"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Collided.")
                break

            # Penalty for driving off road
            if agent_obs["events"]["off_road"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went off road.")
                break

            # Penalty for driving off route
            # if agent_obs["events"]["off_route"]:
            #     reward[agent_id] -= np.float64(10)
            #     print(f"{agent_id}: Went off route.")
            #     break

            # Penalty for driving on road shoulder
            # if agent_obs["events"]["on_shoulder"]:
            #     reward[agent_id] -= np.float64(1)
            #     print(f"{agent_id}: Went on shoulder.")
            #     break

            # Penalty for driving on wrong way
            # if agent_obs["events"]["wrong_way"]:
            #     reward[agent_id] -= np.float64(10)
            #     print(f"{agent_id}: Went wrong way.")
            #     break

            # Reward for reaching goal
            # if agent_obs["events"]["reached_goal"]:
            #     reward[agent_id] += np.float64(30)

            # Reward for distance travelled by driving
            reward[agent_id] += np.float64(env_reward[agent_id])

            # # Check if leader is in front within visual angle
            # if leader:

            #     # Ego's heading with respect to the map's coordinate system.
            #     # Note: All angles returned by smarts is with respect to the map's coordinate system.
            #     #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
            #     ego_heading = (agent_obs["ego_vehicle_state"]["heading"] + np.pi) % self._two_pi - np.pi
            #     ego_pos = agent_obs["ego_vehicle_state"]["position"]

            #     # Leader's angle with respect to the ego's position.
            #     # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
            #     #       Hence, map_angle = np.angle() - π/2
            #     leader_pos = leader["position"]
            #     rel_pos = leader_pos - ego_pos
            #     leader_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - self._half_pi
            #     leader_angle = (leader_angle + np.pi) % self._two_pi - np.pi

            #     # The angle by which ego agent should turn to face leader.
            #     angle_diff = leader_angle - ego_heading
            #     angle_diff = (angle_diff + np.pi) % self._two_pi - np.pi

            #     # Verify the leader is infront of the ego agent.
            #     leader_in_front = -self._half_pi < angle_diff < self._half_pi

            #     # print(f"ego_heading: {ego_heading*180/np.pi}")
            #     # print(f"leader_angle: {leader_angle*180/np.pi}")
            #     # print(f"leader_heading: {leader['heading']*180/np.pi}")
            #     # print(f"angle_diff: {angle_diff*180/np.pi}")

            # Check if leader is in front and within the rgb observation
            if leader:
                rgb = agent_obs["top_down_rgb"]
                h, w, d = rgb.shape
                rgb_masked = rgb[0 : h // 2, :, :]
                leader_in_rgb = (
                    (rgb_masked == self._leader_color.reshape((1, 1, 3)))
                    .all(axis=-1)
                    .any()
                )

                # from contrib_policy.helper import plotter3d
                # print("-----------------------------")
                # plotter3d(obs=rgb_masked,rgb_gray=3,channel_order="last",pause=0)
                # print("-----------------------------")

            # Rewards specific to "platooning" and "following" tasks
            if leader and leader_in_rgb:

                # Reward for being in the same lane as the leader
                ego_lane_idx = agent_obs["ego_vehicle_state"]["lane_index"]
                leader_lane_idx = leader["lane_index"]
                if ego_lane_idx == leader_lane_idx:
                    reward[agent_id] += np.float64(1)
                    # print(f"{agent_id}: In the same lane.")

                # Reward for being within x meters of leader
                # if np.linalg.norm(ego_pos - leader_pos) < 15:
                # reward[agent_id] += np.float64(1)
                # print(f"{agent_id}: Within radius.")

            else:
                reward[agent_id] -= np.float64(0.2)

        # print("^^^^^^^^^^^^^^")
        return reward


def _get_neighbor_vehicles(obs, neighbor_name):
    keys = ["id", "heading", "lane_index", "position", "speed"]
    neighbors_tuple = [
        neighbor
        for neighbor in zip(
            obs["neighborhood_vehicle_states"]["id"],
            obs["neighborhood_vehicle_states"]["heading"],
            obs["neighborhood_vehicle_states"]["lane_index"],
            obs["neighborhood_vehicle_states"]["position"],
            obs["neighborhood_vehicle_states"]["speed"],
        )
        if neighbor_name in neighbor[0]
    ]
    neighbors_dict = [dict(zip(keys, neighbor)) for neighbor in neighbors_tuple]
    return neighbors_dict


def _point_in_rectangle(x1, y1, x2, y2, x, y):
    # bottom-left (x1, y1)
    # top-right (x2, y2)
    if x > x1 and x < x2 and y > y1 and y < y2:
        return True
    else:
        return False
