import gymnasium as gym
import numpy as np


class Reward(gym.Wrapper):
    def __init__(self, env):
        """Constructor for the Reward wrapper."""
        super().__init__(env)
        self._half_pi = np.pi/2
        self._two_pi = 2 * np.pi

    def step(self, action):
        """Adapts the wrapped environment's step.

        Note: Users should not directly call this method.
        """

        obs, reward, terminated, truncated, info = self.env.step(action)       
        wrapped_reward = self._reward(obs, reward)

        for agent_id, agent_done in terminated.items():
            if agent_id != "__all__" and agent_done == True:
                if obs[agent_id]["events"]["reached_goal"]:
                    print(f"{agent_id}: Hooray! Reached goal.")
                    raise Exception(f"{agent_id}: Goal has been leaked to the ego agent!")
                elif obs[agent_id]["events"]["reached_max_episode_steps"]:
                    print(f"{agent_id}: Reached max episode steps.")
                elif (
                    obs[agent_id]["events"]["collisions"]
                    | obs[agent_id]["events"]["off_road"]
                    # | obs[agent_id]["events"]["off_route"]
                    # | obs[agent_id]["events"]["on_shoulder"]
                    # | obs[agent_id]["events"]["wrong_way"]
                ):
                    pass
                elif (
                    obs[agent_id]["events"]["agents_alive_done"]
                ):
                    print(f"{agent_id}: Agents alive done triggered.")
                else:
                    print("Events: ", obs[agent_id]["events"])
                    raise Exception("Episode ended for unknown reason.")

        return obs, wrapped_reward, terminated, truncated, info


    def _reward(self, obs, env_reward):
        reward = {agent_id: np.float64(0) for agent_id in obs.keys()}

        leader_name = "Leader-007"
        leader=None
        for agent_id, agent_obs in obs.items():
            neighbor_vehicles = _get_neighbor_vehicles(obs=agent_obs, neighbor_name=leader_name)
            if neighbor_vehicles:
                leader=neighbor_vehicles[0]
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

            # Reward for distance travelled
            reward[agent_id] += np.float64(env_reward[agent_id])

            # Rewards specific to "platooning" and "following" tasks
            if leader:

                # Ego's heading with respect to the map's coordinate system.
                # Note: All angles returned by smarts is with respect to the map's coordinate system.
                #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
                ego_heading = (agent_obs["ego_vehicle_state"]["heading"] + np.pi) % self._two_pi - np.pi
                ego_pos = agent_obs["ego_vehicle_state"]["position"]

                # Leader's angle with respect to the ego's position.
                # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
                #       Hence, map_angle = np.angle() - π/2
                leader_pos = leader["position"]
                rel_pos = leader_pos - ego_pos
                leader_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - self._half_pi
                leader_angle = (leader_angle + np.pi) % self._two_pi - np.pi

                # The angle by which ego agent should turn to face leader.
                angle_diff = leader_angle - ego_heading
                angle_diff = (angle_diff + np.pi) % self._two_pi - np.pi

                # Verify the leader is infront of the ego agent.
                leader_in_front = -self._half_pi < angle_diff < self._half_pi

                # print(f"ego_heading: {ego_heading*180/np.pi}")
                # print(f"leader_angle: {leader_angle*180/np.pi}")
                # print(f"leader_heading: {leader['heading']*180/np.pi}")
                # print(f"angle_diff: {angle_diff*180/np.pi}")

            if leader and leader_in_front:

                # Reward for being in the same lane as the leader
                ego_lane_idx = agent_obs["ego_vehicle_state"]["lane_index"]
                leader_lane_idx = leader["lane_index"]
                if ego_lane_idx == leader_lane_idx:
                    reward[agent_id] += np.float64(2)
                    print(f"{agent_id}: In the same lane.")

                # Reward for being within x meters of leader
                if np.linalg.norm(ego_pos - leader_pos) < 10:
                    reward[agent_id] += np.float64(2)
                    print(f"{agent_id}: Within radius.")

        # print("^^^^^^^^^^^^^^")
        return reward

def _get_neighbor_vehicles(obs, neighbor_name):
    keys = ["id","heading","lane_index","position","speed"]
    neighbors_tuple = [neighbor for neighbor in zip(
        obs["neighborhood_vehicle_states"]["id"],
        obs["neighborhood_vehicle_states"]["heading"],
        obs["neighborhood_vehicle_states"]["lane_index"],
        obs["neighborhood_vehicle_states"]["position"],
        obs["neighborhood_vehicle_states"]["speed"]) if neighbor_name in neighbor[0]]
    neighbors_dict = [dict(zip(keys,neighbor)) for neighbor in neighbors_tuple]
    return neighbors_dict