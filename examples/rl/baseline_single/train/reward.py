import gymnasium as gym
import numpy as np


class Reward(gym.Wrapper):
    def __init__(self, env):
        """Constructor for the Reward wrapper."""
        super().__init__(env)

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
        reward = {agent_id: np.float64(0) for agent_id in env_reward.keys()}

        leader_name = "Leader-007"
        leader=None
        for agent_id, agent_obs in obs.items():
            neighbor_vehicles = _get_neighbor_vehicles(obs=agent_obs, neighbor_name=leader_name)
            if neighbor_vehicles:
                leader=neighbor_vehicles[0]
                break 

        for agent_id, agent_reward in env_reward.items():
            # Penalty for colliding
            if obs[agent_id]["events"]["collisions"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Collided.")
                break

            # Penalty for driving off road
            if obs[agent_id]["events"]["off_road"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went off road.")
                break

            # Penalty for driving off route
            # if obs[agent_id]["events"]["off_route"]:
            #     reward[agent_id] -= np.float64(10)
            #     print(f"{agent_id}: Went off route.")
            #     break

            # Penalty for driving on road shoulder
            # if obs[agent_id]["events"]["on_shoulder"]:
            #     reward[agent_id] -= np.float64(1)
            #     print(f"{agent_id}: Went on shoulder.")
            #     break

            # Penalty for driving on wrong way
            if obs[agent_id]["events"]["wrong_way"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went wrong way.")
                break

            # Reward for reaching goal
            # if obs[agent_id]["events"]["reached_goal"]:
            #     reward[agent_id] += np.float64(30)

            # Reward for distance travelled
            reward[agent_id] += np.float64(agent_reward)

            # Reward for being in the same lane as the leader
            # reward[agent_id] += np.float64(agent_reward)

            ego_lane_idx = obs[agent_id]["ego_vehicle_state"]["lane_index"]
            leader_lane_idx = leader["lane_index"]
            if ego_lane_idx == leader_lane_idx:
                reward[agent_id] += np.float64(1)

            ego_lane_idx = obs[agent_id]["ego_vehicle_state"]["position"]
            leader_lane_idx = obs[agent_id]["neighborhood_vehicle_states"]["lane_index"]
            if ego_lane_idx == leader_lane_idx:
                reward[agent_id] += np.float64(1)
            

        return reward

def _get_neighbor_vehicles(obs, neighbor_name):
    neighbours = [neighbor for neighbor in zip(
        obs["neighborhood_vehicle_states"]["id"],
        obs["neighborhood_vehicle_states"]["heading"],
        obs["neighborhood_vehicle_states"]["lane_index"],
        obs["neighborhood_vehicle_states"]["position"],
        obs["neighborhood_vehicle_states"]["speed"]) if neighbor_name in neighbor[0]]

    print(neighbours)
    input("dddddddddddddddddddddddddddddddddddd")
    return neighbours