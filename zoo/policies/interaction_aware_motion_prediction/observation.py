import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from smarts.core.utils.math import position_to_ego_frame, wrap_value


class observation_adapter(object):
    def __init__(self, num_neighbors=5):
        self.num_neighbors = num_neighbors
        self.hist_steps = 11

    def cache(self, env_obs):
        ego = env_obs.ego_vehicle_state
        self.buffer[self.timestep][ego.id] = {
            "state": np.concatenate(
                [ego.position[:2], [ego.heading + np.pi / 2, ego.speed]]
            ).astype(np.float32)
        }

        neighbors = env_obs.neighborhood_vehicle_states
        for neighbor in neighbors:
            self.buffer[self.timestep][neighbor.id] = {
                "state": np.concatenate(
                    [
                        neighbor.position[:2],
                        [neighbor.heading + np.pi / 2, neighbor.speed],
                    ]
                ).astype(np.float32)
            }

    def reset(self):
        self.buffer = defaultdict(dict)
        self.timestep = 0

    def ego_history_process(self, id):
        ego_history = np.zeros(shape=(self.hist_steps, 5))
        timesteps = list(self.buffer.keys())
        idx = -1

        for t in reversed(timesteps):
            if id not in self.buffer[t] or idx < -self.hist_steps:
                break

            pos = self.buffer[t][id]["state"][:2]
            head = self.buffer[t][id]["state"][2]
            speed = self.buffer[t][id]["state"][3]
            ego_history[idx, :2] = self.transform(np.append(pos, [0]))[:2]
            ego_history[idx, 2] = self.adjust_heading(head)
            ego_history[idx, 3:] = np.array(
                (
                    speed * np.cos(self.adjust_heading(head)),
                    speed * np.sin(self.adjust_heading(head)),
                )
            )
            idx -= 1

        return ego_history

    def neighbor_history_process(self, ids):
        neighbor_history = np.zeros(shape=(self.num_neighbors, self.hist_steps, 5))

        for i, id in enumerate(ids):
            timesteps = list(self.buffer.keys())
            idx = -1

            for t in reversed(timesteps):
                if id not in self.buffer[t] or idx < -self.hist_steps:
                    break

                pos = self.buffer[t][id]["state"][:2]
                head = self.buffer[t][id]["state"][2]
                speed = self.buffer[t][id]["state"][3]
                neighbor_history[i, idx, :2] = self.transform(np.append(pos, [0]))[:2]
                neighbor_history[i, idx, 2] = self.adjust_heading(head)
                neighbor_history[i, idx, 3:] = np.array(
                    (
                        speed * np.cos(self.adjust_heading(head)),
                        speed * np.sin(self.adjust_heading(head)),
                    )
                )
                idx -= 1

        return neighbor_history

    def __call__(self, env_obs):
        self.current_pos = (
            env_obs.ego_vehicle_state.position,
            env_obs.ego_vehicle_state.heading + np.pi / 2,
        )
        self.cache(env_obs)
        ego_state = self.ego_history_process(env_obs.ego_vehicle_state.id)

        neighbors = {}
        for neighbor in env_obs.neighborhood_vehicle_states:
            neighbors[neighbor.id] = neighbor.position[:2]

        sorted_neighbors = sorted(
            neighbors.items(),
            key=lambda item: np.linalg.norm(item[1] - self.current_pos[0][:2]),
        )
        sorted_neighbors = sorted_neighbors[: self.num_neighbors]
        neighbor_ids = [neighbor[0] for neighbor in sorted_neighbors]

        neighbors_state = self.neighbor_history_process(neighbor_ids)
        self.obs = {"ego_state": ego_state, "neighbors_state": neighbors_state}

        return self.obs

    def transform(self, v):
        return position_to_ego_frame(v, self.current_pos[0], self.current_pos[1])

    def adjust_heading(self, h):
        return wrap_value(h - self.current_pos[1], -math.pi, math.pi)

    def ego_frame_dynamics(self, v):
        ego_v = v.copy()
        ego_v[0] = v[0] * np.cos(self.current_pos[1]) + v[1] * np.sin(
            self.current_pos[1]
        )
        ego_v[1] = v[1] * np.cos(self.current_pos[1]) - v[0] * np.sin(
            self.current_pos[1]
        )

        return ego_v

    def render(self):
        # plot agent
        ego = plt.Circle(
            (self.obs["ego_state"][-1, 0], self.obs["ego_state"][-1, 1]), 0.6, color="r"
        )
        plt.gca().add_patch(ego)
        plt.plot(self.obs["ego_state"][:, 0], self.obs["ego_state"][:, 1], "r")

        for i in range(self.num_neighbors):
            if self.obs["neighbors_state"][i][-1][0] != 0:
                neighbor = plt.Circle(
                    (
                        self.obs["neighbors_state"][i, -1, 0],
                        self.obs["neighbors_state"][i, -1, 1],
                    ),
                    0.6,
                    color="c",
                )
                plt.gca().add_patch(neighbor)
                plt.plot(
                    self.obs["neighbors_state"][i, :, 0],
                    self.obs["neighbors_state"][i, :, 1],
                    "c",
                )

        # plot map
        for i in range(self.obs["map"].shape[0]):
            if self.obs["map"][i, 10, 0] != 0:
                print(self.obs["map"][i].shape)
                plt.plot(self.obs["map"][i, :, 0], self.obs["map"][i, :, 1], "k--")

        # show
        plt.gca().set_aspect("equal")
        plt.show()
