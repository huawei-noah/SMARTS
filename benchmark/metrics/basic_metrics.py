# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
General metrics
"""

import numpy as np
import tableprint as tp

from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from scipy.spatial import distance

from smarts.core.utils.episodes import EpisodeLog


def agent_info_adapter(obs, shaped_reward: float, raw_info: dict):
    info = dict()
    info["speed"] = obs.ego_vehicle_state.speed
    info["collision"] = 1 if len(obs.events.collisions) > 0 else 0

    goal = obs.ego_vehicle_state.mission.goal
    goal_pos = goal.position
    ego_2d_pos = obs.ego_vehicle_state.position[:2]

    info["distance_to_goal"] = distance.euclidean(ego_2d_pos, goal_pos)
    info["events"] = obs.events

    return info


def min_max_mean(data: list):
    return {"min": np.min(data), "max": np.max(data), "mean": np.mean(data)}


@dataclass
class EvaluatedEpisode(EpisodeLog):
    ego_speed: dict = field(default_factory=lambda: defaultdict(lambda: []))
    num_collision: dict = field(default_factory=lambda: defaultdict(lambda: 0))
    distance_to_goal: dict = field(default_factory=lambda: defaultdict(lambda: 0.0))
    distance_to_ego_car: dict = field(default_factory=lambda: defaultdict(lambda: []))
    acceleration: dict = field(default_factory=lambda: defaultdict(lambda: 0.0))
    reach_goal: dict = field(default_factory=lambda: defaultdict(lambda: False))

    def record_step(self, observations, rewards, dones, infos):
        for agent_id, info in infos.items():
            if info.get("_group_info") is not None:
                for i, _info in enumerate(info["_group_info"]):
                    name = f"{agent_id}:AGENT-{i}"
                    self.ego_speed[name] = np.mean(_info["speed"])

                    self.num_collision[name] += len(_info["events"].collisions)

                    if dones[agent_id]:
                        self.reach_goal[name] = _info["events"].reached_goal
                        self.distance_to_goal[name] = _info["distance_to_goal"]
            else:
                self.ego_speed[agent_id].append(info["speed"])
                self.num_collision[agent_id] += len(info["events"].collisions)

                if dones[agent_id]:
                    self.reach_goal[agent_id] = info["events"].reached_goal
                    self.distance_to_goal[agent_id] = info["distance_to_goal"]

        self.steps += 1


MinMeanMax = namedtuple("MinMeanMax", "min, mean, max")


def get_statistics(data: list):
    return MinMeanMax(np.min(data), np.mean(data), np.max(data))


class Metric:
    def __init__(self, num_episode):
        self._logs = [EvaluatedEpisode() for _ in range(num_episode)]

    def log_step(self, observations, rewards, dones, infos, episode):
        self._logs[episode].record_step(observations, rewards, dones, infos)

    def compute(self):
        res = dict()

        for i, log in enumerate(self._logs):
            for agent_id in log.ego_speed.keys():
                speed_list = log.ego_speed[agent_id]

                if res.get(agent_id, None) is None:
                    res[agent_id] = {
                        "collision": [],
                        "goal_distance": [],
                        "reach_goal": [],
                        "speed_statis": dict(),
                    }

                res[agent_id]["collision"].append(log.num_collision[agent_id])
                res[agent_id]["goal_distance"].append(log.distance_to_goal[agent_id])
                res[agent_id]["reach_goal"].append(log.reach_goal[agent_id])
                res[agent_id]["speed_statis"][i] = get_statistics(speed_list)

        for agent_id, statis in res.items():
            statis["collision"] = get_statistics(statis["collision"])
            statis["goal_distance"] = get_statistics(statis["goal_distance"])
            statis["reach_goal_rate"] = sum(statis["reach_goal"]) / len(
                statis["reach_goal"]
            )

        col_width = 32

        with tp.TableContext(
            [
                "Agent ID",
                "Collision (min / mean / max)",
                "Goal distance (min / mean / max)",
                "Goal reached rate",
            ],
            width=col_width,
            style="round",
        ) as table:
            res = OrderedDict(sorted(res.items(), key=lambda x: x[0]))
            collision = [
                f"{e['collision'][0]:.2f} / {e['collision'][1]:.2f} / {e['collision'][2]:.2f}"
                for agent, e in res.items()
            ]
            goal_distances = [
                f"{e['goal_distance'][0]:.2f} / {e['goal_distance'][1]:.2f} / {e['goal_distance'][2]:.2f}"
                for agent, e in res.items()
            ]
            reach_goal_rate = [
                f"{e['reach_goal_rate']:.2f}" for agent, e in res.items()
            ]

            agent_ids = list(res.keys())

            for i in range(len(collision)):
                table(
                    (agent_ids[i], collision[i], goal_distances[i], reach_goal_rate[i],)
                )

        return res
