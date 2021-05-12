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

import csv
import os
import time
from collections import defaultdict

import numpy as np
from scipy.spatial import distance

from benchmark.common import CalObs
from benchmark.metrics import MetricHandler
from benchmark.metrics.basic_metrics import BehaviorMetric
from benchmark.utils import episode_log, format, plot


def agent_info_adapter(env_obs, shaped_reward: float, raw_info: dict):
    info = dict()
    info["speed"] = env_obs.ego_vehicle_state.speed
    info["collision"] = 1 if len(env_obs.events.collisions) > 0 else 0

    goal = env_obs.ego_vehicle_state.mission.goal
    goal_pos = goal.position
    ego_2d_pos = env_obs.ego_vehicle_state.position[:2]

    info["distance_to_goal"] = distance.euclidean(ego_2d_pos, goal_pos)
    info["distance_to_center"] = CalObs.cal_distance_to_center(env_obs, "")

    return info


class MetricKeys:
    AVE_CR = "Average Collision Rate"
    AVE_COMR = "Completion Rate"
    MAX_L = "Max Live Step"
    MIN_L = "Min Live Step"
    MEAN_L = "Mean Live Step"
    MIN_G = "Min Goal Distance"


class BasicMetricHandler(MetricHandler):
    """ MetricHandler serves for the metric """

    def __init__(self):
        """Create a MetricHandler instance to record the"""
        super(BasicMetricHandler, self).__init__()
        self._logs_mapping = dict()
        self._logs = None

    def set_log(self, algorithm, num_episodes):
        self._logs = [episode_log.BasicEpisodeLog() for _ in range(num_episodes)]
        self._logs_mapping[algorithm] = self._logs

    @property
    def logs_mapping(self):
        return self._logs_mapping

    def log_step(self, episode, observations, actions, rewards, dones, infos):
        self._logs[episode].record_step(observations, actions, rewards, dones, infos)

    def show_plots(self, **kwargs):
        """ Show behavior metric plots, support only one algorithm now. """

        behavior_metric = BehaviorMetric()
        results, metric_keys = behavior_metric.compute(self)
        value_dict = defaultdict(lambda: [])
        for algorithm, result in results.items():
            for k, v in result.items():
                value_dict[k].append(v)
        values = []
        for k in metric_keys:
            # normalization
            tmp = np.asarray(value_dict[k])
            tmp = (tmp - tmp.min()) / np.maximum(tmp.max() - tmp.min(), 1.0)
            values.append(tmp)
        values = np.asarray(values).T
        print(
            f">>>>>>>>>>>>>>>> shapes of: {values.shape}, {list(results.keys())}, {len(metric_keys)}"
        )
        plot.radar_plots(
            values, list(results.keys()), metric_keys, title="Behavior Analysis"
        )

    def write_to_csv(self, csv_dir):
        csv_dir = f"{csv_dir}/{int(time.time())}"
        for i, logger in enumerate(self._logs):
            sub_dir = f"{csv_dir}/episode_{i}"
            os.makedirs(sub_dir)
            for agent_id in logger.agent_step.keys():
                # get time step
                f_name = f"{sub_dir}/agent_{agent_id}.csv"
                with open(f_name, "w") as f:
                    writer = csv.writer(f, delimiter=",")
                    headers = [""] + [
                        str(i) for i in range(logger.agent_step[agent_id])
                    ]
                    writer.writerow(headers)
                    writer.writerow(["Speed"] + logger.ego_speed[agent_id])
                    writer.writerow(["GDistance"] + logger.distance_to_goal[agent_id])
                    # writer.writerow(
                    #     ["EDistance"] + logger.distance_to_ego_car[agent_id]
                    # )
                    # writer.writerow(["Acceleration"] + logger.acceleration[agent_id])
                    writer.writerow(
                        ["Num_Collision"] + [logger.num_collision[agent_id]]
                    )

    def read_logs(self, csv_dir):
        agent_record = defaultdict(
            lambda: {
                "Speed": None,
                "GDistance": None,
                "EDistance": None,
                "Num_Collision": None,
                "Acceleration": None,
            }
        )
        for f_name in os.listdir(csv_dir):
            if f_name.endswith(".csv"):
                f_path = os.path.join(csv_dir, f_name)
                agent_id = f_path.split(".")[0]
                print(f"Got file `{f_name}` for agent-{agent_id}")
                with open(
                    f_path,
                ) as f:
                    reader = csv.reader(f, delimiter=",")
                    _ = next(reader)
                    agent_record[agent_id]["Speed"] = next(reader)[1:]
                    agent_record[agent_id]["GDistance"] = next(reader)[1:]
                    # agent_record[agent_id]["EDistance"] = next(reader)[1:]
                    # agent_record[agent_id]["Acceleration"] = next(reader)[1:]
                    agent_record[agent_id]["Num_Collision"] = next(reader)
        return agent_record

    def compute(self, csv_dir):
        # list directory
        sub_dirs = [os.path.join(csv_dir, sub_dir) for sub_dir in os.listdir(csv_dir)]
        agent_metrics = defaultdict(
            lambda: {
                MetricKeys.AVE_CR: 0.0,
                MetricKeys.AVE_COMR: 0.0,
                MetricKeys.MAX_L: 0,
                MetricKeys.MIN_L: 0,
                MetricKeys.MEAN_L: 0.0,
                MetricKeys.MIN_G: 0.0,
            }
        )

        goal_dist_th = 2.0

        for sub_dir in sub_dirs:
            episode_agent_record: dict = self.read_episode(sub_dir)
            for aid, record in episode_agent_record.items():
                am = agent_metrics[aid]
                am[MetricKeys.AVE_CR] += record["Num_Collision"]
                min_goal_dist = record["GDistance"][-1]
                am[MetricKeys.AVE_COMR] += 1.0 if min_goal_dist < goal_dist_th else 0.0
                am[MetricKeys.MAX_L] = max(am[MetricKeys.MAX_L], len(record["Speed"]))
                am[MetricKeys.MIN_L] = min(am[MetricKeys.MIN_L], len(record["Speed"]))
                am[MetricKeys.MEAN_L] += len(record["Speed"])
                am[MetricKeys.MIN_G] = min(am[MetricKeys.MIN_G], min_goal_dist)

        for aid, record in agent_metrics.items():
            record[MetricKeys.MEAN_L] /= len(sub_dirs)
            record[MetricKeys.AVE_COMR] /= len(sub_dirs)
            record[MetricKeys.AVE_CR] /= len(sub_dirs)

        print(format.pretty_dict(agent_metrics))
