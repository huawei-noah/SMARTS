# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import os
from collections import defaultdict
import cv2, time, math, datetime, ray, shutil, dill
import numpy as np
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from smarts.core.utils.episodes import EpisodeLog
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager
import tableprint as tp


class LogInfo:
    def __init__(self):
        self.data = {
            "env_score": 0,
            "episode_reward": 0,
            "dist_center": 0,
            "goal_dist": 0,
            "speed": 0,
            "max_speed_violation": 0,
            "ego_num_violations": 0,
            "social_num_violations": 0,
            "ego_linear_jerk": 0.0,
            "ego_angular_jerk": 0.0,
            "final_pos": [0, 0],
            "start_pos": [0, 0],
            "dist_travelled": 0.0,
            "collision": 0,
            "off_road": 0,
            "off_route": 0,
            "reached_goal": 0,
            "timed_out": 0,
            "episode_length": 0,
        }

    def add(self, infos, rewards):

        self.data["env_score"] += int(infos["logs"]["env_score"])
        self.data["speed"] += infos["logs"]["speed"]
        self.data["max_speed_violation"] += (
            1 if infos["logs"]["speed"] > infos["logs"]["closest_wp"].speed_limit else 0
        )
        self.data["dist_center"] += infos["logs"]["dist_center"]
        self.data["ego_num_violations"] += int(infos["logs"]["ego_num_violations"] > 0)
        self.data["social_num_violations"] += int(
            infos["logs"]["social_num_violations"] > 0
        )
        self.data["goal_dist"] = infos["logs"]["goal_dist"]
        self.data["ego_linear_jerk"] += infos["logs"]["linear_jerk"]
        self.data["ego_angular_jerk"] += infos["logs"]["angular_jerk"]
        self.data["episode_reward"] += rewards
        self.data["final_pos"] = infos["logs"]["position"]
        self.data["start_pos"] = infos["logs"]["start"].position
        self.data["dist_travelled"] = math.sqrt(
            (self.data["final_pos"][1] - self.data["start_pos"][1]) ** 2
            + (self.data["final_pos"][0] - self.data["start_pos"][0]) ** 2
        )
        # recording termination cases
        events = infos["logs"]["events"]
        self.data["collision"] = (
            False
            if len(events.collisions) == 0 or events.collisions[0].collidee_id == 0
            else True
        )
        self.data["off_road"] = int(events.off_road)
        self.data["off_route"] = int(events.off_route)
        self.data["reached_goal"] = int(events.reached_goal)
        self.data["timed_out"] = int(events.reached_max_episode_steps)
        #

    def normalize(self, steps):
        self.data["env_score"] /= steps
        self.data["dist_center"] /= steps
        self.data["episode_length"] = steps
        self.data["speed"] /= steps
        self.data["ego_linear_jerk"] /= steps
        self.data["ego_angular_jerk"] /= steps
        self.data["ego_num_violations"] /= steps
        self.data["social_num_violations"] /= steps
        self.data["max_speed_violation"] /= steps


class Episode:
    def __init__(
        self,
        index,
        agents_itr=defaultdict(lambda: 0),
        eval_count=0,
        all_data=defaultdict(lambda: defaultdict(lambda: [])),
        experiment_name=None,
        etag=None,
        tb_writer=None,
        last_eval_iteration=None,
        log_dir=None,
    ):
        self.info = {}
        self.all_data = all_data
        self.index = index
        self.eval_count = eval_count
        dt = datetime.datetime.today()
        if experiment_name is None:
            self.experiment_name = f"experiment-{dt.year}.{dt.month}.{dt.day}-{dt.hour}:{dt.minute}:{dt.second}"
            if etag:
                self.experiment_name = f"{self.experiment_name}-{etag}"
        else:
            self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.experiment_dir = f"{self.log_dir}/{self.experiment_name}"
        self.model_dir = f"{self.log_dir}/{self.experiment_name}/models"
        self.code_dir = f"{self.log_dir}/{self.experiment_name}/codes"
        self.pkls = f"{self.log_dir}/{self.experiment_name}/pkls"
        self.start_time = time.time()
        self.timestep_sec = 0.1
        self.steps = 1
        self.active_tag = None
        self.tb_writer = tb_writer
        self.last_eval_iteration = last_eval_iteration
        self.agents_itr = agents_itr

    @property
    def sim2wall_ratio(self):
        return self.sim_time / self.wall_time

    @property
    def wall_time(self):
        return time.time() - self.start_time

    @property
    def sim_time(self):
        return self.timestep_sec * self.steps

    @property
    def steps_per_second(self):
        return self.steps / self.wall_time

    def get_itr(self, agent_id):
        return self.agents_itr[agent_id]

    def checkpoint_dir(self, iteration):
        path = f"{self.model_dir}/{iteration}"
        self.make_dir(path)
        return path

    def train_mode(self):
        self.active_tag = "Train"

    def eval_mode(self):
        self.active_tag = "Evaluation"

    def reset(self, mode="Train"):
        self.start_time = time.time()
        self.timestep_sec = 0.1
        self.steps = 1
        self.active_tag = mode
        self.info[self.active_tag] = LogInfo()

    def make_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def log_loss(self, step, loss_output):
        self.initialize_tb_writer()
        for key, data in loss_output.items():
            if step % data["freq"]:
                if data["type"] == "scalar":
                    self.tb_writer.add_scalar(key, data["data"], step)
                else:
                    self.tb_writer.add_histogram(key, data["data"], step)

    def save_episode(self, episode_count):
        self.ep_log_dir = "{}/episode_{}".format(self.log_dir, episode_count)
        if not os.path.exists(self.ep_log_dir):
            os.makedirs(self.ep_log_dir)

    def record_step(self, agent_id, infos, rewards, total_step=0, loss_output=None):
        if loss_output:
            self.log_loss(step=total_step, loss_output=loss_output)
        self.info[self.active_tag].add(infos[agent_id], rewards[agent_id])
        self.steps += 1
        self.agents_itr[agent_id] += 1

    def record_episode(self):
        # normalize some of the data; keep the rest as is
        self.info[self.active_tag].normalize(self.steps)

    def initialize_tb_writer(self):
        if self.tb_writer is None:
            self.tb_writer = SummaryWriter(
                "{}/{}".format(self.log_dir, self.experiment_name)
            )
            self.make_dir(self.log_dir)
            self.make_dir(self.model_dir)

    def record_tensorboard(self, agent_id, save_codes=None):
        # only create tensorboard once from training process
        self.initialize_tb_writer()
        agent_itr = self.get_itr(agent_id)
        data = {}
        for key, value in self.info[self.active_tag].data.items():
            if not isinstance(value, (list, tuple, np.ndarray)):
                self.tb_writer.add_scalar(
                    "{}/{}".format(self.active_tag, key), value, agent_itr
                )
                data[key] = value
        pkls_dir = f"{self.pkls}/{self.active_tag}"
        if not os.path.exists(pkls_dir):
            os.makedirs(pkls_dir)
        self.all_data[self.active_tag][agent_itr] = data
        with open(f"{pkls_dir}/results.pkl", "wb") as handle:
            dill.dump(self.all_data[self.active_tag], handle)

        if save_codes and not os.path.exists(self.code_dir):  # save once
            self.make_dir(self.code_dir)
            for code_path in save_codes:
                try:
                    if os.path.isdir(code_path):
                        shutil.copytree(code_path, self.code_dir)
                    elif os.path.isfile(code_path):
                        shutil.copy(code_path, self.code_dir)
                except FileExistsError:
                    pass


def episodes(n, etag=None, dir=None):
    log_dir = dir
    col_width = 18
    with tp.TableContext(
        [f"Episode", f"Sim/Wall", f"Total Steps", f"Steps/Sec", f"Score",],
        width=col_width,
        style="round",
    ) as table:
        tb_writer = None
        experiment_name = None
        last_eval_iteration = None
        eval_count = 0
        all_data = defaultdict(lambda: defaultdict(lambda: []))
        agents_itr = defaultdict(lambda: 0)
        for i in range(n):
            e = Episode(
                index=i,
                experiment_name=experiment_name,
                tb_writer=tb_writer,
                etag=etag,
                agents_itr=agents_itr,
                last_eval_iteration=last_eval_iteration,
                all_data=all_data,
                eval_count=eval_count,
                log_dir=log_dir,
            )
            yield e
            tb_writer = e.tb_writer
            last_eval_iteration = e.last_eval_iteration
            experiment_name = e.experiment_name
            all_data = e.all_data
            eval_count = e.eval_count
            agents_itr = e.agents_itr
            if e.active_tag:
                row = (
                    f"{e.index}/{n}",
                    f"{e.sim2wall_ratio:.2f}",
                    f"{e.steps}",
                    f"{e.steps_per_second:.2f}",
                    f"{e.info[e.active_tag].data['episode_reward']:.4f}",
                )
                table(row)

            else:
                table(("", "", "", "", ""))
