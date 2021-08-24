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
import datetime
import math
import os
from pathlib import Path

import shutil
import time
from collections import defaultdict

import dill
import numpy as np
import tableprint as tp
from ultra.utils.rllib_log_info import RLlibLogInfo
from ultra.utils.common import gen_experiment_name

import tempfile
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import Logger, UnifiedLogger

from tensorboardX import SummaryWriter


class LogInfo:
    def __init__(self):
        self.data = {
            "env_score": 0.0,
            "episode_return": 0.0,
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
            "episode_length": 1,
            "scenario_name": None,
        }

    def add(self, infos, rewards):
        self.data["env_score"] += infos["logs"]["env_score"]
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
        self.data["episode_return"] += rewards
        self.data["final_pos"] = infos["logs"]["position"][:2]
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

    def step(self):
        self.data["episode_length"] += 1

    def record_scenario_name(self, scenario_name):
        self.data["scenario_name"] = scenario_name

    def normalize(self):
        steps = self.data["episode_length"]
        self.data["dist_center"] /= steps
        self.data["speed"] /= steps
        self.data["ego_linear_jerk"] /= steps
        self.data["ego_angular_jerk"] /= steps
        # self.data["ego_num_violations"] /= steps
        # self.data["social_num_violations"] /= steps
        # self.data["max_speed_violation"] /= steps


class Episode:
    def __init__(
        self,
        index,
        agents_itr=defaultdict(lambda: 0),
        eval_count=0,
        all_data=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))),
        experiment_name=None,
        etag=None,
        tb_writer=None,
        last_eval_iterations=defaultdict(lambda: None),
        log_dir=None,
    ):
        self.info = defaultdict(lambda: defaultdict(lambda: LogInfo()))
        self.all_data = all_data
        self.index = index
        self.eval_count = eval_count

        if experiment_name is None:
            self.experiment_name = gen_experiment_name()
            if etag:
                self.experiment_name = f"{self.experiment_name}-{etag}"
        else:
            self.experiment_name = experiment_name

        if log_dir is None:
            self.log_dir = "logs"
        else:
            self.log_dir = log_dir

        self.experiment_dir = f"{self.log_dir}/{self.experiment_name}"
        self.model_dir = f"{self.log_dir}/{self.experiment_name}/models"
        self.extra_data_dir = f"{self.log_dir}/{self.experiment_name}/extras"
        self.code_dir = f"{self.log_dir}/{self.experiment_name}/codes"
        self.pkls = f"{self.log_dir}/{self.experiment_name}/pkls"
        self.start_time = time.time()
        self.timestep_sec = 0.1
        self.steps = 1
        self.active_tag = None
        self.tb_writer = tb_writer
        self.last_eval_iterations = last_eval_iterations
        self.agents_itr = agents_itr
        self.current_scenario_name = None

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

    @property
    def scenario_name(self):
        return self.current_scenario_name

    def get_itr(self, agent_id):
        return self.agents_itr[agent_id]

    def checkpoint_dir(self, agent_id, iteration):
        path = f"{self.model_dir}/{agent_id}/{iteration}"
        self.make_dir(path)
        return path

    def extras_dir(self, agent_id):
        path = f"{self.extra_data_dir}/{agent_id}"
        self.make_dir(path)
        return path

    def train_mode(self):
        self.active_tag = "Train"

    def eval_train_mode(self):
        self.active_tag = "Evaluation_Training"

    def eval_mode(self):
        self.active_tag = "Evaluation"

    def gap_mode(self):
        self.active_tag = "Gap"

    def calculate_gap(self):
        gap_info = self.info["Gap"]
        for agent_id, agent_info in self.info["Evaluation"].items():
            for key in agent_info.data:
                if np.isscalar(gap_info[agent_id].data[key]):
                    gap_info[agent_id].data[key] = (
                        self.info["Evaluation_Training"][agent_id].data[key]
                        - self.info["Evaluation"][agent_id].data[key]
                    )

    def reset(self, mode="Train"):
        self.start_time = time.time()
        self.timestep_sec = 0.1
        self.steps = 1
        self.active_tag = mode
        self.info[self.active_tag] = defaultdict(lambda: LogInfo())

    def make_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def log_loss(self, step, agent_id, loss_output):
        self.initialize_tb_writer()
        for key, data in loss_output.items():
            if step % data["freq"]:
                loss_name, loss_type = key.split("/")
                if data["type"] == "scalar":
                    self.tb_writer.add_scalar(
                        "{}/{}/{}".format(loss_name, agent_id, loss_type),
                        data["data"],
                        step,
                    )
                else:
                    self.tb_writer.add_histogram(
                        "{}/{}/{}".format(loss_name, agent_id, loss_type),
                        data["data"],
                        step,
                    )

    def save_episode(self, episode_count):
        self.ep_log_dir = "{}/episode_{}".format(self.log_dir, episode_count)
        if not os.path.exists(self.ep_log_dir):
            os.makedirs(self.ep_log_dir)

    def record_step(
        self, agent_ids_to_record, infos, rewards, total_step=0, loss_outputs=None
    ):
        # Record the data for each agent ID.
        for agent_id in agent_ids_to_record:
            if loss_outputs:
                self.log_loss(
                    step=total_step,
                    agent_id=agent_id,
                    loss_output=loss_outputs[agent_id],
                )
            self.info[self.active_tag][agent_id].add(infos[agent_id], rewards[agent_id])
            self.info[self.active_tag][agent_id].step()
            self.agents_itr[agent_id] += 1

        # Increment this episode's step count.
        self.steps += 1

    def record_episode(self, scenario_name):
        self.current_scenario_name = scenario_name
        for _, agent_info in self.info[self.active_tag].items():
            agent_info.record_scenario_name(self.current_scenario_name)
            agent_info.normalize()

    def initialize_tb_writer(self):
        if self.tb_writer is None:
            self.tb_writer = SummaryWriter(
                "{}/{}".format(self.log_dir, self.experiment_name)
            )
            self.make_dir(self.log_dir)
            self.make_dir(self.model_dir)

    def record_tensorboard(self, save_codes=None, recording_step=None):
        # Due to performing evaluation asynchronously, this may record evaluation
        # results out of order. For example, if Evaluation 1 began at t = 100 s
        # and recorded at t = 150 s, but Evaluation 2 began at t = 120 s and
        # recorded at t = 140 s, the plot of the data will appear to go backwards,
        # and also only have a data point for Evaluation 2. This seems to be a
        # known tensorboardX and tensorboard issue that has not yet been resolved:
        # https://github.com/tensorflow/tensorboard/issues/3570,
        # https://github.com/lanpa/tensorboardX/milestone/1.
        # Combat this issue by increasing eval_rate so that overlapping
        # evaluations are less likely.

        # Only create tensorboard once from training process.
        self.initialize_tb_writer()

        for agent_id, agent_info in self.info[self.active_tag].items():
            agent_itr = (
                recording_step if recording_step is not None else self.get_itr(agent_id)
            )
            data = {}

            for key, value in agent_info.data.items():
                if value is not None and not isinstance(
                    value, (str, list, tuple, np.ndarray)
                ):
                    # Only record scalar data to TensorBoard.
                    self.tb_writer.add_scalar(
                        "{}/{}/{}".format(self.active_tag, agent_id, key),
                        value,
                        agent_itr,
                    )
                # Include all the data in the results.
                data[key] = value
            self.all_data[self.active_tag][agent_id][agent_itr] = data

        pkls_dir = f"{self.pkls}/{self.active_tag}"
        if not os.path.exists(pkls_dir):
            os.makedirs(pkls_dir)
        with open(f"{pkls_dir}/results.pkl", "wb") as handle:
            dill.dump(self.all_data[self.active_tag], handle)

        if save_codes and not os.path.exists(self.code_dir):  # Save once.
            self.make_dir(self.code_dir)
            for code_path in save_codes:
                try:
                    if os.path.isdir(code_path):
                        shutil.copytree(code_path, self.code_dir)
                    elif os.path.isfile(code_path):
                        shutil.copy(code_path, self.code_dir)
                except FileExistsError:
                    pass


def episodes(n, etag=None, log_dir=None):
    col_width = [15, 15, 15, 15, 18, 18, 60]
    with tp.TableContext(
        [
            f"Episode",
            f"Sim/Wall",
            f"Total Steps",
            f"Steps/Sec",
            f"Score",
            f"Goal Completed",
            f"Scenario Name",
        ],
        width=col_width,
        style="round",
    ) as table:
        tb_writer = None
        experiment_name = None
        last_eval_iterations = defaultdict(lambda: None)
        eval_count = 0
        all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
        agents_itr = defaultdict(lambda: 0)
        for i in range(n):
            e = Episode(
                index=i,
                experiment_name=experiment_name,
                tb_writer=tb_writer,
                etag=etag,
                agents_itr=agents_itr,
                last_eval_iterations=last_eval_iterations,
                all_data=all_data,
                eval_count=eval_count,
                log_dir=log_dir,
            )
            yield e
            tb_writer = e.tb_writer
            last_eval_iterations = e.last_eval_iterations
            experiment_name = e.experiment_name
            all_data = e.all_data
            eval_count = e.eval_count
            agents_itr = e.agents_itr
            if e.active_tag:
                agent_rewards_strings = [
                    "{}: {:.4f}".format(
                        agent_id,
                        agent_info.data["episode_return"],
                    )
                    for agent_id, agent_info in e.info[e.active_tag].items()
                ]
                agent_goal_completion_strings = [
                    "{}: {}".format(
                        agent_id,
                        bool(agent_info.data["reached_goal"]),
                    )
                    for agent_id, agent_info in e.info[e.active_tag].items()
                ]
                row = (
                    f"{e.index + 1}/{n}",
                    f"{e.sim2wall_ratio:.2f}",
                    f"{e.steps}",
                    f"{e.steps_per_second:.2f}",
                    ", ".join(agent_rewards_strings),
                    ", ".join(agent_goal_completion_strings),
                    f"{e.scenario_name}",
                )
                table(row)
            else:
                table(("", "", "", "", "", "", ""))


class Callbacks(DefaultCallbacks):
    @staticmethod
    def on_episode_start(
        worker,
        base_env,
        policies,
        episode,
        **kwargs,
    ):
        episode.user_data = RLlibLogInfo()

    @staticmethod
    def on_episode_step(
        worker,
        base_env,
        episode,
        **kwargs,
    ):

        single_agent_id = list(episode._agent_to_last_obs)[0]
        policy_id = episode.policy_for(single_agent_id)
        agent_reward_key = (single_agent_id, policy_id)

        info = episode.last_info_for(single_agent_id)
        reward = episode.agent_rewards[agent_reward_key]
        if info:
            episode.user_data.add(info, reward)

    @staticmethod
    def on_episode_end(
        worker,
        base_env,
        policies,
        episode,
        **kwargs,
    ):
        episode.user_data.normalize(episode.length)
        for key, val in episode.user_data.data.items():
            if not isinstance(val, (list, tuple, np.ndarray)):
                episode.custom_metrics[key] = val

        print(
            f"Episode {episode.episode_id} ended:\nlength:{episode.length},\nenv_score:{episode.custom_metrics['env_score']},\ncollision:{episode.custom_metrics['collision']}, \nreached_goal:{episode.custom_metrics['reached_goal']},\ntimeout:{episode.custom_metrics['timed_out']},\noff_road:{episode.custom_metrics['off_road']},\ndist_travelled:{episode.custom_metrics['dist_travelled']},\ngoal_dist:{episode.custom_metrics['goal_dist']}"
        )
        print("--------------------------------------------------------")


def log_creator(log_dir):
    result_dir = log_dir
    result_dir = Path(result_dir).expanduser().resolve().absolute()
    logdir_prefix = gen_experiment_name()

    def logger_creator(config):
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=result_dir)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator
