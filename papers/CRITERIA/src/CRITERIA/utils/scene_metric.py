# MIT License
#
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from __future__ import annotations

import math
import os
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
from argoverse.data_loading.argoverse_forecasting_loader import (
    ArgoverseForecastingLoader,
)
from argoverse.evaluation.eval_forecasting import (
    get_ade,
    get_displacement_errors_and_miss_rate,
    get_drivable_area_compliance,
    get_fde,
)
from argoverse.map_representation.map_api import ArgoverseMap

from ..utils.common import angle_between, get_translation
from ..utils.gen_vis_map import gen_vis_map


class scene_metrics_calculator:
    """
    This class calculte different metrics for prediction result.
    """

    def __init__(self, result, config: dict):
        """
        Please make sure that the result are dictionary with keys preds, gts and cities.
        """
        self.result = result

        # load result
        self.preds = self.result["preds"]
        self.gts = self.result["gts"]
        self.cities = self.result["cities"]

        # parameter for result calculation
        self.num_guess = self.preds[list(self.preds.keys())[0]].shape[0]
        self.horizon = self.preds[list(self.preds.keys())[0]].shape[1]

        self.argoverse_data_dir = config.get("argoverse_data_dir")
        metrics_section = config["metrics"]
        self.reference_frame = metrics_section.get("reference_frame")
        self.trajectory_count = metrics_section.get("trajectory_count")
        self.trajectories_per_second = metrics_section.get("trajectories_per_second")

        self.dao_scale = metrics_section["dao"].get("scale")
        self.vis_map_path = (
            metrics_section["dao"].get("vis_map_path") or config["vis_map_path"]
        )
        self.dao_max_distance = metrics_section["dao"].get("max_distance")
        self.hor_scale = metrics_section["hor"].get("scale")
        self.tad_scale = metrics_section["tad"].get("scale")
        self.sor_scale = metrics_section["sor"].get("scale")
        self.miss_threshold = metrics_section["miss_rate"].get("miss_threshold")
        aggression_danger_section = metrics_section["aggression_danger"]
        self.ad_public = aggression_danger_section.get("ad_public")
        self.ad_normal = aggression_danger_section.get("ad_normal")
        self.ad_aggressive = aggression_danger_section.get("ad_aggressive")
        self.ad_extremely_aggressive = aggression_danger_section.get(
            "ad_extremely_aggressive"
        )

        self.am = ArgoverseMap()
        self.afl = ArgoverseForecastingLoader(self.argoverse_data_dir)

        # basic metric calculation
        self.metric_result = get_displacement_errors_and_miss_rate(
            self.preds, self.gts, self.num_guess, self.horizon, self.miss_threshold
        )

    def get_minADE(self):  # minADE
        return self.metric_result["minADE"]

    def get_avgADE(self):  # average ADE
        ade_list = []
        for k, v in self.preds.items():
            total_ade = 0
            candidate = 0
            for traj in v:
                total_ade = total_ade + get_ade(traj, self.gts[k])
                candidate = candidate + 1
            ade_list.append(total_ade / candidate)

        return sum(ade_list) / len(ade_list)

    def get_minFDE(self):  # minFDE
        return self.metric_result["minFDE"]

    def get_MR(self):  # Miss Rate
        return self.metric_result["MR"]

    def get_RF_avgFDE(self):  # RF and average FDE
        min_fde = self.metric_result["minFDE"]
        fde_list = []

        for k, v in self.preds.items():
            for traj in v:
                fde = get_fde(traj, self.gts[k])
                fde_list.append(fde)

        avg_fde = sum(fde_list) / len(fde_list)
        rf = avg_fde / min_fde
        return rf, avg_fde

    def get_DAC(self):  # Driveable Area Compliance
        total_traj = 0
        num_dac_traj = 0
        for k, v in self.preds.items():
            trajs = deepcopy(v)
            for traj in trajs:
                total_traj += 1
                raster_layer = self.am.get_raster_layer_points_boolean(
                    traj, self.cities[k], "driveable_area"
                )
                if np.sum(raster_layer) == raster_layer.shape[0]:
                    num_dac_traj += 1

        return num_dac_traj / total_traj

    def get_SOR(self):  # Soft offroad rate
        num_scenes = len(self.gts)
        sor = []
        for k, v in self.preds.items():
            city = self.cities[k]
            off_roads_SOR = 0
            num_points = 0
            for traj in v:
                off_roads = self.am.get_raster_layer_points_boolean(
                    traj, city, "driveable_area"
                )
                off_roads_SOR += len(off_roads) - off_roads.sum()
                num_points += len(off_roads)

            sor.append(off_roads_SOR / num_points)

        return sum(sor) / num_scenes * self.sor_scale

    def get_DAO(self):  # Driveable Area Occupancy
        total_dao = 0
        for k, v in self.preds.items():
            city = self.cities[k]
            trans_trajs = deepcopy(v)
            df = pd.read_csv(os.path.join(self.argoverse_data_dir, f"{k}.csv"))
            translation = get_translation(df, self.reference_frame)
            trans_trajs[:, :, 0] -= translation[0]
            trans_trajs[:, :, 1] -= translation[1]

            vis_map = gen_vis_map(city, translation, self.vis_map_path)
            map_w, map_h = vis_map.shape
            trans_trajs = np.expand_dims(trans_trajs, axis=0)
            trans_trajs = trans_trajs + self.dao_max_distance
            trans_trajs[:, :, :, 0] *= map_w / (2 * self.dao_max_distance)
            trans_trajs[:, :, :, 1] *= map_h / (2 * self.dao_max_distance)
            trans_trajs = trans_trajs.astype(np.int64)

            dao_data, dao_mask = dao(trans_trajs, vis_map)
            total_dao += dao_data.sum()

        final_dao = total_dao * self.dao_scale
        return final_dao

    def get_TAD(self):  # trajectories angle discrepency
        avg_TAD_list = []
        for key, val in self.preds.items():
            trajs = deepcopy(val)
            avg_angle_list = []
            # loop for current traj
            for i in range(len(trajs)):
                base_traj = trajs[i]
                total_angles = 0
                total_portion = 0
                # loop over other trajs
                for j in range(len(trajs)):
                    if i != j:
                        cur_traj = trajs[j]
                        # loop for each time step
                        for k in range(len(trajs[i]) - 1):
                            base_traj_vec = base_traj[k + 1] - base_traj[0]
                            cur_traj_vec = cur_traj[k + 1] - cur_traj[0]
                            if (
                                np.linalg.norm(base_traj_vec) == 0
                                or np.linalg.norm(cur_traj_vec) == 0
                            ):
                                continue
                            angle = angle_between(base_traj_vec, cur_traj_vec)
                            total_angles = total_angles + angle
                            total_portion = total_portion + 1
                avg_angle_list.append(total_angles / total_portion)

            avg_angle_list.remove(max(avg_angle_list))
            avg_angle_list.remove(min(avg_angle_list))
            avg_TAD_list.append(sum(avg_angle_list) / len(avg_angle_list))

        return sum(avg_TAD_list) / len(avg_TAD_list) * self.tad_scale

    def get_seTAD(self):  # Trajectories angle discrepency (start, end)
        avg_TAD_list = []
        for key, val in self.preds.items():
            trajs = deepcopy(val)
            avg_angle_list = []
            # loop for current traj
            for i in range(len(trajs)):
                base_traj = trajs[i]
                total_angles = 0
                total_portion = 0
                # loop over other trajs
                for j in range(len(trajs)):
                    if i != j:
                        cur_traj = trajs[j]
                        start_base_traj_vec = base_traj[1] - base_traj[0]
                        start_cur_traj_vec = cur_traj[1] - cur_traj[0]
                        if (
                            np.linalg.norm(start_base_traj_vec) == 0
                            or np.linalg.norm(start_cur_traj_vec) == 0
                        ):
                            continue
                        total_angles += angle_between(
                            start_base_traj_vec, start_cur_traj_vec
                        )
                        total_portion += 1

                        end_base_traj_vec = base_traj[-1] - base_traj[-2]
                        end_cur_traj_vec = cur_traj[-1] - cur_traj[-2]
                        if (
                            np.linalg.norm(end_base_traj_vec) == 0
                            or np.linalg.norm(end_cur_traj_vec) == 0
                        ):
                            continue
                        total_angles += angle_between(
                            end_base_traj_vec, end_cur_traj_vec
                        )
                        total_portion += 1

                if total_portion == 0:
                    continue
                avg_angle_list.append(total_angles / total_portion)

            avg_angle_list.remove(max(avg_angle_list))
            avg_angle_list.remove(min(avg_angle_list))
            avg_TAD_list.append(sum(avg_angle_list) / len(avg_angle_list))

        return sum(avg_TAD_list) / len(avg_TAD_list) * self.tad_scale

    def get_TDD(self):  # Trajectory displacement discrepancy
        avg_TDD_list = []
        for scene_id, preds in self.preds.items():
            # get traj travel distance
            traj_dis_list = []
            for traj in preds:
                total_dis = sum(
                    [
                        np.linalg.norm(traj[i] - traj[i + 1])
                        for i in range(len(traj) - 1)
                    ]
                )
                traj_dis_list.append(total_dis)

            # discrepancy calculation
            average_discrepancy_list = []
            for i in range(len(traj_dis_list)):
                base_dis = traj_dis_list[i]
                discrepancy_list = []
                for j in range(len(traj_dis_list)):
                    if i != j:
                        current_dis = traj_dis_list[j]
                        discrepancy_list.append(abs(base_dis - current_dis))
                average_discrepancy_list.append(np.average(discrepancy_list))

            average_discrepancy_list.remove(max(average_discrepancy_list))
            average_discrepancy_list.remove(min(average_discrepancy_list))
            avg_TDD_list.append(np.average(average_discrepancy_list))

        return np.average(avg_TDD_list)

    def get_TAR(self, dir_check=True):  # trajectory admissible ratio
        # return three result, using different range of acceleration
        total_traj = 0
        bad_traj_normal = 0
        bad_traj_agg = 0
        bad_traj_extagg = 0

        tps = self.trajectories_per_second
        for k, v in self.preds.items():
            seq_path = f"{self.argoverse_data_dir}/{k}.csv"
            agent_obs_traj = self.afl.get(seq_path).agent_traj[: self.trajectory_count]
            trajs = deepcopy(v)
            for traj in trajs:
                total_traj += 1

                # off road check
                raster_layer = self.am.get_raster_layer_points_boolean(
                    traj, self.cities[k], "driveable_area"
                )
                if np.sum(raster_layer) != raster_layer.shape[0]:
                    bad_traj_normal += 1
                    bad_traj_agg += 1
                    bad_traj_extagg += 1
                    continue

                # acceleration check
                begin_past_dis = sum(
                    np.linalg.norm(agent_obs_traj[i + 1] - agent_obs_traj[i])
                    for i in range(tps, tps * 2 - 1)
                )
                begin_current_dis = sum(
                    [np.linalg.norm(traj[i] - traj[i + 1]) for i in range(0, tps - 1)]
                )
                end_past_dis = sum(
                    [
                        np.linalg.norm(traj[i] - traj[i + 1])
                        for i in range(tps, tps * 2 - 1)
                    ]
                )
                end_current_dis = sum(
                    [
                        np.linalg.norm(traj[i] - traj[i + 1])
                        for i in range(tps * 2, tps * 3 - 1)
                    ]
                )

                acc = (
                    (end_current_dis - end_past_dis)
                    + (begin_current_dis - begin_past_dis)
                ) / 2

                # different range acceleration
                if acc < self.ad_normal[0] or acc > self.ad_normal[1]:
                    bad_traj_normal += 1
                    if acc < self.ad_aggressive[0] or acc > self.ad_aggressive[1]:
                        bad_traj_agg += 1
                        if (
                            acc < self.ad_extremely_aggressive[0]
                            or acc > self.ad_extremely_aggressive[1]
                        ):
                            bad_traj_extagg += 1
                    continue

                # lane direction check
                # get traj final heading direction
                if dir_check == True:
                    ts_list = traj[-4:]
                    tj_vector_list = [
                        (ts_list[i + 1] - traj[i]) for i in range(len(ts_list) - 1)
                    ]
                    # retrieval of lanes that traj final 3 timesteps are located on (multiple lanes possible)
                    lane_list = [
                        self.am.get_lane_ids_in_xy_bbox(
                            ts_list[i][0], ts_list[i][1], self.cities[k], 0.1
                        )
                        for i in range(1, 4)
                    ]
                    lane_vector_list = []
                    for obj in lane_list:
                        if obj:  # if lanes exist, get lane direction
                            point_list = [
                                self.am.get_lane_segment_centerline(
                                    obj[i], self.cities[k]
                                )[0][:2]
                                for i in range(len(obj))
                            ]
                            lane_dir = [
                                self.am.get_lane_direction(
                                    point_list[i], self.cities[k]
                                )[0]
                                for i in range(len(point_list))
                            ]
                            lane_vector_list.append(lane_dir)
                        else:  # if final 3 timesteps are not near any lane
                            lane_vector_list.append([])

                    # compare lanes orientation with traj orientation using formula 1 -orien/pi
                    confidence_list = []
                    for i in range(len(tj_vector_list)):
                        traj_orien = tj_vector_list[i]
                        lane_orien_list = lane_vector_list[i]
                        if lane_orien_list:
                            confidence = max(
                                [
                                    max(
                                        0,
                                        (
                                            1
                                            - angle_between(traj_orien, lane_orein)
                                            / math.pi
                                        ),
                                    )
                                    for lane_orein in lane_orien_list
                                ]
                            )
                        else:  # no lanes alignment find
                            confidence = 0
                        confidence_list.append(confidence)

                    # if traj is not aligned with any lanes nearby
                    if all(conf <= 0.5 for conf in confidence_list):
                        bad_traj_normal += 1
                        bad_traj_agg += 1
                        bad_traj_extagg += 1

        # bad_traj_normal: include traj acc outside normal range
        # bad_traj_agg: include traj acc outside aggresive range
        # bad_traj danger: include traj acc outside extremely aggresive range

        return (
            bad_traj_normal / total_traj,
            bad_traj_agg / total_traj,
            bad_traj_extagg / total_traj,
        )

    def get_TAR_ablation(
        self, dir_check=True
    ):  # ablation of different component of TAR
        total_traj = 0

        bad_traj_offroad = 0
        bad_traj_acc = 0
        bad_traj_wd = 0

        normal = self.ad_normal

        for k, v in self.preds.items():
            seq_path = f"{self.argoverse_data_dir}/{k}.csv"
            agent_obs_traj = self.afl.get(seq_path).agent_traj[: self.trajectory_count]
            trajs = deepcopy(v)
            for traj in trajs:
                total_traj += 1

                # off road check
                raster_layer = self.am.get_raster_layer_points_boolean(
                    traj, self.cities[k], "driveable_area"
                )
                if np.sum(raster_layer) != raster_layer.shape[0]:
                    bad_traj_offroad += 1

                # acceleration check
                begin_past_dis = sum(
                    np.linalg.norm(agent_obs_traj[i + 1] - agent_obs_traj[i])
                    for i in range(
                        self.trajectories_per_second, self.trajectories_per_second - 1
                    )
                )
                begin_current_dis = sum(
                    [
                        np.linalg.norm(traj[i] - traj[i + 1])
                        for i in range(0, self.trajectories_per_second - 1)
                    ]
                )
                end_past_dis = sum(
                    [
                        np.linalg.norm(traj[i] - traj[i + 1])
                        for i in range(
                            self.trajectories_per_second,
                            self.trajectories_per_second * 2 - 1,
                        )
                    ]
                )
                end_current_dis = sum(
                    [
                        np.linalg.norm(traj[i] - traj[i + 1])
                        for i in range(
                            self.trajectories_per_second * 2,
                            self.trajectories_per_second - 1,
                        )
                    ]
                )

                acc = (
                    (end_current_dis - end_past_dis)
                    + (begin_current_dis - begin_past_dis)
                ) / 2

                if acc < normal[0] or acc > normal[1]:
                    bad_traj_acc += 1

                # lane direction check
                if dir_check == True:
                    ts_list = traj[-4:]
                    tj_vector_list = [
                        (ts_list[i + 1] - traj[i]) for i in range(len(ts_list) - 1)
                    ]
                    lane_list = [
                        self.am.get_lane_ids_in_xy_bbox(
                            ts_list[i][0], ts_list[i][1], self.cities[k], 0.1
                        )
                        for i in range(1, 4)
                    ]
                    lane_vector_list = []
                    for obj in lane_list:
                        if obj:
                            point_list = [
                                self.am.get_lane_segment_centerline(
                                    obj[i], self.cities[k]
                                )[0][:2]
                                for i in range(len(obj))
                            ]
                            lane_dir = [
                                self.am.get_lane_direction(
                                    point_list[i], self.cities[k]
                                )[0]
                                for i in range(len(point_list))
                            ]
                            lane_vector_list.append(lane_dir)
                        else:
                            lane_vector_list.append([])
                    # compare
                    confidence_list = []
                    for i in range(len(tj_vector_list)):
                        traj_orien = tj_vector_list[i]
                        lane_orien_list = lane_vector_list[i]
                        if lane_orien_list:
                            confidence = max(
                                [
                                    max(
                                        0,
                                        (
                                            1
                                            - angle_between(traj_orien, lane_orein)
                                            / math.pi
                                        ),
                                    )
                                    for lane_orein in lane_orien_list
                                ]
                            )
                        else:
                            confidence = 0
                        confidence_list.append(confidence)

                    if all(conf <= 0.5 for conf in confidence_list):
                        bad_traj_wd += 1

        return (
            bad_traj_offroad / total_traj,
            bad_traj_acc / total_traj,
            bad_traj_wd / total_traj,
        )

    def return_inad_trajs(self, dir_check=True):  # trajectory admissible ratio
        good_traj = []

        bad_traj_offroad = []
        bad_traj_acc = []
        bad_traj_wd = []

        normal = self.ad_normal

        for k, v in self.preds.items():
            seq_path = f"{self.argoverse_data_dir}/{k}.csv"
            agent_obs_traj = self.afl.get(seq_path).agent_traj[: self.trajectory_count]
            trajs = deepcopy(v)
            for traj in trajs:
                # off road check
                raster_layer = self.am.get_raster_layer_points_boolean(
                    traj, self.cities[k], "driveable_area"
                )

                if np.sum(raster_layer) != raster_layer.shape[0]:
                    bad_traj_offroad.append(traj)
                    continue

                # acceleration check
                begin_past_dis = sum(
                    np.linalg.norm(agent_obs_traj[i + 1] - agent_obs_traj[i])
                    for i in range(
                        self.trajectories_per_second,
                        self.trajectories_per_second * 2 - 1,
                    )
                )
                begin_current_dis = sum(
                    [
                        np.linalg.norm(traj[i] - traj[i + 1])
                        for i in range(0, self.trajectories_per_second - 1)
                    ]
                )
                end_past_dis = sum(
                    [
                        np.linalg.norm(traj[i] - traj[i + 1])
                        for i in range(
                            self.trajectories_per_second,
                            self.trajectories_per_second * 2 - 1,
                        )
                    ]
                )
                end_current_dis = sum(
                    [
                        np.linalg.norm(traj[i] - traj[i + 1])
                        for i in range(
                            self.trajectories_per_second * 2,
                            self.trajectories_per_second * 3 - 1,
                        )
                    ]
                )

                acc = (
                    (end_current_dis - end_past_dis)
                    + (begin_current_dis - begin_past_dis)
                ) / 2

                if acc < normal[0] or acc > normal[1]:
                    bad_traj_acc.append(traj)
                    continue

                # lane direction check
                if dir_check == True:
                    ts_list = traj[-4:]
                    tj_vector_list = [
                        (ts_list[i + 1] - traj[i]) for i in range(len(ts_list) - 1)
                    ]
                    lane_list = [
                        self.am.get_lane_ids_in_xy_bbox(
                            ts_list[i][0], ts_list[i][1], self.cities[k], 0.1
                        )
                        for i in range(1, 4)
                    ]
                    lane_vector_list = []
                    for obj in lane_list:
                        if obj:
                            point_list = [
                                self.am.get_lane_segment_centerline(
                                    obj[i], self.cities[k]
                                )[0][:2]
                                for i in range(len(obj))
                            ]
                            lane_dir = [
                                self.am.get_lane_direction(
                                    point_list[i], self.cities[k]
                                )[0]
                                for i in range(len(point_list))
                            ]
                            lane_vector_list.append(lane_dir)
                        else:
                            lane_vector_list.append([])
                    # compare
                    confidence_list = []
                    for i in range(len(tj_vector_list)):
                        traj_orien = tj_vector_list[i]
                        lane_orien_list = lane_vector_list[i]
                        if lane_orien_list:
                            confidence = max(
                                [
                                    max(
                                        0,
                                        (
                                            1
                                            - angle_between(traj_orien, lane_orein)
                                            / math.pi
                                        ),
                                    )
                                    for lane_orein in lane_orien_list
                                ]
                            )
                        else:
                            confidence = 0
                        confidence_list.append(confidence)

                    if all(conf <= 0.5 for conf in confidence_list):
                        bad_traj_wd.append(traj)
                        continue

                good_traj.append(traj)

        return (good_traj, bad_traj_offroad, bad_traj_acc, bad_traj_wd)

    def get_minASD(self):  # minimum average self distance
        minASD_list = []
        for k, v in self.preds.items():
            trajs = deepcopy(v)
            trajs_list = []
            scene_avgASD_list = []
            for traj in trajs:
                trajs_list.append(traj)
            traj_combs = combinations(trajs_list, 2)
            for comb in traj_combs:
                traj_1 = comb[0]
                traj_2 = comb[1]
                total_dis = 0
                for i in range(0, len(traj_1)):
                    x1, y1 = traj_1[i][0], traj_1[i][1]
                    x2, y2 = traj_2[i][0], traj_2[i][1]
                    distance = (x2 - x1) ** 2 + (y2 - y1) ** 2
                    total_dis += distance
                scene_avgASD_list.append(total_dis / len(traj_1))
            minASD_list.append(min(scene_avgASD_list))

        return sum(minASD_list) / len(minASD_list)

    def get_minFSD(self):  # minimum final self distance
        minFSD_list = []
        for k, v in self.preds.items():
            trajs = deepcopy(v)
            trajs_list = []
            scene_FSD_list = []
            for traj in trajs:
                trajs_list.append(traj)
            traj_combs = combinations(trajs_list, 2)
            for comb in traj_combs:
                traj_1 = comb[0]
                traj_2 = comb[1]
                x1, y1 = traj_1[-1][0], traj_1[-1][1]
                x2, y2 = traj_2[-1][0], traj_2[-1][1]
                distance = (x2 - x1) ** 2 + (y2 - y1) ** 2
                scene_FSD_list.append(distance)

            minFSD_list.append(min(scene_FSD_list))

        return sum(minFSD_list) / len(minFSD_list)

    def return_paper_metrics(self):
        metric_dict = {}

        minADE = self.get_minADE()
        minFDE = self.get_minFDE()
        dac = self.get_DAC()
        rf, avgFDE = self.get_RF_avgFDE()
        dao = self.get_DAO()
        tad = self.get_seTAD()
        tdd = self.get_TDD()
        tar = self.get_TAR()
        minFSD = self.get_minFSD()
        minASD = self.get_minASD()

        metric_dict = {
            "minADE": minADE,  #
            "minFDE": minFDE,  #
            "DAC": dac,  #
            "RF": rf,  #
            "DAO": dao,  #
            "TAD": tad,  #
            "TDD": tdd,  #
            "TAR": tar,  #
            "minFSD": minFSD,  #
            "minASD": minASD,  #
        }
        return metric_dict

    def return_all_metrics(self):
        metric_dict = {}

        minADE = self.get_minADE()
        avgADE = self.get_avgADE()
        minFDE = self.get_minFDE()
        dac = self.get_DAC()
        mr = self.get_MR()
        rf, avgFDE = self.get_RF_avgFDE()
        sor = self.get_SOR()
        dao = self.get_DAO()
        tad = self.get_seTAD()
        setad = self.get_seTAD()
        tdd = self.get_TDD()
        tar = self.get_TAR()
        minFSD = self.get_minFSD()
        minASD = self.get_minASD()

        metric_dict = {
            "minADE": minADE,  #
            "minFDE": minFDE,  #
            "DAC": dac,  #
            "MR": mr,
            "RF": rf,  #
            "SOR": sor,
            "DAO": dao,  #
            "avgADE": avgADE,
            "avgFDE": avgFDE,
            "TAD": tad,  #
            "seTAD": setad,
            "TDD": tdd,  #
            "TAR": tar,  #
            "minFSD": minFSD,  #
            "minASD": minASD,  #
        }

        return metric_dict

    def return_diversity_metrics(self):
        metric_dict = {}

        rf = self.get_RF_avgFDE()[0]
        minFSD = self.get_minFSD()
        minASD = self.get_minASD()
        tad = self.get_seTAD()
        tdd = self.get_TDD()

        metric_dict = {
            "RF": rf,
            "minFSD": minFSD,
            "minASD": minASD,
            "TAD": tad,
            "TDD": tdd,
        }

        return metric_dict

    def return_admissibility_metrics(self):
        metric_dict = {}
        dao = self.get_DAO()
        dac = self.get_DAC()
        tar = self.get_TAR()

        metric_dict = {"DAO": dao, "DAC": dac, "TAR": tar}

        return metric_dict


def dao(gen_trajs, map_array):
    map_h, map_w = map_array.shape
    da_mask = map_array > 0

    num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]

    oom_mask = np.any(
        np.logical_or(gen_trajs >= [[[map_w, map_h]]], gen_trajs < [[[0, 0]]]), axis=-1
    )
    agent_oom = oom_mask.sum(axis=(1, 2)) > 0

    dao = np.array([0.0 for i in range(num_agents)])
    for j in range(num_agents):
        if agent_oom[j]:
            continue

        gen_trajs_j = gen_trajs[j]
        gen_trajs_j_flat = gen_trajs_j.reshape(num_candidates * decoding_timesteps, 2)

        ravel = np.ravel_multi_index(gen_trajs_j_flat.T, dims=(map_w, map_h))
        ravel_unqiue = np.unique(ravel)

        x, y = np.unravel_index(ravel_unqiue, shape=(map_w, map_h))
        in_da = da_mask[y, x]
        dao[j] = in_da.sum() / da_mask.sum()

    dao_mask = np.logical_not(agent_oom)

    return dao, dao_mask
