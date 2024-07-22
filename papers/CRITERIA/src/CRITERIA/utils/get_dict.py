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
# organized result dictionary
import os
import os.path
import pickle as pkl
from copy import deepcopy

import numpy as np
import torch


def get_dictionary(pkl_path, model_name, afl, argoverse_data_path):
    try:
        result_dict_ = torch.load(pkl_path)
    except:
        with open(pkl_path, "rb") as f:
            result_dict_ = pkl.load(f)

    result_dict = deepcopy(result_dict_)

    # FTGN dictionary
    if model_name == "FTGN":
        # put most possible trajectory in first place
        for k, v in result_dict["file2pred"].items():
            scene_id = k
            # get most possible trajectory
            best_index = get_best_pred_index(model_name, result_dict, scene_id)
            # swap with first trajectory
            temp_traj = result_dict["file2pred"][scene_id][0]
            result_dict["file2pred"][scene_id][0] = result_dict["file2pred"][scene_id][
                best_index
            ]
            result_dict["file2pred"][scene_id][best_index] = temp_traj

        result_dict["preds"] = result_dict.pop("file2pred")
        result_dict["gts"] = result_dict.pop("file2labels")
        result_dict["cities"] = {}

        for scenario_id in result_dict["preds"]:
            city = afl.get(
                os.path.join(argoverse_data_path, str(scenario_id) + ".csv")
            ).city
            result_dict["cities"][scenario_id] = city

    # HiVT dictionary
    elif model_name == "HiVT":
        # put most possible trajectory in first place
        for k, v in result_dict["forecasted_trajectories"].items():
            scene_id = k
            # get most possible trajectory index
            best_index = get_best_pred_index(model_name, result_dict, scene_id)
            # swap with first trajectory
            temp_traj = result_dict["forecasted_trajectories"][scene_id][0]
            result_dict["forecasted_trajectories"][scene_id][0] = result_dict[
                "forecasted_trajectories"
            ][scene_id][best_index]
            result_dict["forecasted_trajectories"][scene_id][best_index] = temp_traj

        result_dict["preds"] = result_dict.pop("forecasted_trajectories")
        result_dict["cities"] = result_dict.pop("city_names")
        result_dict["gts"] = {}

        for scenario_id in result_dict["preds"]:
            gt = afl.get(
                os.path.join(argoverse_data_path, str(scenario_id) + ".csv")
            ).agent_traj[
                20:
            ]  # shape (30, 2)
            result_dict["gts"][scenario_id] = gt

    # mmTransformer dictionary
    elif model_name == "mmTransformer":
        # put most possible trajectory in first place
        for k, v in result_dict["forecasted_trajectories"].items():
            scene_id = k
            # get most possible trajectory index
            best_index = get_best_pred_index(model_name, result_dict, scene_id)
            # swap with first trajectory
            temp_traj = result_dict["forecasted_trajectories"][scene_id][0]
            result_dict["forecasted_trajectories"][scene_id][0] = result_dict[
                "forecasted_trajectories"
            ][scene_id][best_index]
            result_dict["forecasted_trajectories"][scene_id][best_index] = temp_traj

        result_dict["preds"] = result_dict.pop("forecasted_trajectories")
        result_dict["gts"] = result_dict.pop("gt_trajectories")
        result_dict["cities"] = result_dict.pop("city_names")

        for scenario_id in result_dict["preds"]:
            result_dict["preds"][scenario_id] = np.array(
                result_dict["preds"][scenario_id]
            )
            result_dict["gts"][scenario_id] = np.array(result_dict["gts"][scenario_id])

        result_dict["preds"] = {int(k): v for k, v in result_dict["preds"].items()}
        result_dict["gts"] = {int(k): v for k, v in result_dict["gts"].items()}
        result_dict["cities"] = {int(k): v for k, v in result_dict["cities"].items()}

    # TNT dictionary
    elif model_name == "TNT":
        result_dict["preds"] = result_dict.pop("forecasted_trajectories")
        result_dict["gts"] = result_dict.pop("gt_trajectories")
        result_dict["cities"] = {}

        for scenario_id in result_dict["preds"]:
            city = afl.get(
                os.path.join(argoverse_data_path, str(scenario_id) + ".csv")
            ).city
            result_dict["cities"][scenario_id] = city
            result_dict["preds"][scenario_id] = np.array(
                result_dict["preds"][scenario_id]
            )
            result_dict["gts"][scenario_id] = np.array(result_dict["gts"][scenario_id])

        # print(result_dict)

    # laneGCN dictionary
    elif model_name == "LaneGCN":
        pass

    # result___ dictionary
    elif model_name == "result___":
        pass

    return result_dict


def get_best_pred_index(model_name, data, current_seq):
    if model_name == "HiVT":
        max_index = np.argmax(data["forecasted_probabilities"][int(current_seq)])

    elif model_name == "FTGN":
        max_index = np.argmax(data["file2probs"][int(current_seq)])

    elif model_name == "LaneGCN":
        return 0

    elif model_name == "mmTransformer":
        max_index = data["forecasted_probabilities"][current_seq].index(
            max(data["forecasted_probabilities"][current_seq])
        )

    elif model_name == "TNT":
        return 0

    elif model_name == "result___":
        return 0

    return max_index


def get_best_prediction(model_name, data, current_seq):
    if model_name == "HiVT":
        agent_pred_traj = data["forecasted_trajectories"][int(current_seq)]
        max_index = np.argmax(data["forecasted_probabilities"][int(current_seq)])
        most_prob_traj = agent_pred_traj[max_index]

    elif model_name == "FTGN":
        agent_pred_traj = data["file2pred"][int(current_seq)]
        max_index = np.argmax(data["file2probs"][int(current_seq)])
        most_prob_traj = agent_pred_traj[max_index]

    elif model_name == "LaneGCN":
        agent_pred_traj = data["preds"][int(current_seq)]
        most_prob_traj = agent_pred_traj[0]

    elif model_name == "mmTransformer":
        agent_pred_traj = data["forecasted_trajectories"][current_seq]
        max_index = data["forecasted_probabilities"][current_seq].index(
            max(data["forecasted_probabilities"][current_seq])
        )
        most_prob_traj = agent_pred_traj[max_index]

    elif model_name == "TNT":
        agent_pred_traj = data["forecasted_trajectories"][int(current_seq)]
        most_prob_traj = agent_pred_traj[0]

    elif model_name == "result___":
        agent_pred_traj = data["preds"][int(current_seq)]
        most_prob_traj = agent_pred_traj[0]

    return most_prob_traj


def get_best_preds_dict(pkl_path, model_name, afl, argoverse_data_path):
    try:
        result_dict = torch.load(pkl_path)
    except:
        with open(pkl_path, "rb") as f:
            result_dict = pkl.load(f)

    # Final dict
    best_preds_dict = {}

    # Sub dict in final dict
    preds_dict = {}
    gts_dict = {}
    cities_dict = {}
    # FTGN dictionary
    if model_name == "FTGN":
        for scenario_id in result_dict["file2pred"]:
            best_pred = get_best_prediction(model_name, result_dict, scenario_id)
            best_pred = np.expand_dims(best_pred, axis=0)
            gt = result_dict["file2labels"][scenario_id]
            city = afl.get(
                os.path.join(argoverse_data_path, str(scenario_id) + ".csv")
            ).city

            preds_dict[scenario_id] = best_pred
            gts_dict[scenario_id] = gt
            cities_dict[scenario_id] = city

    # HiVT dictionary
    elif model_name == "HiVT":
        for scenario_id in result_dict["forecasted_trajectories"]:
            best_pred = get_best_prediction(model_name, result_dict, scenario_id)
            best_pred = np.expand_dims(best_pred, axis=0)
            gt = afl.get(
                os.path.join(argoverse_data_path, str(scenario_id) + ".csv")
            ).agent_traj[
                20:
            ]  # shape (30, 2)
            city = result_dict["city_names"][scenario_id]

            preds_dict[scenario_id] = best_pred
            gts_dict[scenario_id] = gt
            cities_dict[scenario_id] = city

    # mmTransformer dictionary
    elif model_name == "mmTransformer":
        for scenario_id in result_dict["forecasted_trajectories"]:
            best_pred = np.array(
                get_best_prediction(model_name, result_dict, scenario_id)
            )
            best_pred = np.expand_dims(best_pred, axis=0)
            gt = np.array(result_dict["gt_trajectories"][scenario_id])
            city = result_dict["city_names"][scenario_id]

            preds_dict[scenario_id] = best_pred
            gts_dict[scenario_id] = gt
            cities_dict[scenario_id] = city

    # TNT dictionary
    elif model_name == "TNT":
        for scenario_id in result_dict["forecasted_trajectories"]:
            best_pred = np.array(
                get_best_prediction(model_name, result_dict, scenario_id)
            )
            best_pred = np.expand_dims(best_pred, axis=0)
            gt = np.array(result_dict["gt_trajectories"][scenario_id])
            city = afl.get(
                os.path.join(argoverse_data_path, str(scenario_id) + ".csv")
            ).city

            preds_dict[scenario_id] = best_pred
            gts_dict[scenario_id] = gt
            cities_dict[scenario_id] = city

    # laneGCN dictionary
    elif model_name == "LaneGCN":
        for scenario_id in result_dict["preds"]:
            best_pred = np.array(
                get_best_prediction(model_name, result_dict, scenario_id)
            )
            best_pred = np.expand_dims(best_pred, axis=0)
            gt = np.array(result_dict["gts"][scenario_id])
            city = result_dict["cities"][scenario_id]

            preds_dict[scenario_id] = best_pred
            gts_dict[scenario_id] = gt
            cities_dict[scenario_id] = city

    # result___ dictionary
    elif model_name == "result___":
        for scenario_id in result_dict["preds"]:
            best_pred = np.array(
                get_best_prediction(model_name, result_dict, scenario_id)
            )
            best_pred = np.expand_dims(best_pred, axis=0)
            gt = np.array(result_dict["gts"][scenario_id])
            city = result_dict["cities"][scenario_id]

            preds_dict[scenario_id] = best_pred
            gts_dict[scenario_id] = gt
            cities_dict[scenario_id] = city

    best_preds_dict["preds"] = preds_dict
    best_preds_dict["gts"] = gts_dict
    best_preds_dict["cities"] = cities_dict

    return best_preds_dict
