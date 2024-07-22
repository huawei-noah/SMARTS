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
# A explaination of structure of model_metrics_results.pkl

# The structure of model_metrics_results dictionary
# The first level dictionary: (key, value) is (models name, all metric results dictionary)
first_level_dict = {
    "TNT": {},  # all metrics result of TNT
    "LaneGCN": {},
    "HiVT": {},
    "FTGN": {},
    "mmTransformer": {},
}

# The structure of all metrics result dictionary
# The second level dictionary: (key, value) is (metric name, single metric result dictionary)
second_level_dict = {
    "minFDE": {},  # single metric result dictionary
    "minADE": {},
    "RF": {},
    "minFSD": {},
    "minASD": {},
    "TAD": {},
    "TDD": {},
    "DAO": {},
    "DAC": {},
    "TAR": {},
    "TAR_ablation": {},
}

# The structure of single metrics result dictionary
# The third level dictionary: (key, value) is (scenario id, metric result of that scenario)
scene_id = 16230
third_level_dict = {scene_id: 0}  # result of that scenario

# There are some exceptions, TAD, TAR, TAR ablation
# They compose by multiple component, and structure is below
TAD_dict = {
    "TAD": 0,  # result of TAD, using all timestep of preds
    "seTAD": 0,  # result of seTAD, using only start timestep and final timestep
}

TAR_dict = {
    "normal_TAR": 0,  # TAR using normal range of acceleration (-2.0, 1.47)
    "agg_TAR": 0,  # TAR using aggresive range of acceleration (-5.08, 3.07)
    "ext_agg_TAR": 0,  # TAR using extremely aggresive range of acceleration (-5.6, 7.6)
}

TAR_ablation_dict = {
    "offroad_TAR": 0,  # TAR that consider only offroad
    "badacc_TAR": 0,  # TAR that consider only acceleration
    "wd_TAR": 0,  # TAR that consider only lane alignment
}
