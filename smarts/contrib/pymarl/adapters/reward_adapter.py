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
import numpy as np

from .observation_adapter import default_obs_adapter


def default_reward_adapter(env_obs, env_reward):
    obs = default_obs_adapter(env_obs)
    center_penalty = -np.abs(obs["distance_from_center"])

    # penalize flip occurences (taking into account that the vehicle spawns in the air)
    flip_penalty = 0
    if (
        env_obs.ego_vehicle_state.speed >= 25
        and env_obs.ego_vehicle_state.position[2] > 0.85
    ):
        flip_penalty = -2 * env_obs.ego_vehicle_state.speed

    # penalise sharp turns done at high speeds
    steering_penalty = 0
    if env_obs.ego_vehicle_state.speed > 60:
        steering_penalty = -pow(
            (env_obs.ego_vehicle_state.speed - 60)
            / 20
            * (env_obs.ego_vehicle_state.steering)
            * 45
            / 4,
            2,
        )

    # penalize close proximity to other cars
    crash_penalty = -5 if bool(obs["ego_will_crash"]) else 0

    total_reward = np.sum(
        [
            1.0 * env_reward,
        ]
    )
    total_penalty = np.sum([0.1 * center_penalty, 1 * steering_penalty, crash_penalty])

    if flip_penalty != 0:
        return float((-total_reward + total_penalty) / 200.0)
    else:
        return float((total_reward + total_penalty) / 200.0)
