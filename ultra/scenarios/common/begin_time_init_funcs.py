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
import numpy as np


def basic_vehicle_initial_distribution(probability, min_step=100000, extra_args=None):

    count = 1
    while True:
        p = np.random.uniform(0, 1) < probability
        if probability == 0.0 or p or count > min_step:
            break
        count += 1
    return count


def basic_begin_time_init_func(num_lanes, num_vehicle, **params):
    begin_times = [[] for _ in range(num_lanes)]
    for j in range(num_lanes):
        [
            begin_times[j].append(basic_vehicle_initial_distribution(**params))
            for _ in range(num_vehicle)
        ]
        begin_times[j] = np.cumsum(begin_times[j])
        begin_times[j] = [float(e) for e in begin_times[j]]
    return begin_times


def burst_begin_time_init_func(
    num_lanes,
    num_vehicle,
    vehicle_cluster_size=(4, 5),
    time_between_cluster=(12, 25),
    time_for_each_cluster=5,
):
    begin_times = []
    for j in range(num_lanes):
        vehicle_left = num_vehicle
        _begin_times = []
        time_so_far = 0

        while vehicle_left > 0:
            num_vehicle_in_cluster = min(
                np.random.randint(vehicle_cluster_size[0], vehicle_cluster_size[1] + 1),
                vehicle_left,
            )

            vehicle_spawn_time = np.random.uniform(
                0, time_for_each_cluster, size=num_vehicle_in_cluster
            )
            # vehicle_spawn_time = np.clip(np.random.exponential(10, size=len(behaviors)), 0.0, allocated_time_for_cluster)
            vehicle_spawn_time += time_so_far
            _begin_times.extend(vehicle_spawn_time)

            time_to_next_cluster = np.random.uniform(*time_between_cluster)
            time_so_far += time_to_next_cluster

            vehicle_left -= num_vehicle_in_cluster

        begin_times.append(_begin_times)
    return begin_times


def poisson_init(num_lanes, num_vehicle, temperature=10):
    vehicle_spawn_time = np.random.exponential(
        temperature, size=(num_lanes, num_vehicle)
    )
    vehicle_spawn_time = list(vehicle_spawn_time)
    vehicle_spawn_time = [np.cumsum(e) for e in vehicle_spawn_time]
    return vehicle_spawn_time
