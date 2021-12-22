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
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from smarts.core.utils.episodes import EpisodeLog


@dataclass
class BasicEpisodeLog(EpisodeLog):
    ego_speed: dict = field(default_factory=lambda: defaultdict(lambda: []))
    num_collision: dict = field(default_factory=lambda: defaultdict(lambda: 0))
    distance_to_center: dict = field(default_factory=lambda: defaultdict(lambda: []))
    distance_to_goal: dict = field(default_factory=lambda: defaultdict(lambda: []))
    distance_to_ego_car: dict = field(default_factory=lambda: defaultdict(lambda: []))
    acceleration: dict = field(default_factory=lambda: defaultdict(lambda: 0.0))
    reach_goal: dict = field(default_factory=lambda: defaultdict(lambda: False))
    agent_step: dict = field(default_factory=lambda: defaultdict(lambda: 0))
    operations: dict = field(default_factory=lambda: defaultdict(lambda: []))

    def record_step(
        self, observations=None, actions=None, rewards=None, dones=None, infos=None
    ):
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
                self.num_collision[agent_id] += info["collision"]
                self.distance_to_goal[agent_id].append(info["distance_to_goal"])
                self.agent_step[agent_id] += 1
                self.operations[agent_id].append(actions[agent_id])
                self.distance_to_center[agent_id].append(
                    infos[agent_id]["distance_to_center"]
                )

        self.steps += 1
