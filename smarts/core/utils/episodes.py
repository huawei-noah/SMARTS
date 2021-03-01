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
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import tableprint as tp


@dataclass
class EpisodeLog:
    index: int = 0
    start_time: float = field(default_factory=lambda: time.time())
    timestep_sec: float = 0
    scores: dict = field(default_factory=lambda: defaultdict(lambda: 0))
    steps: int = 0
    scenario_map: str = ""
    scenario_routes: str = ""
    mission_hash: str = ""

    @property
    def wall_time(self):
        return time.time() - self.start_time

    @property
    def sim_time(self):
        return self.timestep_sec * self.steps

    @property
    def sim2wall_ratio(self):
        return self.sim_time / self.wall_time

    @property
    def steps_per_second(self):
        return self.steps / self.wall_time

    def record_scenario(self, scenario_log):
        self.timestep_sec = scenario_log["timestep_sec"]
        self.scenario_map = scenario_log["scenario_map"]
        self.scenario_routes = scenario_log["scenario_routes"]
        self.mission_hash = scenario_log["mission_hash"]

    def record_step(self, observations=None, rewards=None, dones=None, infos=None):
        self.steps += 1

        if dones.get("__all__", False) and infos is not None:
            for agent, score in infos.items():
                self.scores[agent] = score["score"]


def episodes(n):
    col_width = 18
    with tp.TableContext(
        [
            "Episode",
            "Sim T / Wall T",
            "Total Steps",
            "Steps / Sec",
            "Scenario Map",
            "Scenario Routes",
            "Mission (Hash)",
            "Scores",
        ],
        width=col_width,
        style="round",
    ) as table:
        for i in range(n):
            e = EpisodeLog(i)
            yield e

            row = (
                f"{e.index}/{n}",
                f"{e.sim2wall_ratio:.2f}",
                e.steps,
                f"{e.steps_per_second:.2f}",
                e.scenario_map[:col_width],
                e.scenario_routes[:col_width],
                e.mission_hash[:col_width],
            )

            score_summaries = [
                f"{score:.2f} - {agent}" for agent, score in e.scores.items()
            ]

            if len(score_summaries) == 0:
                table(row + ("",))
                continue

            table(row + (score_summaries[0],))
            if len(score_summaries) > 1:
                for s in score_summaries[1:]:
                    table(("", "", "", "", "", "", "", s))
