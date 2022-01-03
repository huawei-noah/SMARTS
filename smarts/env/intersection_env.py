# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import pathlib
from typing import Dict, Optional

from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv


class IntersectionEnv(HiWayEnv):
    """An intersection environment where the agent needs to make an unprotected
    left turn in the presence of traffic.

    Episode termination:
        If any of the DoneCriteria is met or if the max steps per episode is
        reached.

    Solved requirements:
        Considered solved when the average return is greater than or equal to
        200.0 over 100 consecutive trials.
    """

    def __init__(
        self,
        agent_specs: Dict[str, AgentSpec],
        sim_name: Optional[str] = None,
        headless: bool = False,
        visdom: bool = False,
        seed: int = 42,
        envision_record_data_replay_path: Optional[str] = None,
    ):
        """
        Args:
            agent_specs (Dict[str, AgentSpec]): Specification of the agents
                that will run in the environment.
            sim_name (Optional[str], optional): Simulation name. Defaults to
                None.
            headless (bool, optional): If True, disables visualization in
                Envision. Defaults to False.
            visdom (bool, optional): If True, enables visualization of observed
                RGB images in Visdom. Defaults to False.
            seed (int, optional): Random number generator seed. Defaults to 42.
            envision_record_data_replay_path (Optional[str], optional):
                Envision's data replay output directory. Defaults to None.
        """

        scenario = str(
            pathlib.Path(__file__).absolute().parents[2]
            / "scenarios"
            / "intersections"
            / "2lane_left_turn"
        )
        build_scenario = f"scl scenario build {scenario}"
        os.system(build_scenario)

        super(IntersectionEnv, self).__init__(
            scenarios=[scenario],
            agent_specs=agent_specs,
            sim_name=sim_name,
            headless=headless,
            visdom=visdom,
            seed=seed,
            envision_record_data_replay_path=envision_record_data_replay_path,
        )
