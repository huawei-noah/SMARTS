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
import math
import threading

import pytest

from smarts.core.agent_interface import (
    ActionSpaceType,
    AgentInterface,
    NeighborhoodVehicles,
)
from smarts.core.colors import SceneColors
from smarts.core.coordinates import Heading, Pose
from smarts.core.renderer import Renderer
from smarts.core.scenario import EndlessGoal, Mission, Scenario, Start
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation


@pytest.fixture
def smarts():
    buddha = AgentInterface(
        max_episode_steps=1000,
        neighborhood_vehicles=NeighborhoodVehicles(radius=20),
        action=ActionSpaceType.Lane,
    )
    agents = {"Agent-007": buddha}
    smarts = SMARTS(
        agents,
        traffic_sim=SumoTrafficSimulation(headless=True),
        envision=None,
    )
    yield smarts
    smarts.destroy()


@pytest.fixture
def scenario():
    mission = Mission(
        start=Start((71.65, 63.78), Heading(math.pi * 0.91)), goal=EndlessGoal()
    )
    scenario = Scenario(
        scenario_root="scenarios/loop",
        route="basic.rou.xml",
        missions={"Agent-007": mission},
    )
    return scenario


class RenderThread(threading.Thread):
    def __init__(self, r, scenario, num_steps=3):
        self._rid = "r{}".format(r)
        super().__init__(target=self.test_renderer, name=self._rid)
        self._rdr = Renderer(self._rid)
        self._scenario = scenario
        self._num_steps = num_steps
        self._vid = "r{}_car".format(r)

    def test_renderer(self):
        self._rdr.setup(self._scenario)
        pose = Pose(
            position=[71.65, 53.78, 0],
            orientation=[0, 0, 0, 0],
            heading_=Heading(math.pi * 0.91),
        )
        self._rdr.create_vehicle_node(
            "simple_car.glb", self._vid, SceneColors.SocialVehicle.value, pose
        )
        self._rdr.begin_rendering_vehicle(self._vid, is_agent=False)
        for s in range(self._num_steps):
            self._rdr.render()
            pose.position[0] = pose.position[0] + s
            pose.position[1] = pose.position[1] + s
            self._rdr.update_vehicle_node(self._vid, pose)
        self._rdr.remove_vehicle_node(self._vid)


def test_multiple_renderers(scenario):
    num_renderers = 3
    rts = [RenderThread(r, scenario) for r in range(num_renderers)]
    for rt in rts:
        rt.start()
    for rt in rts:
        rt.join()


def test_optional_renderer(smarts, scenario):
    smarts.reset(scenario)
    assert not smarts.is_rendering
    for _ in range(10):
        smarts.step({})
    renderer = smarts.renderer
    assert smarts.is_rendering
    for _ in range(10):
        smarts.step({})
