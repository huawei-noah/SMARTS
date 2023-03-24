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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import math
import threading

import numpy as np
import pytest
from panda3d.core import Thread as p3dThread

from smarts.core.agent_interface import (
    ActionSpaceType,
    AgentInterface,
    DoneCriteria,
    NeighborhoodVehicles,
)
from smarts.core.colors import SceneColors
from smarts.core.coordinates import Heading, Pose
from smarts.core.plan import EndlessGoal, Mission, Start
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.custom_exceptions import RendererException

AGENT_ID = "Agent-007"


def _smarts_with_agent(agent) -> SMARTS:
    agents = {AGENT_ID: agent}
    return SMARTS(
        agents,
        traffic_sims=[SumoTrafficSimulation(headless=True)],
        envision=None,
    )


@pytest.fixture
def smarts():
    buddha = AgentInterface(
        debug=True,
        done_criteria=DoneCriteria(collision=False, off_road=False, off_route=False),
        top_down_rgb=True,
        occupancy_grid_map=True,
        drivable_area_grid_map=True,
        neighborhood_vehicle_states=NeighborhoodVehicles(radius=20),
        action=ActionSpaceType.Lane,
    )
    smarts = _smarts_with_agent(buddha)
    yield smarts
    smarts.destroy()


@pytest.fixture
def smarts_wo_renderer():
    buddha = AgentInterface(
        debug=True,
        done_criteria=DoneCriteria(collision=False, off_road=False, off_route=False),
        neighborhood_vehicle_states=NeighborhoodVehicles(radius=20),
        action=ActionSpaceType.Lane,
    )
    smarts = _smarts_with_agent(buddha)
    yield smarts
    smarts.destroy()


@pytest.fixture
def scenario():
    mission = Mission(
        start=Start(np.array((71.65, 63.78)), Heading(math.pi * 0.91)),
        goal=EndlessGoal(),
    )
    scenario = Scenario(
        scenario_root="scenarios/sumo/loop",
        traffic_specs=["scenarios/sumo/loop/build/traffic/basic.rou.xml"],
        missions={AGENT_ID: mission},
    )
    return scenario


class RenderThread(threading.Thread):
    def __init__(self, r, scenario, renderer_debug_mode: str, num_steps=3):
        self._rid = "r{}".format(r)
        super().__init__(target=self.test_renderer, name=self._rid)

        try:
            from smarts.core.renderer import DEBUG_MODE as RENDERER_DEBUG_MODE
            from smarts.core.renderer import Renderer

            self._rdr = Renderer(
                self._rid, RENDERER_DEBUG_MODE[renderer_debug_mode.upper()]
            )
        except ImportError as e:
            raise RendererException.required_to("run test_renderer.py")
        except Exception as e:
            raise e

        self._scenario = scenario
        self._num_steps = num_steps
        self._vid = "r{}_car".format(r)

    def test_renderer(self):
        self._rdr.setup(self._scenario)
        pose = Pose(
            position=np.array([71.65, 53.78, 0]),
            orientation=np.array([0, 0, 0, 0]),
            heading_=Heading(math.pi * 0.91),
        )
        self._rdr.create_vehicle_node(
            "simple_car.glb", self._vid, SceneColors.SocialVehicle, pose
        )
        self._rdr.begin_rendering_vehicle(self._vid, is_agent=False)
        for s in range(self._num_steps):
            self._rdr.render()
            pose.position[0] = pose.position[0] + s
            pose.position[1] = pose.position[1] + s
            self._rdr.update_vehicle_node(self._vid, pose)
        self._rdr.remove_vehicle_node(self._vid)


def test_multiple_renderers(scenario, request):
    assert p3dThread.isThreadingSupported()
    renderer_debug_mode = request.config.getoption("--renderer-debug-mode")
    num_renderers = 3
    rts = [RenderThread(r, scenario, renderer_debug_mode) for r in range(num_renderers)]
    for rt in rts:
        rt.start()
    for rt in rts:
        rt.join()


def test_optional_renderer(smarts: SMARTS, scenario):
    assert not smarts.is_rendering
    renderer = smarts.renderer
    assert renderer

    smarts._renderer = None
    smarts.reset(scenario)
    assert smarts.is_rendering

    smarts._renderer = None
    for _ in range(10):
        smarts.step({AGENT_ID: "keep_lane"})

    assert not smarts.is_rendering


def test_no_renderer(smarts_wo_renderer: SMARTS, scenario):
    assert not smarts_wo_renderer.is_rendering
    smarts_wo_renderer.reset(scenario)
    assert not smarts_wo_renderer.is_rendering
    for _ in range(10):
        smarts_wo_renderer.step({AGENT_ID: "keep_lane"})

    assert not smarts_wo_renderer.is_rendering
