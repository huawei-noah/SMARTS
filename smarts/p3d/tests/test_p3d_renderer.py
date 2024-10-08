# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import importlib.resources as pkg_resources
import math
import threading

import numpy as np
import pytest
from panda3d.core import Thread as p3dThread

import smarts.assets
from smarts.core.colors import SceneColors
from smarts.core.coordinates import Heading, Pose
from smarts.core.plan import EndlessGoal, NavigationMission, Start
from smarts.core.scenario import Scenario
from smarts.core.utils.custom_exceptions import RendererException

AGENT_ID = "Agent-007"


@pytest.fixture
def scenario():
    mission = NavigationMission(
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
            from smarts.core.renderer_base import DEBUG_MODE as RENDERER_DEBUG_MODE
            from smarts.p3d.renderer import Renderer

            self._renderer = Renderer(
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
        self._renderer.setup(self._scenario)
        pose = Pose(
            position=np.array([71.65, 53.78, 0]),
            orientation=np.array([0, 0, 0, 0]),
            heading_=Heading(math.pi * 0.91),
        )
        with pkg_resources.path(smarts.assets, "simple_car.glb") as path:
            self._renderer.create_vehicle_node(
                path, self._vid, SceneColors.SocialVehicle, pose
            )
        self._renderer.begin_rendering_vehicle(self._vid, is_agent=False)
        for s in range(self._num_steps):
            self._renderer.render()
            pose.position[0] = pose.position[0] + s
            pose.position[1] = pose.position[1] + s
            self._renderer.update_vehicle_node(self._vid, pose)
        self._renderer.remove_vehicle_node(self._vid)


def test_multiple_renderers(scenario, request):
    assert p3dThread.isThreadingSupported()
    renderer_debug_mode = request.config.getoption("--renderer-debug-mode")
    num_renderers = 3
    rts = [RenderThread(r, scenario, renderer_debug_mode) for r in range(num_renderers)]
    for rt in rts:
        rt.start()
    for rt in rts:
        rt.join()
