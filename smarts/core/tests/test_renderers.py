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
import importlib.resources as pkg_resources
import math
from typing import Optional, Set

import numpy as np
import pytest

import smarts.core.glsl
from smarts.core.agent_interface import (
    ActionSpaceType,
    AgentInterface,
    DoneCriteria,
    NeighborhoodVehicles,
)
from smarts.core.colors import SceneColors
from smarts.core.coordinates import Heading, Pose
from smarts.core.plan import EndlessGoal, NavigationMission, Start
from smarts.core.renderer_base import (
    RendererBase,
    RendererNotSetUpWarning,
    ShaderStepBufferDependency,
)
from smarts.core.scenario import Scenario
from smarts.core.shader_buffer import BufferID
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.custom_exceptions import RendererException
from smarts.core.utils.tests.fixtures import large_observation

AGENT_ID = "Agent-007"


def _smarts_with_agent(agent) -> SMARTS:
    agents = {AGENT_ID: agent}
    return SMARTS(
        agents,
        traffic_sims=[SumoTrafficSimulation(headless=True)],
        envision=None,
    )


@pytest.fixture
def smarts_w_renderer():
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


@pytest.fixture(params=["p3d"], scope="module")
def renderer(request: pytest.FixtureRequest):
    renderer: Optional[RendererBase] = None

    if request.param == "p3d":
        from smarts.p3d.renderer import BACKEND_LITERALS, DEBUG_MODE, Renderer

        renderer = Renderer("na", debug_mode=DEBUG_MODE.WARNING)

    assert renderer is not None
    yield renderer

    renderer.destroy()


@pytest.fixture
def observation_buffers() -> Set[BufferID]:
    return {v for v in BufferID}


def test_optional_renderer(smarts_w_renderer: SMARTS, scenario: Scenario):
    assert not smarts_w_renderer.is_rendering
    renderer = smarts_w_renderer.renderer
    assert renderer

    smarts_w_renderer._renderer = None
    smarts_w_renderer.reset(scenario)
    assert smarts_w_renderer.is_rendering

    smarts_w_renderer._renderer = None

    with pytest.raises(expected_exception=AttributeError, match=r"NoneType"):
        smarts_w_renderer.step({AGENT_ID: "keep_lane"})

    assert not smarts_w_renderer.is_rendering


def test_no_renderer(smarts_wo_renderer: SMARTS, scenario):
    assert not smarts_wo_renderer.is_rendering
    smarts_wo_renderer.reset(scenario)
    assert not smarts_wo_renderer.is_rendering
    for _ in range(10):
        smarts_wo_renderer.step({AGENT_ID: "keep_lane"})

    assert not smarts_wo_renderer.is_rendering


def test_custom_shader_pass_buffers(
    renderer: Optional[RendererBase],
    observation_buffers: Set[BufferID],
    large_observation,
):
    # Use a full dummy observation.
    # Construct the renderer
    # Construct the shader pass
    # Use a shader that uses all buffers
    # Assign all shader buffers in the pass.
    # Update the shader pass using the current observation
    # Render
    # Test that the pixels default to white. (black is error)

    assert renderer
    renderer.reset()
    camera_id = "test_shader"
    with pkg_resources.path(
        smarts.core.glsl, "test_custom_shader_pass_shader.frag"
    ) as fshader:
        renderer.build_shader_step(
            camera_id,
            fshader_path=fshader,
            dependencies=[
                ShaderStepBufferDependency(
                    buffer_id=b_id, script_uniform_name=b_id.value
                )
                for b_id in observation_buffers
            ],
            priority=40,
            height=100,
            width=100,
        )

    camera = renderer.camera_for_id(camera_id=camera_id)
    camera.update(Pose.from_center(np.array([0, 0, 0]), Heading(0)), 10)
    camera.update(observation=large_observation)

    with pytest.warns(RendererNotSetUpWarning):
        renderer.render()

    ram_image = camera.wait_for_ram_image("RGB")
    mem_view = memoryview(ram_image)
    image: np.ndarray = np.frombuffer(mem_view, np.uint8)[::3]

    assert image[0] == 0
