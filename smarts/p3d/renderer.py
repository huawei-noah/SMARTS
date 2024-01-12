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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# to allow for typing to refer to class being defined (Renderer)
from __future__ import annotations

import importlib.resources as pkg_resources
import logging
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Collection, Dict, Literal, Optional, Tuple, Union

import gltf
import numpy as np
from direct.showbase.ShowBase import ShowBase

# pytype: disable=import-error
from panda3d.core import (
    Camera,
    CardMaker,
    FrameBufferProperties,
    Geom,
    GeomLinestrips,
    GeomNode,
    GeomTrifans,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexReader,
    GeomVertexWriter,
    GraphicsOutput,
    GraphicsPipe,
    NodePath,
    OrthographicLens,
    Shader,
    ShaderInput,
    Texture,
    WindowProperties,
    loadPrcFileData,
)

from smarts.core import glsl
from smarts.core.colors import Colors, SceneColors
from smarts.core.coordinates import Point, Pose
from smarts.core.masks import RenderMasks
from smarts.core.renderer_base import (
    DEBUG_MODE,
    OffscreenCamera,
    RendererBase,
    RendererNotSetUpWarning,
    ShaderStep,
    ShaderStepBufferDependency,
    ShaderStepCameraDependency,
    ShaderStepDependencyBase,
    ShaderStepVariableDependency,
)
from smarts.core.scenario import Scenario
from smarts.core.shader_buffer import BufferName
from smarts.core.signals import SignalState, signal_state_to_color
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.vehicle_state import VehicleState

if TYPE_CHECKING:
    from smarts.core.observations import Observation


# pytype: enable=import-error

BACKEND_LITERALS = Literal[
    "pandagl",
    "pandadx9",
    "pandagles",
    "pandagles2",
    "p3headlessgl",
    "p3tinydisplay",
]


class _ShowBaseInstance(ShowBase):
    """Wraps a singleton instance of ShowBase from Panda3D."""

    _debug_mode: DEBUG_MODE = DEBUG_MODE.WARNING
    _rendering_backend: BACKEND_LITERALS = "pandagl"

    def __new__(cls):
        # Singleton pattern:  ensure only 1 ShowBase instance
        if "__it__" not in cls.__dict__:
            if cls._debug_mode <= DEBUG_MODE.INFO:
                loadPrcFileData("", "gl-debug #t")
            loadPrcFileData(
                "",
                f"load-display {cls._rendering_backend}",
            )
            loadPrcFileData("", "aux-display pandagl")
            loadPrcFileData("", "aux-display pandadx9")
            loadPrcFileData("", "aux-display pandadx8")
            # loadPrcFileData("", "aux-display pandagles")
            # loadPrcFileData("", "aux-display pandagles2")
            loadPrcFileData("", "aux-display p3headlessgl")
            loadPrcFileData("", "aux-display p3tinydisplay")

            # disable vsync otherwise we are limited to refresh-rate of screen
            loadPrcFileData("", "sync-video false")
            loadPrcFileData("", "model-path %s" % os.getcwd())
            # TODO: the following speeds up rendering a bit... might consider it.
            # loadPrcFileData("", "model-cache-dir %s/.panda3d_cache" % os.getcwd())
            loadPrcFileData("", "audio-library-name null")
            loadPrcFileData("", "gl-version 3 3")
            loadPrcFileData("", f"notify-level {cls._debug_mode.name.lower()}")
            loadPrcFileData("", "print-pipe-types false")
            # loadPrcFileData("", "basic-shaders-only #t")
            # https://www.panda3d.org/manual/?title=Multithreaded_Render_Pipeline
            # loadPrcFileData('', 'threading-model Cull/Draw')
            # have makeTextureBuffer create a visible window
            # loadPrcFileData('', 'show-buffers true')
        it = cls.__dict__.get("__it__")
        if it is None:
            cls.__it__ = it = object.__new__(cls)
            it.init()
        return it

    def __init__(self):
        pass  # singleton pattern, uses init() instead (don't call super().__init__() here!)

    def init(self):
        """Initializer for the purposes of maintaining a singleton of this class."""
        self._render_lock = Lock()
        try:
            # There can be only 1 ShowBase instance at a time.
            super().__init__(windowType="offscreen")

            gltf.patch_loader(self.loader)
            self.setBackgroundColor(0, 0, 0, 1)

            # Displayed framerate is misleading since we are not using a realtime clock
            self.setFrameRateMeter(False)

        except Exception as e:
            raise e

    @classmethod
    def set_rendering_verbosity(cls, debug_mode: DEBUG_MODE):
        """Set rendering debug information verbosity."""
        cls._debug_mode = debug_mode
        loadPrcFileData("", f"notify-level {cls._debug_mode.name.lower()}")

    @classmethod
    def set_rendering_backend(
        cls,
        rendering_backend: BACKEND_LITERALS,
    ):
        """Sets the rendering backend."""
        if "__it__" not in cls.__dict__:
            cls._rendering_backend = rendering_backend
        else:
            warnings.warn("Cannot apply rendering backend after setup.")

    def destroy(self):
        """Destroy this renderer and clean up all remaining resources."""
        super().destroy()
        self.__class__.__it__ = None

    def __del__(self):
        try:
            self.destroy()
        except (AttributeError, TypeError):
            pass

    def setup_sim_root(self, simid: str):
        """Creates the simulation root node in the scene graph."""
        root_np = NodePath(simid)
        with self._render_lock:
            root_np.reparentTo(self.render)
        with pkg_resources.path(
            glsl, "unlit_shader.vert"
        ) as vshader_path, pkg_resources.path(
            glsl, "unlit_shader.frag"
        ) as fshader_path:
            unlit_shader = Shader.load(
                Shader.SL_GLSL,
                vertex=str(vshader_path.absolute()),
                fragment=str(fshader_path.absolute()),
            )
            root_np.setShader(unlit_shader, priority=10)
        return root_np

    def render_node(self, sim_root: NodePath):
        """Render a panda3D scene graph from the given node."""
        # Hack to prevent other SMARTS instances from also rendering
        # when we call poll() here.
        hidden = []
        with self._render_lock:
            for node_path in self.render.children:
                if node_path != sim_root and not node_path.isHidden():
                    node_path.hide()
                    hidden.append(node_path)
            self.taskMgr.mgr.poll()
            for node_path in hidden:
                node_path.show()


@dataclass
class _P3DCameraMixin:

    camera_np: NodePath
    buffer: GraphicsOutput
    tex: Texture

    def wait_for_ram_image(self, img_format: str, retries=100):
        """Attempt to acquire a graphics buffer."""
        # Rarely, we see dropped frames where an image is not available
        # for our observation calculations.
        #
        # We've seen this happen fairly reliable when we are initializing
        # a multi-agent + multi-instance simulation.
        #
        # To deal with this, we can try to force a render and block until
        # we are fairly certain we have an image in ram to return to the user
        for i in range(retries):
            if self.tex.mightHaveRamImage():
                break
            getattr(self, "renderer").log.debug(
                f"No image available (attempt {i}/{retries}), forcing a render"
            )
            region = self.buffer.getDisplayRegion(0)
            region.window.engine.renderFrame()

        assert self.tex.mightHaveRamImage()
        ram_image = self.tex.getRamImageAs(img_format)
        assert ram_image is not None
        return ram_image

    @property
    def image_dimensions(self):
        return (self.tex.getXSize(), self.tex.getYSize())

    @property
    def position(self) -> Tuple[float, float, float]:
        raise NotImplementedError()

    @property
    def padding(self) -> Tuple[int, int, int, int]:
        """The padding on the image. This follows the "css" convention: (top, left, bottom, right)."""
        return self.tex.getPadYSize(), self.tex.getPadXSize(), 0, 0

    @property
    def heading(self) -> float:
        return np.radians(self.camera_np.getH())

    def teardown(self):
        """Clean up internal resources."""
        self.camera_np.removeNode()
        region = self.buffer.getDisplayRegion(0)
        region.window.clearRenderTextures()
        self.buffer.removeAllDisplayRegions()
        getattr(self, "renderer").remove_buffer(self.buffer)


@dataclass
class P3DOffscreenCamera(_P3DCameraMixin, OffscreenCamera):
    """A camera used for rendering images to a graphics buffer."""

    def update(self, pose: Pose, height: float, *args, **kwargs):
        """Update the location of the camera.
        Args:
            pose:
                The pose of the camera target.
            height:
                The height of the camera above the camera target.
        """
        pos, heading = pose.as_panda3d()
        self.camera_np.setPos(pos[0], pos[1], height)
        self.camera_np.lookAt(*pos)
        self.camera_np.setH(heading)

    @property
    def position(self) -> Tuple[float, float, float]:
        return self.camera_np.getPos()


@dataclass
class P3DShaderStep(_P3DCameraMixin, ShaderStep):
    """A camera used for rendering images using a shader and a full-screen quad."""

    fullscreen_quad_node: NodePath

    def update(self, pose: Pose = None, height: float = None, **kwargs):
        """Update the location of the shader directional values.
        Args:
            pose:
                The pose of the camera target.
            height:
                The height of the camera above the camera target.
        """
        inputs = {}
        if pose:
            self.fullscreen_quad_node.setShaderInputs(
                iHeading=pose.heading,
                iTranslation=(pose.point.x, pose.point.y),
            )
            inputs["iHeading"] = pose.heading
            inputs["iTranslation"] = (pose.point.x, pose.point.y)
            # self.fullscreen_quad_node.setShaderInput("iHeading", pose.heading)
            # self.fullscreen_quad_node.setShaderInput(
            #     "iTranslation", n1=pose.point.x, n2=pose.point.y
            # )
        if height:
            inputs["iElevation"] = height
        if observation := kwargs.get("observation"):
            observation: Observation
            inputs[BufferName.DELTA_TIME.value] = observation.dt
            inputs[BufferName.ELAPSED_SIM_TIME.value] = observation.elapsed_sim_time
            inputs[
                BufferName.EGO_VEHICLE_STATE_HEADING.value
            ] = observation.ego_vehicle_state.heading
            inputs[
                BufferName.EGO_VEHICLE_STATE_SPEED.value
            ] = observation.ego_vehicle_state.speed
            inputs[
                BufferName.EGO_VEHICLE_STATE_YAW_RATE.value
            ] = observation.ego_vehicle_state.yaw_rate
            inputs[
                BufferName.EGO_VEHICLE_STATE_LANE_INDEX.value
            ] = observation.ego_vehicle_state.yaw_rate
            inputs[BufferName.DISTANCE_TRAVELLED.value] = observation.distance_travelled

            inputs[BufferName.STEP_COUNT.value] = observation.step_count
            inputs[BufferName.STEPS_COMPLETED.value] = observation.steps_completed
            if vehicle_spec := observation.ego_vehicle_state.mission.vehicle_spec:
                inputs[BufferName.VEHICLE_TYPE.value] = hash(
                    vehicle_spec.veh_config_type
                )
            else:
                inputs[BufferName.VEHICLE_TYPE.value] = -1

            inputs[BufferName.EVENTS_OFF_ROAD.value] = int(observation.events.off_road)
            inputs[BufferName.EVENTS_OFF_ROUTE.value] = int(
                observation.events.off_route
            )
            inputs[BufferName.EVENTS_ON_SHOULDER.value] = int(
                observation.events.on_shoulder
            )
            inputs[BufferName.EVENTS_WRONG_WAY.value] = int(
                observation.events.wrong_way
            )
            inputs[BufferName.EVENTS_NOT_MOVING.value] = int(
                observation.events.not_moving
            )
            inputs[BufferName.EVENTS_REACHED_GOAL.value] = int(
                observation.events.reached_goal
            )
            inputs[BufferName.EVENTS_REACHED_MAX_EPISODE_STEPS.value] = int(
                observation.events.reached_max_episode_steps
            )
            inputs[BufferName.EVENTS_AGENTS_ALIVE_DONE.value] = int(
                observation.events.agents_alive_done
            )
            inputs[BufferName.EVENTS_INTEREST_DONE.value] = int(
                observation.events.interest_done
            )
            inputs[BufferName.UNDER_THIS_VEHICLE_CONTROL.value] = int(
                observation.under_this_agent_control
            )

            inputs[BufferName.EGO_VEHICLE_STATE_POSITION.value] = tuple(
                observation.ego_vehicle_state.position
            )
            inputs[
                BufferName.EGO_VEHICLE_STATE_BOUNDING_BOX.value
            ] = observation.ego_vehicle_state.bounding_box.as_lwh
            if lane_position := observation.ego_vehicle_state.lane_position:
                inputs[BufferName.EGO_VEHICLE_STATE_LANE_POSITION.value] = tuple(
                    lane_position
                )

            inputs[BufferName.EGO_VEHICLE_STATE_LINEAR_VELOCITY.value] = tuple(
                observation.ego_vehicle_state.linear_velocity
            )
            inputs[BufferName.EGO_VEHICLE_STATE_ANGULAR_VELOCITY.value] = tuple(
                observation.ego_vehicle_state.angular_velocity
            )
            if observation.ego_vehicle_state.linear_acceleration is not None:
                inputs[BufferName.EGO_VEHICLE_STATE_LINEAR_ACCELERATION.value] = tuple(
                    observation.ego_vehicle_state.linear_acceleration
                )
                inputs[BufferName.EGO_VEHICLE_STATE_ANGULAR_ACCELERATION.value] = tuple(
                    observation.ego_vehicle_state.angular_acceleration
                )
            if observation.ego_vehicle_state.linear_jerk is not None:
                inputs[BufferName.EGO_VEHICLE_STATE_LINEAR_JERK.value] = tuple(
                    observation.ego_vehicle_state.linear_jerk
                )
                inputs[BufferName.EGO_VEHICLE_STATE_ANGULAR_JERK.value] = tuple(
                    observation.ego_vehicle_state.angular_jerk
                )

            inputs[BufferName.EGO_VEHICLE_STATE_ROAD_ID.value] = hash(
                observation.ego_vehicle_state.road_id
            )
            inputs[BufferName.EGO_VEHICLE_STATE_LANE_ID.value] = hash(
                observation.ego_vehicle_state.lane_id
            )

            # XXX: Float cast is needed because Panda3D reacts badly to numpy types.
            nvs_positions = [
                tuple(float(round(d, 4)) for d in vs.position)
                for vs in observation.neighborhood_vehicle_states
            ]
            inputs[
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_POSITION.value
            ] = nvs_positions
            nvs_bounding_box = [
                vs.bounding_box.as_lwh for vs in observation.neighborhood_vehicle_states
            ]
            inputs[
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_BOUNDING_BOX.value
            ] = nvs_bounding_box
            nvs_lane_positions = [
                tuple(float(v) for v in vs.lane_position)
                if vs.lane_position is not None
                else (-1.0, -1.0, -1.0)
                for vs in observation.neighborhood_vehicle_states
            ]
            inputs[
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_LANE_POSITION.value
            ] = nvs_lane_positions

            nvs_headings = [
                float(vs.heading) for vs in observation.neighborhood_vehicle_states
            ]
            inputs[BufferName.NEIGHBORHOOD_VEHICLE_STATES_HEADING.value] = nvs_headings
            nvs_speeds = [
                float(vs.speed) for vs in observation.neighborhood_vehicle_states
            ]
            inputs[BufferName.NEIGHBORHOOD_VEHICLE_STATES_SPEED.value] = nvs_speeds

            nvs_road_ids = [
                hash(vs.road_id) for vs in observation.neighborhood_vehicle_states
            ]
            inputs[BufferName.NEIGHBORHOOD_VEHICLE_STATES_ROAD_ID.value] = nvs_road_ids
            nvs_lane_ids = [
                hash(vs.lane_id) for vs in observation.neighborhood_vehicle_states
            ]
            inputs[BufferName.NEIGHBORHOOD_VEHICLE_STATES_LANE_ID.value] = nvs_lane_ids
            nvs_lane_indices = [
                vs.lane_index for vs in observation.neighborhood_vehicle_states
            ]
            inputs[
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_LANE_INDEX.value
            ] = nvs_lane_indices
            nvs_lane_interest = [
                int(vs.interest) for vs in observation.neighborhood_vehicle_states
            ]
            inputs[
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_INTEREST.value
            ] = nvs_lane_interest

        if inputs:
            self.fullscreen_quad_node.setShaderInputs(**inputs)

    @property
    def position(self) -> Tuple[float, float, float]:
        raise ValueError("Shader step does not have a position.")


class Renderer(RendererBase):
    """The utility used to render simulation geometry."""

    def __init__(
        self,
        simid: str,
        debug_mode: DEBUG_MODE = DEBUG_MODE.ERROR,
        rendering_backend: BACKEND_LITERALS = "pandagl",
    ):
        self._log: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._is_setup = False
        self._simid = simid
        self._root_np = None
        self._vehicles_np = None
        self._signals_np = None
        self._road_map_np = None
        self._dashed_lines_np = None
        self._vehicle_nodes = {}
        self._signal_nodes = {}
        self._camera_nodes: Dict[str, Union[P3DOffscreenCamera, P3DShaderStep]] = {}
        _ShowBaseInstance.set_rendering_verbosity(debug_mode=debug_mode)
        _ShowBaseInstance.set_rendering_backend(rendering_backend=rendering_backend)
        # Note: Each instance of the SMARTS simulation will have its own Renderer,
        # but all Renderer objects share the same ShowBaseInstance.
        self._showbase_instance: _ShowBaseInstance = _ShowBaseInstance()
        self._interest_filter: Optional[re.Pattern] = None
        self._interest_color: Optional[Union[Colors, SceneColors]] = None

    @property
    def id(self):
        """The id of the simulation rendered."""
        return self._simid

    @property
    def is_setup(self) -> bool:
        """If the renderer has been fully initialized."""
        return self._is_setup

    @property
    def log(self) -> logging.Logger:
        """The rendering logger."""
        return self._log

    def remove_buffer(self, buffer):
        """Remove the rendering buffer."""
        self._showbase_instance.graphicsEngine.removeWindow(buffer)

    def _load_line_data(self, path: Path, name: str) -> GeomNode:
        # Extract geometry from model
        lines = []
        road_lines_np = self._showbase_instance.loader.loadModel(path, noCache=True)
        geomNodeCollection = road_lines_np.findAllMatches("**/+GeomNode")
        for nodePath in geomNodeCollection:
            geomNode = nodePath.node()
            geom = geomNode.getGeom(0)
            vdata = geom.getVertexData()
            vreader = GeomVertexReader(vdata, "vertex")
            pts = []
            while not vreader.isAtEnd():
                v = vreader.getData3()
                pts.append((v.x, v.y, v.z))
            lines.append(pts)

        # Create geometry node
        geo_format = GeomVertexFormat.getV3()
        vdata = GeomVertexData(name, geo_format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")

        prim = GeomLinestrips(Geom.UHStatic)
        for pts in lines:
            for x, y, z in pts:
                vertex.addData3(x, y, z)
            prim.add_next_vertices(len(pts))
            assert prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node_path = GeomNode(name)
        node_path.addGeom(geom)
        return node_path

    def _ensure_root(self):
        if self._root_np is None:
            self._root_np = self._showbase_instance.setup_sim_root(self._simid)

    def load_road_map(self, map_path: Path):
        # Load map
        self._ensure_root()
        if self._road_map_np:
            self._log.debug(
                "road_map=%s already exists. Removing and adding a new "
                "one from glb_path=%s",
                self._road_map_np,
                map_path,
            )
        map_np = self._showbase_instance.loader.loadModel(map_path, noCache=True)
        node_path = self._root_np.attachNewNode("road_map")
        map_np.reparent_to(node_path)
        node_path.hide(RenderMasks.OCCUPANCY_HIDE)
        node_path.setColor(SceneColors.Road.value)
        self._road_map_np = node_path
        self._is_setup = True
        return map_np.getBounds()

    def setup(self, scenario: Scenario):
        """Initialize this renderer."""
        self._ensure_root()
        self._vehicles_np = self._root_np.attachNewNode("vehicles")
        self._signals_np = self._root_np.attachNewNode("signals")

        map_path = scenario.map_glb_filepath
        map_dir = Path(map_path).parent

        # Load map
        self.load_road_map(map_path)

        # Road lines (solid, yellow)
        road_lines_path = map_dir / "road_lines.glb"
        if road_lines_path.exists():
            road_lines_np = self._load_line_data(road_lines_path, "road_lines")
            solid_lines_np = self._root_np.attachNewNode(road_lines_np)
            solid_lines_np.setColor(SceneColors.EdgeDivider.value)
            solid_lines_np.hide(RenderMasks.OCCUPANCY_HIDE)
            solid_lines_np.setRenderModeThickness(2)

        # Lane lines (dashed, white)
        lane_lines_path = map_dir / "lane_lines.glb"
        if lane_lines_path.exists():
            lane_lines_np = self._load_line_data(lane_lines_path, "lane_lines")
            dashed_lines_np = self._root_np.attachNewNode(lane_lines_np)
            dashed_lines_np.setColor(SceneColors.LaneDivider.value)
            dashed_lines_np.hide(RenderMasks.OCCUPANCY_HIDE)
            dashed_lines_np.setRenderModeThickness(2)
            with pkg_resources.path(
                glsl, "dashed_line_shader.vert"
            ) as vshader_path, pkg_resources.path(
                glsl, "dashed_line_shader.frag"
            ) as fshader_path:
                dashed_line_shader = Shader.load(
                    Shader.SL_GLSL,
                    vertex=str(vshader_path.absolute()),
                    fragment=str(fshader_path.absolute()),
                )
                dashed_lines_np.setShader(dashed_line_shader, priority=20)
                dashed_lines_np.setShaderInput(
                    "iResolution", self._showbase_instance.getSize()
                )
            self._dashed_lines_np = dashed_lines_np
        if scenario_metadata := scenario.metadata:
            if interest_pattern := scenario_metadata.get(
                "actor_of_interest_re_filter", None
            ):
                self._interest_filter = re.compile(interest_pattern)
                self._interest_color = scenario_metadata.get(
                    "actor_of_interest_color", SceneColors.SocialAgent
                )
        self._is_setup = True

    def render(self):
        """Render the scene graph of the simulation."""
        if not self._is_setup:
            self._ensure_root()
            warnings.warn(
                "Renderer is not setup. Rendering before scene setup may be unintentional.",
                RendererNotSetUpWarning,
            )
        self._showbase_instance.render_node(self._root_np)

    def reset(self):
        """Reset the render back to initialized state."""
        self._vehicles_np.removeNode()
        self._vehicles_np = self._root_np.attachNewNode("vehicles")
        self._signals_np.removeNode()
        self._signals_np = self._root_np.attachNewNode("signals")
        self._vehicle_nodes = {}
        self._signal_nodes = {}

    def step(self):
        """provided for non-SMARTS uses; normally not used by SMARTS."""
        self._showbase_instance.taskMgr.step()

    def sync(self, sim_frame: SimulationFrame):
        """Update the current state of the vehicles and signals within the renderer."""
        signal_ids = set()
        for actor_id, actor_state in sim_frame.actor_states_by_id.items():
            if isinstance(actor_state, VehicleState):
                self.update_vehicle_node(actor_id, actor_state.pose)
            elif isinstance(actor_state, SignalState):
                signal_ids.add(actor_id)
                color = signal_state_to_color(actor_state.state)
                if actor_id not in self._signal_nodes:
                    self.create_signal_node(actor_id, actor_state.stopping_pos, color)
                    self.begin_rendering_signal(actor_id)
                else:
                    self.update_signal_node(actor_id, actor_state.stopping_pos, color)

        missing_vehicle_ids = set(self._vehicle_nodes) - set(sim_frame.vehicle_ids)
        missing_signal_ids = set(self._signal_nodes) - signal_ids

        for vid in missing_vehicle_ids:
            self.remove_vehicle_node(vid)
        for sig_id in missing_signal_ids:
            self.remove_signal_node(sig_id)

    def teardown(self):
        """Clean up internal resources."""
        if self._root_np is not None:
            self._root_np.clearLight()
            self._root_np.removeNode()
            self._root_np = None
        self._vehicles_np = None
        for sig_id in list(self._signal_nodes):
            self.remove_signal_node(sig_id)
        self._signals_np = None
        self._road_map_np = None
        self._dashed_lines_np = None
        self._is_setup = False

    def destroy(self):
        """Destroy the renderer. Cleans up all remaining renderer resources."""
        self.teardown()
        self._showbase_instance = None

    def __del__(self):
        self.destroy()

    def set_interest(self, interest_filter: re.Pattern, interest_color: Colors):
        """Sets the color of all vehicles that have ids that match the given pattern.

        Args:
            interest_filter (re.Pattern): The regular expression pattern to match.
            interest_color (Colors): The color that the vehicle should show as.
        """
        assert isinstance(interest_filter, re.Pattern)
        self._interest_filter = interest_filter
        self._interest_color = interest_color

    def create_vehicle_node(
        self,
        glb_model: Union[str, Path],
        vid: str,
        color: Union[Colors, SceneColors],
        pose: Pose,
    ):
        """Create a vehicle node."""
        if vid in self._vehicle_nodes:
            return False
        node_path = self._showbase_instance.loader.loadModel(glb_model)
        node_path.setName("vehicle-%s" % vid)
        if (
            self._interest_filter is not None
            and self._interest_color is not None
            and self._interest_filter.match(vid)
        ):
            node_path.setColor(self._interest_color.value)
        else:
            node_path.setColor(color.value)
        pos, heading = pose.as_panda3d()
        node_path.setPosHpr(*pos, heading, 0, 0)
        node_path.hide(RenderMasks.DRIVABLE_AREA_HIDE)
        if color in (Colors.Red, SceneColors.Agent):
            node_path.hide(RenderMasks.OCCUPANCY_HIDE)
        self._vehicle_nodes[vid] = node_path
        return True

    def begin_rendering_vehicle(self, vid: str, is_agent: bool):
        """Add the vehicle node to the scene graph"""
        vehicle_path = self._vehicle_nodes.get(vid, None)
        if not vehicle_path:
            self._log.warning("Renderer ignoring invalid vehicle id: %s", vid)
            return
        vehicle_path.reparentTo(self._vehicles_np)

    def update_vehicle_node(self, vid: str, pose: Pose):
        """Move the specified vehicle node."""
        vehicle_path = self._vehicle_nodes.get(vid, None)
        if not vehicle_path:
            self._log.warning("Renderer ignoring invalid vehicle id: %s", vid)
            return
        pos, heading = pose.as_panda3d()
        vehicle_path.setPosHpr(*pos, heading, 0, 0)

    def remove_vehicle_node(self, vid: str):
        """Remove a vehicle node"""
        vehicle_path = self._vehicle_nodes.get(vid, None)
        if not vehicle_path:
            self._log.warning("Renderer ignoring invalid vehicle id: %s", vid)
            return
        vehicle_path.removeNode()
        del self._vehicle_nodes[vid]

    def create_signal_node(
        self, sig_id: str, position: Point, color: Union[Colors, SceneColors]
    ):
        """Create a signal node."""
        if sig_id in self._signal_nodes:
            return False

        # Create geometry node
        name = f"signal-{sig_id}"
        geo_format = GeomVertexFormat.getV3()
        vdata = GeomVertexData(name, geo_format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")

        num_pts = 10  # number of points around the circumference
        seg_radians = 2 * math.pi / num_pts
        vertex.addData3(0, 0, 0)
        for i in range(num_pts):
            angle = i * seg_radians
            x = math.cos(angle)
            y = math.sin(angle)
            vertex.addData3(x, y, 0)

        prim = GeomTrifans(Geom.UHStatic)
        prim.addVertex(0)  # add center point
        prim.add_next_vertices(num_pts)  # add outer points
        prim.addVertex(1)  # add first outer point again to complete the circle
        assert prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        geom_node = GeomNode(name)
        geom_node.addGeom(geom)

        node_path = self._root_np.attachNewNode(geom_node)
        node_path.setName(name)
        node_path.setColor(color.value)
        node_path.setPos(position.x, position.y, 0.01)
        node_path.setScale(0.9, 0.9, 1)
        node_path.hide(RenderMasks.DRIVABLE_AREA_HIDE)
        self._signal_nodes[sig_id] = node_path
        return True

    def begin_rendering_signal(self, sig_id: str):
        """Add the signal node to the scene graph"""
        signal_np = self._signal_nodes.get(sig_id, None)
        if not signal_np:
            self._log.warning("Renderer ignoring invalid signal id: %s", sig_id)
            return
        signal_np.reparentTo(self._signals_np)

    def update_signal_node(
        self, sig_id: str, position: Point, color: Union[Colors, SceneColors]
    ):
        """Move the specified signal node."""
        signal_np = self._signal_nodes.get(sig_id, None)
        if not signal_np:
            self._log.warning("Renderer ignoring invalid signal id: %s", sig_id)
            return
        signal_np.setPos(position.x, position.y, 0.01)
        signal_np.setColor(color.value)

    def remove_signal_node(self, sig_id: str):
        """Remove a signal node"""
        signal_np = self._signal_nodes.get(sig_id, None)
        if not signal_np:
            self._log.warning("Renderer ignoring invalid signal id: %s", sig_id)
            return
        signal_np.removeNode()
        del self._signal_nodes[sig_id]

    def camera_for_id(self, camera_id: str) -> Union[P3DOffscreenCamera, P3DShaderStep]:
        """Get a camera by its id."""
        camera = self._camera_nodes.get(camera_id)
        assert (
            camera is not None
        ), f"Camera {camera_id} does not exist, have you created this camera?"
        return camera

    def build_offscreen_camera(
        self,
        name: str,
        mask: int,
        width: int,
        height: int,
        resolution: float,
    ) -> None:
        """Generates a new off-screen camera."""
        # setup buffer
        win_props = WindowProperties.size(width, height)
        fb_props = FrameBufferProperties()
        fb_props.setRgbColor(True)
        fb_props.setRgbaBits(8, 8, 8, 1)
        # XXX: Though we don't need the depth buffer returned, setting this to 0
        #      causes undefined behavior where the ordering of meshes is random.
        fb_props.setDepthBits(8)

        buffer = self._showbase_instance.win.engine.makeOutput(
            self._showbase_instance.pipe,
            "{}-buffer".format(name),
            -100,
            fb_props,
            win_props,
            GraphicsPipe.BFRefuseWindow,
            self._showbase_instance.win.getGsg(),
            self._showbase_instance.win,
        )
        buffer.setClearColor((0, 0, 0, 0))  # Set background color to black

        # Necessary for the lane lines to be in the proper proportions
        if self._dashed_lines_np is not None:
            self._dashed_lines_np.setShaderInput(
                "iResolution", (buffer.size.x, buffer.size.y)
            )

        # setup texture
        tex = Texture()
        region = buffer.getDisplayRegion(0)
        region.window.addRenderTexture(
            tex, GraphicsOutput.RTM_copy_ram, GraphicsOutput.RTP_color
        )

        # setup camera
        lens = OrthographicLens()
        lens.setFilmSize(width * resolution, height * resolution)

        camera_np = self._showbase_instance.makeCamera(
            buffer, camName=name, scene=self._root_np, lens=lens
        )
        camera_np.reparentTo(self._root_np)

        # mask is set to make undesirable objects invisible to this camera
        camera_np.node().setCameraMask(camera_np.node().getCameraMask() & mask)

        camera = P3DOffscreenCamera(self, camera_np, buffer, tex)
        self._camera_nodes[name] = camera

    def build_shader_step(
        self,
        name: str,
        fshader_path: Union[str, Path],
        dependencies: Collection[
            ShaderStepDependencyBase  # Union[ShaderStepCameraDependency, ShaderStepVariableDependency, ShaderStepBufferDependency]
        ],
        priority: int,
        height: int,
        width: int,
    ) -> None:
        # setup buffer
        win_props = WindowProperties.size(width, height)
        fb_props = FrameBufferProperties()
        fb_props.setRgbColor(True)
        fb_props.setRgbaBits(8, 8, 8, 0)
        # XXX: Though we don't need the depth buffer returned, setting this to 0
        #      causes undefined behavior where the ordering of meshes is random.
        fb_props.setDepthBits(0)

        buffer = self._showbase_instance.win.engine.makeOutput(
            self._showbase_instance.pipe,
            "{}-buffer".format(name),
            priority,
            fb_props,
            win_props,
            GraphicsPipe.BFRefuseWindow,
            self._showbase_instance.win.getGsg(),
            self._showbase_instance.win,
        )

        cm = CardMaker("filter-stage-quad")
        cm.setFrameFullscreenQuad()
        quad = NodePath(cm.generate())
        quad.setDepthTest(0)
        quad.setDepthWrite(0)
        quad.setColor(1, 0.5, 0.5, 1)

        # setup texture
        tex = Texture()
        # tex.setup_2d_texture(width, height, Texture.T_unsigned_byte, Texture.F_r8i)
        region = buffer.getDisplayRegion(0)
        region.window.addRenderTexture(
            tex, GraphicsOutput.RTM_copy_ram, GraphicsOutput.RTP_color
        )

        # setup camera
        lens = OrthographicLens()
        lens.setFilmSize(width, height)
        lens.setFilmSize(2, 2)
        lens.setFilmOffset(0, 0)
        lens.setNearFar(-1000, 1000)

        quadcamnode = Camera(name)
        quadcamnode.setLens(lens)
        quadcam: NodePath = quad.attachNewNode(quadcamnode)

        dr = buffer.makeDisplayRegion((0, 1, 0, 1))
        dr.disableClears()
        dr.setCamera(quadcam)
        dr.setActive(True)
        dr.setScissorEnabled(False)

        # buffer clearing
        buffer.setClearColor((0, 0, 0, 0))  # Set background color to black
        buffer.setClearColorActive(True)

        assert tex.getExpectedRamImageSize() == tex.getXSize() * tex.getYSize() * 3

        with pkg_resources.path(glsl, "unlit_shader.vert") as vshader_path:
            quad.setShader(
                Shader.load(Shader.SL_GLSL, vertex=vshader_path, fragment=fshader_path)
            )
            camera_dependencies = tuple(
                c for c in dependencies if isinstance(c, ShaderStepCameraDependency)
            )
            cameras = tuple(
                self.camera_for_id(c.camera_id)
                for c in camera_dependencies
                if isinstance(c, ShaderStepCameraDependency)
            )
            for dep, dep_cam in zip(camera_dependencies, cameras):
                quad.setShaderInput(dep.script_variable_name, dep_cam.tex)
                size_x, size_y = dep_cam.image_dimensions
                quad.setShaderInput(
                    f"{dep.script_variable_name}Resolution", n1=size_x, n2=size_y
                )
            buffer_dependencies = tuple(
                v for v in dependencies if isinstance(v, ShaderStepBufferDependency)
            )

            Renderer._set_shader_inputs(
                quad, width, height, buffers=buffer_dependencies
            )
            variable_dependencies = tuple(
                v for v in dependencies if isinstance(v, ShaderStepVariableDependency)
            )
            for dep in variable_dependencies:
                shader_input = ShaderInput(dep.script_variable_name, dep.value)
                quad.setShaderInput(shader_input)

            camera = P3DShaderStep(
                self,
                shader_file=fshader_path,
                camera_dependencies=cameras,
                buffer_dependencies=buffer_dependencies,
                camera_np=quadcam,
                buffer=buffer,
                tex=tex,
                fullscreen_quad_node=quad,
            )
            self._camera_nodes[name] = camera

    @staticmethod
    def _set_shader_inputs(
        surface, width, height, buffers: Tuple[ShaderStepBufferDependency]
    ):

        inputs = {
            "iResolution": (width, height),
            "iTranslation": (0.0, 0.0),
            "iHeading": 0.0,
            "iElevation": 0.0,
        }

        for svn, bn in ((b.script_variable_name, b.buffer_name) for b in buffers):
            initial_value = None
            # SINGLE VALUES
            if bn in (
                BufferName.DELTA_TIME,
                BufferName.ELAPSED_SIM_TIME,
                BufferName.EGO_VEHICLE_STATE_HEADING,
                BufferName.EGO_VEHICLE_STATE_SPEED,
                BufferName.EGO_VEHICLE_STATE_STEERING,
                BufferName.EGO_VEHICLE_STATE_YAW_RATE,
                BufferName.EGO_VEHICLE_STATE_LANE_INDEX,
                BufferName.DISTANCE_TRAVELLED,
            ):
                initial_value = float()
            elif bn in (
                BufferName.STEP_COUNT,
                BufferName.STEPS_COMPLETED,
                BufferName.VEHICLE_TYPE,
            ):
                initial_value = int()
            elif bn in (
                BufferName.EVENTS_OFF_ROAD,
                BufferName.EVENTS_OFF_ROUTE,
                BufferName.EVENTS_ON_SHOULDER,
                BufferName.EVENTS_WRONG_WAY,
                BufferName.EVENTS_NOT_MOVING,
                BufferName.EVENTS_REACHED_GOAL,
                BufferName.EVENTS_REACHED_MAX_EPISODE_STEPS,
                BufferName.EVENTS_AGENTS_ALIVE_DONE,
                BufferName.EVENTS_INTEREST_DONE,
                BufferName.UNDER_THIS_VEHICLE_CONTROL,
            ):
                initial_value = bool()
            elif bn in (
                BufferName.EGO_VEHICLE_STATE_POSITION,
                BufferName.EGO_VEHICLE_STATE_BOUNDING_BOX,
                BufferName.EGO_VEHICLE_STATE_LANE_POSITION,
            ):
                initial_value = (float(), float(), float())
            elif bn in (
                BufferName.EGO_VEHICLE_STATE_LINEAR_VELOCITY,
                BufferName.EGO_VEHICLE_STATE_ANGULAR_VELOCITY,
                BufferName.EGO_VEHICLE_STATE_LINEAR_ACCELERATION,
                BufferName.EGO_VEHICLE_STATE_ANGULAR_ACCELERATION,
                BufferName.EGO_VEHICLE_STATE_LINEAR_JERK,
                BufferName.EGO_VEHICLE_STATE_ANGULAR_JERK,
            ):
                initial_value = (float(), float())
            elif bn in (
                BufferName.EGO_VEHICLE_STATE_ROAD_ID,
                BufferName.EGO_VEHICLE_STATE_LANE_ID,
            ):
                initial_value = int()

            # Vector of NEIGHBORHOOD_VEHICLE_STATES
            elif bn in (
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_POSITION,
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_BOUNDING_BOX,
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_LANE_POSITION,
            ):
                initial_value = [
                    (float(), float(), float()),
                ] * 20
            elif bn in (
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_HEADING,
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_SPEED,
            ):
                initial_value = [
                    float(),
                ]
            elif bn in (
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_ROAD_ID,
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_LANE_ID,
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_LANE_INDEX,
                BufferName.NEIGHBORHOOD_VEHICLE_STATES_INTEREST,
            ):
                initial_value = [
                    int(),
                ]

            # Vector of waypoints from WAYPOINT_PATHS
            elif bn in (BufferName.WAYPOINT_PATHS_POSITION,):
                initial_value = [(float(), float())]
            elif bn in (
                BufferName.WAYPOINT_PATHS_HEADING,
                BufferName.WAYPOINT_PATHS_LANE_WIDTH,
                BufferName.WAYPOINT_PATHS_SPEED_LIMIT,
                BufferName.WAYPOINT_PATHS_LANE_OFFSET,
            ):
                initial_value = [
                    float(),
                ]
            elif bn in (
                BufferName.WAYPOINT_PATHS_LANE_ID,
                BufferName.WAYPOINT_PATHS_LANE_INDEX,
            ):
                initial_value = [
                    int(),
                ]

            # Vector of waypoints from ROAD_WAYPOINTS
            elif bn in (BufferName.ROAD_WAYPOINTS_POSITIONS,):
                initial_value = [
                    (float(), float()),
                ]
            elif bn in (
                BufferName.ROAD_WAYPOINTS_HEADING,
                BufferName.ROAD_WAYPOINTS_LANE_WIDTH,
                BufferName.ROAD_WAYPOINTS_SPEED_LIMIT,
                BufferName.ROAD_WAYPOINTS_LANE_OFFSET,
            ):
                initial_value = [
                    float(),
                ]
            elif bn in (
                BufferName.ROAD_WAYPOINTS_LANE_ID,
                BufferName.ROAD_WAYPOINTS_LANE_INDEX,
            ):
                initial_value = [
                    int(),
                ]

            # Vector of via data from VIA_DATA
            elif bn in (BufferName.VIA_DATA_NEAR_VIA_POINTS_POSITION,):
                initial_value = [
                    (float(), float()),
                ]
            elif bn in (
                BufferName.VIA_DATA_NEAR_VIA_POINTS_LANE_INDEX,
                BufferName.VIA_DATA_NEAR_VIA_POINTS_ROAD_ID,
                BufferName.VIA_DATA_NEAR_VIA_POINTS_HIT,
            ):
                initial_value = [
                    int(),
                ]
            elif bn in (BufferName.VIA_DATA_NEAR_VIA_POINTS_REQUIRED_SPEED,):
                initial_value = [
                    float(),
                ]

            # Vector of lidar point information from LIDAR_POINT_CLOUD
            elif bn in (
                BufferName.LIDAR_POINT_CLOUD_POINTS,
                BufferName.LIDAR_POINT_CLOUD_ORIGIN,
                BufferName.LIDAR_POINT_CLOUD_DIRECTION,
            ):
                initial_value = [
                    (float(), float(), float()),
                ]
            elif bn in (BufferName.LIDAR_POINT_CLOUD_HITS,):
                initial_value = [
                    int(),
                ]

            # SIGNALS
            elif bn in (
                BufferName.SIGNALS_LIGHT_STATE,
                # BufferName.SIGNALS_CONTROLLED_LANES,
            ):
                initial_value = [
                    int(),
                ]
            elif bn in (
                BufferName.SIGNALS_STOP_POINT,
                BufferName.SIGNALS_LAST_CHANGED,
            ):
                initial_value = [
                    float(),
                ]
            else:
                raise ValueError(f"Buffer `{bn}` is not yet implemented.")

            inputs[bn.value] = initial_value

        surface.setShaderInputs(**inputs)
