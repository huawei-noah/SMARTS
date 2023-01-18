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
import os
from enum import IntEnum
from pathlib import Path
from threading import Lock
from typing import NamedTuple

import gltf
from direct.showbase.ShowBase import ShowBase

# pytype: disable=import-error
from panda3d.core import (
    FrameBufferProperties,
    Geom,
    GeomLinestrips,
    GeomNode,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexReader,
    GeomVertexWriter,
    GraphicsOutput,
    GraphicsPipe,
    NodePath,
    OrthographicLens,
    Shader,
    Texture,
    WindowProperties,
    loadPrcFileData,
)

from . import glsl, models
from .colors import SceneColors
from .coordinates import Pose
from .masks import RenderMasks
from .scenario import Scenario

# pytype: enable=import-error


class DEBUG_MODE(IntEnum):
    """The rendering debug information level."""

    SPAM = 1
    DEBUG = 2
    INFO = 3
    WARNING = 4
    ERROR = 5


class _ShowBaseInstance(ShowBase):
    """Wraps a singleton instance of ShowBase from Panda3D."""

    _debug_mode: DEBUG_MODE = DEBUG_MODE.WARNING
    _rendering_backend: str = "p3headlessgl"

    def __new__(cls):
        # Singleton pattern:  ensure only 1 ShowBase instance
        if "__it__" not in cls.__dict__:
            if cls._debug_mode <= DEBUG_MODE.INFO:
                loadPrcFileData("", "gl-debug #t")
            loadPrcFileData(
                "",
                f"load-display {cls._rendering_backend}",
            )
            loadPrcFileData("", "aux-display p3headlessgl")
            loadPrcFileData("", "aux-display pandagl")
            loadPrcFileData("", "aux-display pandadx9")
            loadPrcFileData("", "aux-display pandagles")
            loadPrcFileData("", "aux-display pandagles2")
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

    def destroy(self):
        """Destroy this renderer and clean up all remaining resources."""
        super().destroy()
        self.__class__.__it__ = None

    def __del__(self):
        self.destroy()

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
            for np in self.render.children:
                if np != sim_root and not np.isHidden():
                    np.hide()
                    hidden.append(np)
            self.taskMgr.mgr.poll()
            for np in hidden:
                np.show()


class Renderer:
    """The utility used to render simulation geometry."""

    def __init__(self, simid: str, debug_mode: DEBUG_MODE = DEBUG_MODE.ERROR):
        self._log: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._is_setup = False
        self._simid = simid
        self._root_np = None
        self._vehicles_np = None
        self._road_map_np = None
        self._dashed_lines_np = None
        self._vehicle_nodes = {}
        _ShowBaseInstance.set_rendering_verbosity(debug_mode=debug_mode)
        # Note: Each instance of the SMARTS simulation will have its own Renderer,
        # but all Renderer objects share the same ShowBaseInstance.
        self._showbase_instance: _ShowBaseInstance = _ShowBaseInstance()

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
        format = GeomVertexFormat.getV3()
        vdata = GeomVertexData(name, format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")

        prim = GeomLinestrips(Geom.UHStatic)
        for pts in lines:
            for x, y, z in pts:
                vertex.addData3(x, y, z)
            prim.add_next_vertices(len(pts))
            assert prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        np = GeomNode(name)
        np.addGeom(geom)
        return np

    def setup(self, scenario: Scenario):
        """Initialize this renderer."""
        self._root_np = self._showbase_instance.setup_sim_root(self._simid)
        self._vehicles_np = self._root_np.attachNewNode("vehicles")

        map_path = scenario.map_glb_filepath
        map_dir = Path(map_path).parent

        # Load map
        if self._road_map_np:
            self._log.debug(
                "road_map={} already exists. Removing and adding a new "
                "one from glb_path={}".format(self._road_map_np, map_path)
            )
        map_np = self._showbase_instance.loader.loadModel(map_path, noCache=True)
        np = self._root_np.attachNewNode("road_map")
        map_np.reparent_to(np)
        np.hide(RenderMasks.OCCUPANCY_HIDE)
        np.setColor(SceneColors.Road.value)
        self._road_map_np = np

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
                    "Resolution", self._showbase_instance.getSize()
                )
            self._dashed_lines_np = dashed_lines_np

        self._is_setup = True

    def render(self):
        """Render the scene graph of the simulation."""
        assert self._is_setup
        self._showbase_instance.render_node(self._root_np)

    def step(self):
        """provided for non-SMARTS uses; normally not used by SMARTS."""
        self._showbase_instance.taskMgr.step()

    def teardown(self):
        """Clean up internal resources."""
        if self._root_np is not None:
            self._root_np.clearLight()
            self._root_np.removeNode()
            self._root_np = None
        self._vehicles_np = None
        self._road_map_np = None
        self._dashed_lines_np = None
        self._is_setup = False

    def destroy(self):
        """Destroy the renderer. Cleans up all remaining renderer resources."""
        self.teardown()
        self._showbase_instance = None

    def __del__(self):
        self.destroy()

    def create_vehicle_node(self, glb_model: str, vid: str, color, pose: Pose):
        """Create a vehicle node."""
        with pkg_resources.path(models, glb_model) as path:
            node_path = self._showbase_instance.loader.loadModel(str(path.absolute()))
        node_path.setName("vehicle-%s" % vid)
        node_path.setColor(color)
        pos, heading = pose.as_panda3d()
        node_path.setPosHpr(*pos, heading, 0, 0)
        node_path.hide(RenderMasks.DRIVABLE_AREA_HIDE)
        self._vehicle_nodes[vid] = node_path

    def begin_rendering_vehicle(self, vid: str, is_agent: bool):
        """Add the vehicle node to the scene graph"""
        vehicle_path = self._vehicle_nodes.get(vid, None)
        if not vehicle_path:
            self._log.warning(f"Renderer ignoring invalid vehicle id: {vid}")
            return
        # TAI: consider reparenting hijacked vehicles too?
        vehicle_path.reparentTo(self._vehicles_np if is_agent else self._root_np)

    def update_vehicle_node(self, vid: str, pose: Pose):
        """Move the specified vehicle node."""
        vehicle_path = self._vehicle_nodes.get(vid, None)
        if not vehicle_path:
            self._log.warning(f"Renderer ignoring invalid vehicle id: {vid}")
            return
        pos, heading = pose.as_panda3d()
        vehicle_path.setPosHpr(*pos, heading, 0, 0)

    def remove_vehicle_node(self, vid: str):
        """Remove a vehicle node"""
        vehicle_path = self._vehicle_nodes.get(vid, None)
        if not vehicle_path:
            self._log.warning(f"Renderer ignoring invalid vehicle id: {vid}")
            return
        vehicle_path.removeNode()

    class OffscreenCamera(NamedTuple):
        """A camera used for rendering images to a graphics buffer."""

        camera_np: NodePath
        buffer: GraphicsOutput
        tex: Texture
        renderer: Renderer

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
                self.renderer.log.debug(
                    f"No image available (attempt {i}/{retries}), forcing a render"
                )
                region = self.buffer.getDisplayRegion(0)
                region.window.engine.renderFrame()

            assert self.tex.mightHaveRamImage()
            ram_image = self.tex.getRamImageAs(img_format)
            assert ram_image is not None
            return ram_image

        def update(self, pose: Pose, height: float):
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

        def teardown(self):
            """Clean up internal resources."""
            self.camera_np.removeNode()
            region = self.buffer.getDisplayRegion(0)
            region.window.clearRenderTextures()
            self.buffer.removeAllDisplayRegions()
            self.renderer.remove_buffer(self.buffer)

    def build_offscreen_camera(
        self,
        name: str,
        mask: int,
        width: int,
        height: int,
        resolution: float,
    ) -> Renderer.OffscreenCamera:
        """Generates a new offscreen camera."""
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
                "Resolution", (buffer.size.x, buffer.size.y)
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

        return Renderer.OffscreenCamera(camera_np, buffer, tex, self)
