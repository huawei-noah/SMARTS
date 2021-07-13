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


from __future__ import (  # to allow for typing to refer to class being defined (Renderer)
    annotations,
)

import importlib.resources as pkg_resources
import logging
import os
from threading import Lock
from typing import NamedTuple

import gltf
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    FrameBufferProperties,
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


class _ShowBaseInstance(ShowBase):
    """ Wraps a singleton instance of ShowBase from Panda3D. """

    def __new__(cls):
        # Singleton pattern:  ensure only 1 ShowBase instance
        if "__it__" not in cls.__dict__:
            # disable vsync otherwise we are limited to refresh-rate of screen
            loadPrcFileData("", "sync-video false")
            loadPrcFileData("", "model-path %s" % os.getcwd())
            # TODO: the following speeds up rendering a bit... might consider it.
            # loadPrcFileData("", "model-cache-dir %s/.panda3d_cache" % os.getcwd())
            loadPrcFileData("", "audio-library-name null")
            loadPrcFileData("", "gl-version 3 3")
            loadPrcFileData("", "notify-level error")
            loadPrcFileData("", "print-pipe-types false")
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
        self._render_lock = Lock()
        try:
            # There can be only 1 ShowBase instance at a time.
            super().__init__(windowType="offscreen")

            gltf.patch_loader(self.loader)
            self.setBackgroundColor(0, 0, 0, 1)

            # Displayed framerate is misleading since we are not using a realtime clock
            self.setFrameRateMeter(False)

        except Exception as e:
            # Known reasons for this failing:
            raise Exception(
                f"Error in initializing framework for opening graphical display and creating scene graph. "
                "A typical reason is display not found. Try running with different configurations of "
                "`export DISPLAY=` using `:0`, `:1`... . If this does not work please consult "
                "the documentation.\nException was: {e}"
            ) from e

    def destroy(self):
        super().destroy()
        self.__class__.__it__ = None

    def __del__(self):
        self.destroy()

    def setup_sim_root(self, simid: str):
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
            root_np.setShader(unlit_shader)
        return root_np

    def render_node(self, sim_root: NodePath):
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
    def __init__(self, simid: str):
        self._log = logging.getLogger(self.__class__.__name__)
        self._is_setup = False
        self._simid = simid
        self._root_np = None
        self._vehicles_np = None
        self._road_network_np = None
        self._vehicle_nodes = {}
        # Note: Each instance of the SMARTS simulation will have its own Renderer,
        # but all Renderer objects share the same ShowBaseInstance.
        self._showbase_instance = _ShowBaseInstance()

    @property
    def id(self):
        return self._simid

    def setup(self, scenario: Scenario):
        self._root_np = self._showbase_instance.setup_sim_root(self._simid)
        self._vehicles_np = self._root_np.attachNewNode("vehicles")

        map_path = scenario.map_glb_filepath
        if self._road_network_np:
            self._log.debug(
                "road_network={} already exists. Removing and adding a new "
                "one from glb_path={}".format(self._road_network_np, map_path)
            )
        map_np = self._showbase_instance.loader.loadModel(map_path, noCache=True)
        np = self._root_np.attachNewNode("road_network")
        map_np.reparent_to(np)
        np.hide(RenderMasks.OCCUPANCY_HIDE)
        np.setColor(SceneColors.Road.value)
        self._road_network_np = np

        self._is_setup = True

    def render(self):
        assert self._is_setup
        self._showbase_instance.render_node(self._root_np)

    def step(self):
        """ provided for non-SMARTS uses; normally not used by SMARTS. """
        self._showbase_instance.taskMgr.step()

    def teardown(self):
        if self._root_np is not None:
            self._root_np.clearLight()
            self._root_np.removeNode()
            self._root_np = None
        self._vehicles_np = None
        self._road_network_np = None
        self._is_setup = False

    def destroy(self):
        self.teardown()
        self._showbase_instance = None

    def __del__(self):
        self.destroy()

    def create_vehicle_node(self, glb_model: str, vid: str, color, pose: Pose):
        with pkg_resources.path(models, glb_model) as path:
            node_path = self._showbase_instance.loader.loadModel(str(path.absolute()))
        node_path.setName("vehicle-%s" % vid)
        node_path.setColor(color)
        pos, heading = pose.as_panda3d()
        node_path.setPosHpr(*pos, heading, 0, 0)
        node_path.hide(RenderMasks.DRIVABLE_AREA_HIDE)
        self._vehicle_nodes[vid] = node_path

    def begin_rendering_vehicle(self, vid: str, is_agent: bool):
        """ adds the vehicle node to the scene graph """
        vehicle_path = self._vehicle_nodes.get(vid, None)
        if not vehicle_path:
            self._log.warning(f"Renderer ignoring invalid vehicle id: {vid}")
            return
        # TAI: consider reparenting hijacked vehicles too?
        vehicle_path.reparentTo(self._vehicles_np if is_agent else self._root_np)

    def update_vehicle_node(self, vid: str, pose: Pose):
        vehicle_path = self._vehicle_nodes.get(vid, None)
        if not vehicle_path:
            self._log.warning(f"Renderer ignoring invalid vehicle id: {vid}")
            return
        pos, heading = pose.as_panda3d()
        vehicle_path.setPosHpr(*pos, heading, 0, 0)

    def remove_vehicle_node(self, vid: str):
        vehicle_path = self._vehicle_nodes.get(vid, None)
        if not vehicle_path:
            self._log.warning(f"Renderer ignoring invalid vehicle id: {vid}")
            return
        vehicle_path.removeNode()

    class OffscreenCamera(NamedTuple):
        camera_np: NodePath
        buffer: GraphicsOutput
        tex: Texture
        renderer: Renderer

        def wait_for_ram_image(self, img_format: str, retries=100):
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
                self.renderer._log.debug(
                    f"No image available (attempt {i}/{retries}), forcing a render"
                )
                region = self.buffer.getDisplayRegion(0)
                region.window.engine.renderFrame()

            assert self.tex.mightHaveRamImage()
            ram_image = self.tex.getRamImageAs(img_format)
            assert ram_image is not None
            return ram_image

        def update(self, pose: Pose, height: float):
            pos, heading = pose.as_panda3d()
            self.camera_np.setPos(pos[0], pos[1], height)
            self.camera_np.lookAt(*pos)
            self.camera_np.setH(heading)

        def teardown(self):
            self.camera_np.removeNode()
            region = self.buffer.getDisplayRegion(0)
            region.window.clearRenderTextures()
            self.buffer.removeAllDisplayRegions()
            self.renderer._showbase_instance.graphicsEngine.removeWindow(self.buffer)

    def build_offscreen_camera(
        self,
        name: str,
        mask: int,
        width: int,
        height: int,
        resolution: float,
    ):
        # setup buffer
        win_props = WindowProperties.size(width, height)
        fb_props = FrameBufferProperties()
        fb_props.setRgbColor(True)
        fb_props.setRgbaBits(8, 8, 8, 1)
        # XXX: Though we don't need the depth buffer returned, setting this to 0
        #      causes undefined behaviour where the ordering of meshes is random.
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

        # mask is set to make undesireable objects invisible to this camera
        camera_np.node().setCameraMask(camera_np.node().getCameraMask() & mask)

        return Renderer.OffscreenCamera(camera_np, buffer, tex, self)
