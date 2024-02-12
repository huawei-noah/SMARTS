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
from __future__ import annotations

import abc
import importlib.resources as pkg_resources
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Optional, Tuple, Union

import numpy as np

from smarts.core import glsl
from smarts.core.actor import ActorState
from smarts.core.agent_interface import (
    CustomRenderBufferDependency,
    CustomRenderCameraDependency,
    CustomRenderVariableDependency,
    RenderDependencyBase,
)
from smarts.core.coordinates import Dimensions, Pose, RefLinePoint
from smarts.core.lidar import Lidar
from smarts.core.masks import RenderMasks
from smarts.core.observations import (
    CustomRenderData,
    DrivableAreaGridMap,
    GridMapMetadata,
    Observation,
    OcclusionRender,
    OccupancyGridMap,
    RoadWaypoints,
    SignalObservation,
    TopDownRGB,
    ViaPoint,
)
from smarts.core.renderer_base import (
    RendererBase,
    ShaderStepBufferDependency,
    ShaderStepCameraDependency,
    ShaderStepVariableDependency,
)
from smarts.core.road_map import RoadMap, Waypoint
from smarts.core.shader_buffer import BufferID, CameraSensorID
from smarts.core.signals import SignalState
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.utils.core_math import squared_dist
from smarts.core.vehicle_state import neighborhood_vehicles_around_vehicle

if TYPE_CHECKING:
    from smarts.core.actor import ActorState
    from smarts.core.lidar_sensor_params import SensorParams
    from smarts.core.plan import Plan
    from smarts.core.provider import ProviderState
    from smarts.core.simulation_frame import SimulationFrame
    from smarts.core.vehicle_state import VehicleState


def _gen_sensor_name(base_name: str, vehicle_state: VehicleState):
    return _gen_base_sensor_name(base_name, vehicle_state.actor_id)


def _gen_base_sensor_name(base_name: str, actor_id: str):
    return f"{base_name}_{actor_id}"


class Sensor(metaclass=abc.ABCMeta):
    """The sensor base class."""

    def step(self, sim_frame: SimulationFrame, **kwargs):
        """Update sensor state."""

    @abc.abstractmethod
    def teardown(self, **kwargs):
        """Clean up internal resources"""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @property
    def mutable(self) -> bool:
        """If this sensor mutates on call."""
        return True

    @property
    def serializable(self) -> bool:
        """If this sensor can be serialized."""
        return True


class CameraSensor(Sensor):
    """The base for a sensor that renders images."""

    def __init__(
        self,
        vehicle_state: VehicleState,
        renderer: RendererBase,
        name: str,
        mask: int,
        width: int,
        height: int,
        resolution: float,
        build_camera: bool = True,
    ):
        assert renderer
        self._log = logging.getLogger(self.__class__.__name__)
        self._name = name
        self._camera_name = _gen_sensor_name(name, vehicle_state)
        if build_camera:
            renderer.build_offscreen_camera(
                self._camera_name,
                mask,
                width,
                height,
                resolution,
            )
            self._follow_actor(
                vehicle_state, renderer
            )  # ensure we have a correct initial camera position
        self._target_actor = vehicle_state.actor_id
        self._mask = mask
        self._width = width
        self._height = height
        self._resolution = resolution

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, self.__class__)
            and self._target_actor == __value._target_actor
            and self._camera_name == __value._camera_name
            and self._mask == __value._mask
            and self._width == __value._width
            and self._height == __value._height
            and self._resolution == __value._resolution
        )

    def teardown(self, **kwargs):
        renderer: Optional[RendererBase] = kwargs.get("renderer")
        if not renderer:
            return
        camera = renderer.camera_for_id(self._camera_name)
        camera.teardown()

    def step(self, sim_frame: SimulationFrame, **kwargs):
        if not self._target_actor in sim_frame.actor_states_by_id:
            return
        self._follow_actor(
            sim_frame.actor_states_by_id[self._target_actor], kwargs.get("renderer")
        )

    def _follow_actor(self, actor_state: ActorState, renderer: RendererBase):
        if not renderer:
            return
        camera = renderer.camera_for_id(self._camera_name)
        pose = actor_state.get_pose()
        dimensions = actor_state.get_dimensions()
        if pose == None or dimensions == None:
            return
        camera.update(pose, dimensions.height + 10)

    @property
    def camera_name(self) -> str:
        """The name of the camera this sensor is using."""
        return self._camera_name

    @property
    def name(self) -> str:
        """The name of this sensor."""
        return self._name

    @property
    def serializable(self) -> bool:
        return False


class DrivableAreaGridMapSensor(CameraSensor):
    """A sensor that renders drivable area from around its target actor."""

    def __init__(
        self,
        vehicle_state: VehicleState,
        width: int,
        height: int,
        resolution: float,
        renderer: RendererBase,
    ):
        super().__init__(
            vehicle_state,
            renderer,
            CameraSensorID.DRIVABLE_AREA_GRID_MAP.value,
            RenderMasks.DRIVABLE_AREA_HIDE,
            width,
            height,
            resolution,
        )
        self._resolution = resolution

    def __call__(self, renderer: RendererBase) -> DrivableAreaGridMap:
        camera = renderer.camera_for_id(self._camera_name)
        assert camera is not None, "Drivable area grid map has not been initialized"

        ram_image = camera.wait_for_ram_image(img_format="A")
        mem_view = memoryview(ram_image)
        image: np.ndarray = np.frombuffer(mem_view, np.uint8)
        width, height = camera.image_dimensions
        image.shape = (height, width, 1)
        image = np.flipud(image)

        metadata = GridMapMetadata(
            resolution=self._resolution,
            height=image.shape[0],
            width=image.shape[1],
            camera_position=camera.position,
            camera_heading=camera.heading,
        )
        return DrivableAreaGridMap(data=image, metadata=metadata)


class OGMSensor(CameraSensor):
    """A sensor that renders occupancy information from around its target actor."""

    def __init__(
        self,
        vehicle_state: VehicleState,
        width: int,
        height: int,
        resolution: float,
        renderer: RendererBase,
    ):
        super().__init__(
            vehicle_state,
            renderer,
            CameraSensorID.OCCUPANCY_GRID_MAP.value,
            RenderMasks.OCCUPANCY_HIDE,
            width,
            height,
            resolution,
        )

    def __call__(self, renderer: RendererBase) -> OccupancyGridMap:
        base_camera = renderer.camera_for_id(self._camera_name)
        assert base_camera is not None, "OGM has not been initialized"

        ram_image = base_camera.wait_for_ram_image(img_format="A")
        mem_view = memoryview(ram_image)
        grid: np.ndarray = np.frombuffer(mem_view, np.uint8)
        width, height = base_camera.image_dimensions
        grid.shape = (height, width, 1)
        grid = np.flipud(grid)

        metadata = GridMapMetadata(
            resolution=self._resolution,
            height=grid.shape[0],
            width=grid.shape[1],
            camera_position=base_camera.position,
            camera_heading=base_camera.heading,
        )
        return OccupancyGridMap(data=grid, metadata=metadata)


class RGBSensor(CameraSensor):
    """A sensor that renders color values from around its target actor."""

    def __init__(
        self,
        vehicle_state: VehicleState,
        width: int,
        height: int,
        resolution: float,
        renderer: RendererBase,
    ):
        super().__init__(
            vehicle_state,
            renderer,
            CameraSensorID.TOP_DOWN_RGB.value,
            RenderMasks.RGB_HIDE,
            width,
            height,
            resolution,
        )
        self._resolution = resolution

    def __call__(self, renderer: RendererBase) -> TopDownRGB:
        camera = renderer.camera_for_id(self._camera_name)
        assert camera is not None, "RGB has not been initialized"

        ram_image = camera.wait_for_ram_image(img_format="RGB")
        mem_view = memoryview(ram_image)
        image: np.ndarray = np.frombuffer(mem_view, np.uint8)
        width, height = camera.image_dimensions
        image.shape = (height, width, 3)
        image = np.flipud(image)

        metadata = GridMapMetadata(
            resolution=self._resolution,
            height=image.shape[0],
            width=image.shape[1],
            camera_position=camera.position,
            camera_heading=camera.heading,
        )
        return TopDownRGB(data=image, metadata=metadata)


class OcclusionMapSensor(CameraSensor):
    """A sensor that demonstrates only the areas that can be seen by the vehicle."""

    def __init__(
        self,
        vehicle_state: VehicleState,
        width: int,
        height: int,
        resolution: float,
        renderer: RendererBase,
        ogm_sensor: OGMSensor,
        add_surface_noise: bool,
    ):
        self._effect_cameras = []
        super().__init__(
            vehicle_state,
            renderer,
            CameraSensorID.OCCLUSION.value,
            RenderMasks.NONE,
            width,
            height,
            resolution,
            build_camera=False,
        )

        occlusion_camera0 = ogm_sensor.camera_name
        occlusion_camera1 = occlusion_camera0

        if add_surface_noise:
            # generate simplex camera
            with pkg_resources.path(glsl, "simplex_shader.frag") as simplex_shader_path:
                simplex_camera_name = _gen_sensor_name("simplex", vehicle_state)
                renderer.build_shader_step(
                    name=simplex_camera_name,
                    fshader_path=simplex_shader_path,
                    dependencies=(
                        ShaderStepVariableDependency(
                            value=1.0 / resolution,
                            script_variable_name="scale",
                        ),
                        ShaderStepCameraDependency(
                            _gen_sensor_name(
                                CameraSensorID.DRIVABLE_AREA_GRID_MAP.value,
                                vehicle_state,
                            ),
                            "iChannel0",
                        ),
                    ),
                    priority=10,
                    width=width,
                    height=height,
                )
            self._effect_cameras.append(simplex_camera_name)
            occlusion_camera1 = simplex_camera_name

        # feed simplex and ogm to composite
        with pkg_resources.path(glsl, "occlusion_shader.frag") as composite_shader_path:
            composite_camera_name = _gen_sensor_name(
                CameraSensorID.OCCLUSION.value, vehicle_state
            )
            renderer.build_shader_step(
                name=composite_camera_name,
                fshader_path=composite_shader_path,
                dependencies=(
                    ShaderStepCameraDependency(occlusion_camera0, "iChannel0"),
                    ShaderStepCameraDependency(occlusion_camera1, "iChannel1"),
                ),
                priority=30,
                width=width,
                height=height,
            )
        self._effect_cameras.append(composite_camera_name)

    def _follow_actor(self, actor_state: ActorState, renderer: RendererBase):
        if not renderer:
            return
        for effect_name in self._effect_cameras:
            pose = actor_state.get_pose()
            dimensions = actor_state.get_dimensions()
            if pose == None or dimensions == None:
                continue
            camera = renderer.camera_for_id(effect_name)
            camera.update(pose, dimensions.height)

    def teardown(self, **kwargs):
        renderer: Optional[RendererBase] = kwargs.get("renderer")
        if not renderer:
            return
        for effect_name in self._effect_cameras:
            camera = renderer.camera_for_id(effect_name)
            camera.teardown()

    def __call__(self, renderer: RendererBase) -> OcclusionRender:
        effect_camera = renderer.camera_for_id(self._effect_cameras[-1])

        ram_image = effect_camera.wait_for_ram_image("RGB")
        mem_view = memoryview(ram_image)
        grid: np.ndarray = np.frombuffer(mem_view, np.uint8)[::3]
        grid.shape = effect_camera.image_dimensions
        grid = np.flipud(grid)

        metadata = GridMapMetadata(
            resolution=self._resolution,
            height=grid.shape[0],
            width=grid.shape[1],
            camera_position=(math.inf, math.inf, math.inf),
            camera_heading=math.inf,
        )
        return OcclusionRender(data=grid, metadata=metadata)


class CustomRenderSensor(CameraSensor):
    """Defines a configurable image sensor."""

    def __init__(
        self,
        vehicle_state: VehicleState,
        width: int,
        height: int,
        resolution: float,
        renderer: RendererBase,
        fragment_shader_path: str,
        render_dependencies: Tuple[RenderDependencyBase, ...],
        ogm_sensor: Optional[OGMSensor],
        top_down_rgb_sensor: Optional[RGBSensor],
        drivable_area_grid_map_sensor: Optional[DrivableAreaGridMapSensor],
        occlusion_map_sensor: Optional[OcclusionMapSensor],
        name: str,
    ):
        super().__init__(
            vehicle_state,
            renderer,
            name,
            RenderMasks.NONE,
            width,
            height,
            resolution,
            build_camera=False,
        )

        dependencies = []
        named_camera_sensors = (
            (CameraSensorID.OCCUPANCY_GRID_MAP, ogm_sensor),
            (CameraSensorID.TOP_DOWN_RGB, top_down_rgb_sensor),
            (CameraSensorID.DRIVABLE_AREA_GRID_MAP, drivable_area_grid_map_sensor),
            (CameraSensorID.OCCLUSION, occlusion_map_sensor),
        )

        def has_required(dependency_name, required_name, sensor) -> bool:
            if dependency_name == required_name:
                if sensor:
                    return True
                raise UserWarning(
                    f"Custom render depency requires `{d.name}` but the sensor is not attached in the interface."
                )
            return False

        for d in render_dependencies:
            if isinstance(d, CustomRenderCameraDependency):
                for csn, sensor in named_camera_sensors:
                    if has_required(d.camera_dependency_name, csn.value, sensor):
                        break

                camera_id = (
                    _gen_sensor_name(d.camera_dependency_name, vehicle_state)
                    if d.is_self_targetted()
                    else _gen_base_sensor_name(d.camera_dependency_name, d.target_actor)
                )
                dependency = ShaderStepCameraDependency(
                    camera_id,
                    d.variable_name,
                )
            elif isinstance(d, CustomRenderVariableDependency):
                dependency = ShaderStepVariableDependency(
                    value=d.value,
                    script_variable_name=d.variable_name,
                )
            elif isinstance(d, CustomRenderBufferDependency):
                if isinstance(d.buffer_dependency_name, str):
                    buffer_name = BufferID(d.buffer_dependency_name)
                else:
                    buffer_name = d.buffer_dependency_name

                dependency = ShaderStepBufferDependency(
                    buffer_id=buffer_name,
                    script_uniform_name=d.variable_name,
                )
            else:
                raise TypeError(
                    f"Dependency must be a subtype of `{RenderDependencyBase}`"
                )
            dependencies.append(dependency)

        renderer.build_shader_step(
            name=self._camera_name,
            fshader_path=fragment_shader_path,
            dependencies=dependencies,
            priority=40,
            width=width,
            height=height,
        )

    def step(self, sim_frame: SimulationFrame, **kwargs):
        if not self._target_actor in sim_frame.actor_states_by_id:
            return
        actor_state = sim_frame.actor_states_by_id[self._target_actor]
        renderer = kwargs.get("renderer")
        observations: Optional[Dict[str, Observation]] = kwargs.get("observations")

        target = None
        if isinstance(observations, dict):
            for k, o in observations.items():
                o: Observation
                if o.ego_vehicle_state.id == self._target_actor:
                    target = o
            assert isinstance(target, Observation)

        if not renderer:
            return
        renderer: RendererBase
        camera = renderer.camera_for_id(self._camera_name)
        pose = actor_state.get_pose()
        dimensions = actor_state.get_dimensions()
        if not target:
            camera.update(
                pose=pose, height=(dimensions.height + 10) if dimensions else None
            )
        else:
            camera.update(observation=target)

    def teardown(self, **kwargs):
        renderer = kwargs.get("renderer")
        if not renderer:
            return
        renderer: RendererBase
        camera = renderer.camera_for_id(self._camera_name)
        camera.teardown()

    def __call__(self, renderer: RendererBase) -> CustomRenderData:
        effect_camera = renderer.camera_for_id(self._camera_name)

        ram_image = effect_camera.wait_for_ram_image("RGB")
        mem_view = memoryview(ram_image)
        grid: np.ndarray = np.frombuffer(mem_view, np.uint8)
        grid.shape = effect_camera.image_dimensions + (3,)
        grid = np.flipud(grid)

        metadata = GridMapMetadata(
            resolution=self._resolution,
            height=grid.shape[0],
            width=grid.shape[1],
            camera_position=(math.inf, math.inf, math.inf),  # has no position
            camera_heading=math.inf,  # has no heading
        )
        return CustomRenderData(data=grid, metadata=metadata)


class LidarSensor(Sensor):
    """A lidar sensor."""

    def __init__(
        self,
        vehicle_state: VehicleState,
        sensor_params: Optional[SensorParams] = None,
        lidar_offset=(0, 0, 1),
    ):
        self._lidar_offset = np.array(lidar_offset)

        self._lidar = Lidar(
            vehicle_state.pose.position + self._lidar_offset,
            sensor_params,
        )

    def follow_vehicle(self, vehicle_state: VehicleState):
        """Update the sensor to target the given vehicle."""
        self._lidar.origin = vehicle_state.pose.position + self._lidar_offset

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, LidarSensor) and (
            (self._lidar_offset == __value._lidar_offset).all()
            and (self._lidar.origin == __value._lidar.origin).all()
        )

    def __call__(self, bullet_client):
        return self._lidar.compute_point_cloud(bullet_client)

    def teardown(self, **kwargs):
        pass

    @property
    def serializable(self) -> bool:
        return False


@dataclass(frozen=True)
class _DrivenPathSensorEntry:
    timestamp: float
    position: Tuple[float, float]


class DrivenPathSensor(Sensor):
    """Tracks the driven path as a series of positions (regardless if the vehicle is
    following the route or not). For performance reasons it only keeps the last
    N=max_path_length path segments.
    """

    def __init__(self, max_path_length: int = 500):
        self._driven_path = deque(maxlen=max_path_length)

    def track_latest_driven_path(self, elapsed_sim_time, vehicle_state):
        """Records the current location of the tracked vehicle."""
        position = vehicle_state.pose.position[:2]
        self._driven_path.append(
            _DrivenPathSensorEntry(timestamp=elapsed_sim_time, position=tuple(position))
        )

    def __call__(self, count=sys.maxsize):
        return [x.position for x in self._driven_path][-count:]

    def teardown(self, **kwargs):
        pass

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, DrivenPathSensor)
            and self._driven_path == __value._driven_path
        )

    def distance_travelled(
        self,
        elapsed_sim_time,
        last_n_seconds: Optional[float] = None,
        last_n_steps: Optional[int] = None,
    ):
        """Find the amount of distance travelled over the last # of seconds XOR steps"""
        if last_n_seconds is None and last_n_steps is None:
            raise ValueError("Either last N seconds or last N steps must be provided")

        if last_n_steps is not None:
            n = last_n_steps + 1  # to factor in the current step we're on
            filtered_pos = [x.position for x in self._driven_path][-n:]
        else:  # last_n_seconds
            threshold = elapsed_sim_time - last_n_seconds
            filtered_pos = [
                x.position for x in self._driven_path if x.timestamp >= threshold
            ]

        xs = np.array([p[0] for p in filtered_pos])
        ys = np.array([p[1] for p in filtered_pos])
        dist_array = (xs[:-1] - xs[1:]) ** 2 + (ys[:-1] - ys[1:]) ** 2
        return np.sum(np.sqrt(dist_array))


class TripMeterSensor(Sensor):
    """Tracks distance travelled along the route (in meters). Meters driven while
    off-route are not counted as part of the total.
    """

    def __init__(self):
        self._wps_for_distance: List[Waypoint] = []
        self._dist_travelled = 0.0
        self._last_dist_travelled = 0.0
        self._last_actor_position = None

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, TripMeterSensor) and (
            self._wps_for_distance == __value._wps_for_distance
            and self._dist_travelled == __value._dist_travelled
            and self._last_dist_travelled == __value._last_dist_travelled
        )

    def update_distance_wps_record(
        self,
        waypoint_paths: List[List[Waypoint]],
        vehicle_state: VehicleState,
        plan: Plan,
        road_map: RoadMap,
    ):
        """Append a waypoint to the history if it is not already counted."""
        # Distance calculation. Intention is the shortest trip travelled at the lane
        # level the agent has travelled. This is to prevent lateral movement from
        # increasing the total distance travelled.
        self._last_dist_travelled = self._dist_travelled

        new_wp = waypoint_paths[0][0]
        wp_road = road_map.lane_by_id(new_wp.lane_id).road.road_id

        should_count_wp = (
            plan.mission == None
            # if we do not have a fixed route, we count all waypoints we accumulate
            or not plan.mission.requires_route
            # if we have a route to follow, only count wps on route
            or wp_road in [road.road_id for road in plan.route.roads]
        )

        if not self._wps_for_distance:
            self._last_actor_position = vehicle_state.pose.position
            if should_count_wp:
                self._wps_for_distance.append(new_wp)
            return  # sensor does not have enough history
        most_recent_wp = self._wps_for_distance[-1]

        # TODO: Instead of storing a waypoint every 0.5m just find the next one immediately
        threshold_for_counting_wp = 0.5  # meters from last tracked waypoint
        if (
            np.linalg.norm(new_wp.pos - most_recent_wp.pos) > threshold_for_counting_wp
            and should_count_wp
        ):
            self._wps_for_distance.append(new_wp)
        additional_distance = TripMeterSensor._compute_additional_dist_travelled(
            most_recent_wp,
            new_wp,
            vehicle_state.pose.position,
            self._last_actor_position,
        )
        self._dist_travelled += additional_distance
        self._last_actor_position = vehicle_state.pose.position

    @staticmethod
    def _compute_additional_dist_travelled(
        recent_wp: Waypoint,
        new_waypoint: Waypoint,
        vehicle_position: np.ndarray,
        last_vehicle_pos: np.ndarray,
    ):
        # old waypoint minus current ahead waypoint
        wp_disp_vec = new_waypoint.pos - recent_wp.pos
        # make unit vector
        wp_unit_vec = wp_disp_vec / (np.linalg.norm(wp_disp_vec) or 1)
        # vehicle position minus last vehicle position
        position_disp_vec = vehicle_position[:2] - last_vehicle_pos[:2]
        # distance of vehicle between last and current wp
        distance = np.dot(position_disp_vec, wp_unit_vec)
        return distance

    def __call__(self, increment: bool = False):
        if increment:
            return self._dist_travelled - self._last_dist_travelled

        return self._dist_travelled

    def teardown(self, **kwargs):
        pass


class NeighborhoodVehiclesSensor(Sensor):
    """Detects other vehicles around the sensor equipped vehicle."""

    def __init__(self, radius: Optional[float] = None):
        self._radius = radius

    @property
    def radius(self) -> float:
        """Radius to check for nearby vehicles."""
        return self._radius

    def __call__(
        self, vehicle_state: VehicleState, vehicle_states: Collection[VehicleState]
    ) -> List[VehicleState]:
        return neighborhood_vehicles_around_vehicle(
            vehicle_state, vehicle_states, radius=self._radius
        )

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, NeighborhoodVehiclesSensor)
            and self._radius == __value._radius
        )

    def teardown(self, **kwargs):
        pass

    @property
    def mutable(self) -> bool:
        return False


class WaypointsSensor(Sensor):
    """Detects waypoints leading forward along the vehicle plan."""

    def __init__(self, lookahead: int = 32):
        self._lookahead = lookahead

    def __call__(self, vehicle_state: VehicleState, plan: Plan, road_map: RoadMap):
        return road_map.waypoint_paths(
            pose=vehicle_state.pose,
            lookahead=self._lookahead,
            route=plan.route,
        )

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, WaypointsSensor)
            and self._lookahead == __value._lookahead
        )

    def teardown(self, **kwargs):
        pass

    @property
    def mutable(self) -> bool:
        return False


class RoadWaypointsSensor(Sensor):
    """Detects waypoints from all paths nearby the vehicle."""

    def __init__(self, horizon: int = 32):
        self._horizon = horizon

    def __call__(
        self, vehicle_state: VehicleState, plan: Plan, road_map: RoadMap
    ) -> RoadWaypoints:
        veh_pt = vehicle_state.pose.point
        lane = road_map.nearest_lane(veh_pt)
        if not lane:
            return RoadWaypoints(lanes={})
        road = lane.road
        lane_paths = {}
        for croad in (
            [road] + road.parallel_roads + road.oncoming_roads_at_point(veh_pt)
        ):
            for lane in croad.lanes:
                lane_paths[lane.lane_id] = self._paths_for_lane(
                    lane, vehicle_state, plan
                )

        return RoadWaypoints(lanes=lane_paths)

    def _paths_for_lane(
        self,
        lane: RoadMap.Lane,
        vehicle_state: VehicleState,
        plan: Plan,
        overflow_offset: Optional[float] = None,
    ):
        """Gets waypoint paths along the given lane."""
        # XXX: the following assumes waypoint spacing is 1m
        if overflow_offset is None:
            offset = lane.offset_along_lane(vehicle_state.pose.point)
            start_offset = offset - self._horizon
        else:
            start_offset = lane.length + overflow_offset

        incoming_lanes = lane.incoming_lanes
        paths = []
        if start_offset < 0 and len(incoming_lanes) > 0:
            for lane in incoming_lanes:
                paths += self._paths_for_lane(lane, vehicle_state, plan, start_offset)

        start_offset = max(0, start_offset)
        wp_start = lane.from_lane_coord(RefLinePoint(start_offset))
        adj_pose = Pose.from_center(wp_start, vehicle_state.pose.heading)
        wps_to_lookahead = self._horizon * 2
        paths += lane.waypoint_paths_for_pose(
            pose=adj_pose,
            lookahead=wps_to_lookahead,
            route=plan.route,
        )
        return paths

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, RoadWaypoints) and self._horizon == __value._horizon

    def teardown(self, **kwargs):
        pass

    @property
    def mutable(self) -> bool:
        return False


class AccelerometerSensor(Sensor):
    """Tracks motion changes within the vehicle equipped with this sensor."""

    def __init__(self):
        self.linear_velocities = deque(maxlen=3)
        self.angular_velocities = deque(maxlen=3)

    def __call__(self, linear_velocity, angular_velocity, dt: float):
        if linear_velocity is not None:
            self.linear_velocities.append(linear_velocity)
        if angular_velocity is not None:
            self.angular_velocities.append(angular_velocity)

        linear_acc = np.array((0.0, 0.0, 0.0))
        angular_acc = np.array((0.0, 0.0, 0.0))
        linear_jerk = np.array((0.0, 0.0, 0.0))
        angular_jerk = np.array((0.0, 0.0, 0.0))

        if not dt:
            return (linear_acc, angular_acc, linear_jerk, angular_jerk)

        if len(self.linear_velocities) >= 2:
            linear_acc = (self.linear_velocities[-1] - self.linear_velocities[-2]) / dt
            if len(self.linear_velocities) >= 3:
                last_linear_acc = (
                    self.linear_velocities[-2] - self.linear_velocities[-3]
                ) / dt
                linear_jerk = linear_acc - last_linear_acc
        if len(self.angular_velocities) >= 2:
            angular_acc = (
                self.angular_velocities[-1] - self.angular_velocities[-2]
            ) / dt
            if len(self.angular_velocities) >= 3:
                last_angular_acc = (
                    self.angular_velocities[-2] - self.angular_velocities[-3]
                ) / dt
                angular_jerk = angular_acc - last_angular_acc

        return (linear_acc, angular_acc, linear_jerk, angular_jerk)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, AccelerometerSensor) and (
            [
                (a == b).all()
                for a, b in zip(self.linear_velocities, __value.linear_velocities)
            ]
            and [
                (a == b).all()
                for a, b in zip(self.angular_velocities, __value.angular_velocities)
            ]
        )

    def teardown(self, **kwargs):
        pass


class LanePositionSensor(Sensor):
    """Tracks lane-relative RefLine (Frenet) coordinates."""

    def __init__(self):
        pass

    def __call__(self, lane: RoadMap.Lane, vehicle_state: VehicleState):
        return lane.to_lane_coord(vehicle_state.pose.point)

    def __eq__(self, __value: object) -> bool:
        return True

    def teardown(self, **kwargs):
        pass

    @property
    def mutable(self) -> bool:
        return False


class ViaSensor(Sensor):
    """Tracks collection of ViaPoint collectibles"""

    def __init__(self, lane_acquisition_range, speed_accuracy):
        self._consumed_via_points = set()
        self._acquisition_range = lane_acquisition_range
        self._speed_accuracy = speed_accuracy

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, ViaSensor) and (
            self._consumed_via_points == __value._consumed_via_points
            and self._acquisition_range == __value._acquisition_range
            and self._speed_accuracy == __value._speed_accuracy
        )

    def __call__(self, vehicle_state: VehicleState, plan: Plan, road_map: RoadMap):
        if plan.mission is None:
            return ()

        near_points: List[Tuple[float, ViaPoint]] = []
        vehicle_position = vehicle_state.pose.position[:2]

        @lru_cache()
        def closest_point_on_lane(position, lane_id, road_map):
            lane = road_map.lane_by_id(lane_id)
            return lane.center_at_point(position)

        for via in plan.mission.via:
            closest_position_on_lane = closest_point_on_lane(
                tuple(vehicle_position), via.lane_id, road_map
            )
            closest_position_on_lane = closest_position_on_lane[:2]

            dist_from_lane_sq = squared_dist(vehicle_position, closest_position_on_lane)
            if dist_from_lane_sq > self._acquisition_range**2:
                continue

            dist_from_point_sq = squared_dist(vehicle_position, via.position)
            hit = (
                dist_from_point_sq <= via.hit_distance**2
                and via not in self._consumed_via_points
                and np.isclose(
                    vehicle_state.speed, via.required_speed, atol=self._speed_accuracy
                )
            )

            point = ViaPoint(
                tuple(via.position),
                lane_index=via.lane_index,
                road_id=via.road_id,
                required_speed=via.required_speed,
                hit=hit,
            )

            near_points.append((dist_from_point_sq, point))
            if hit:
                self._consumed_via_points.add(via)

        near_points.sort(key=lambda dist, _: dist)
        return tuple(p for _, p in near_points)

    def teardown(self, **kwargs):
        pass


class SignalsSensor(Sensor):
    """Reports state of traffic signals (lights) in the lanes ahead of vehicle."""

    def __init__(self, lookahead: float):
        self._lookahead = lookahead

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, SignalsSensor) and self._lookahead == __value._lookahead
        )

    @staticmethod
    def _is_signal_type(feature: RoadMap.Feature) -> bool:
        # XXX:  eventually if we add other types of dynamic features, we'll need to update this.
        return (
            feature.type == RoadMap.FeatureType.FIXED_LOC_SIGNAL or feature.is_dynamic
        )

    def __call__(
        self,
        lane: Optional[RoadMap.Lane],
        lane_pos: RefLinePoint,
        state: VehicleState,
        plan: Plan,
        provider_state: ProviderState,
    ) -> Tuple[SignalObservation, ...]:
        if not lane:
            return ()
        result = []
        upcoming_signals = []
        for feat in lane.features:
            if not self._is_signal_type(feat):
                continue
            for pt in feat.geometry:
                # TAI: we might want to use the position of our back bumper
                # instead of centroid to allow agents to have some (even more)
                # imprecision in their handling of stopping at signals.
                if lane.offset_along_lane(pt) >= lane_pos.s:
                    upcoming_signals.append(feat)
                    break
        lookahead = self._lookahead - lane.length + lane_pos.s
        self._find_signals_ahead(lane, lookahead, plan.route, upcoming_signals)

        for signal in upcoming_signals:
            for actor_state in provider_state.actors:
                if actor_state.actor_id == signal.feature_id:
                    signal_state = actor_state
                    break
            else:
                logger = logging.getLogger(self.__class__.__name__)
                logger.warning(
                    "could not find signal state corresponding with feature_id=%s}",
                    signal.feature_id,
                )
                continue
            assert isinstance(signal_state, SignalState)
            controlled_lanes = None
            if signal_state.controlled_lanes:
                controlled_lanes = signal_state.controlled_lanes
            result.append(
                SignalObservation(
                    state=signal_state.state,
                    stop_point=signal_state.stopping_pos,
                    controlled_lanes=tuple(controlled_lanes),
                    last_changed=signal_state.last_changed,
                )
            )

        return tuple(result)

    def _find_signals_ahead(
        self,
        lane: RoadMap.Lane,
        lookahead: float,
        route: Optional[RoadMap.Route],
        upcoming_signals: List[RoadMap.Feature],
    ):
        if lookahead <= 0:
            return
        for ogl in lane.outgoing_lanes:
            if route and route.road_length > 0 and ogl.road not in route.roads:
                continue
            upcoming_signals += [
                feat for feat in ogl.features if self._is_signal_type(feat)
            ]
            self._find_signals_ahead(
                ogl, lookahead - lane.length, route, upcoming_signals
            )

    def teardown(self, **kwargs):
        pass
