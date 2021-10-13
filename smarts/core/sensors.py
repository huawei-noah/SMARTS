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
import logging
import time
from collections import deque, namedtuple
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, NamedTuple, Set, Tuple

import numpy as np

from smarts.core.agent_interface import AgentsAliveDoneCriteria
from smarts.core.plan import Plan
from smarts.core.road_map import Waypoint
from smarts.core.utils.math import squared_dist, vec_2d, yaw_from_quaternion

from .coordinates import Dimensions, Heading, Point, Pose, RefLinePoint
from .events import Events
from .lidar import Lidar
from .lidar_sensor_params import SensorParams
from .masks import RenderMasks
from .plan import Mission, Via

logger = logging.getLogger(__name__)


class VehicleObservation(NamedTuple):
    id: str
    position: Tuple[float, float, float]
    bounding_box: Dimensions
    heading: Heading
    speed: float
    road_id: str
    lane_id: str
    lane_index: int


class EgoVehicleObservation(NamedTuple):
    id: str
    position: Tuple[float, float, float]
    bounding_box: Dimensions
    heading: Heading
    speed: float
    steering: float
    yaw_rate: float
    road_id: str
    lane_id: str
    lane_index: int
    mission: Mission
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration: np.ndarray
    linear_jerk: np.ndarray
    angular_jerk: np.ndarray


class RoadWaypoints(NamedTuple):
    lanes: Dict[str, List[List[Waypoint]]]
    route_waypoints: List[List[Waypoint]]


class GridMapMetadata(NamedTuple):
    # time at which the map was loaded
    created_at: int
    # map resolution in world-space-distance/cell
    resolution: float
    # map width in # of cells
    width: int
    # map height in # of cells
    height: int
    # camera position when project onto the map
    camera_pos: Tuple[float, float, float]
    # camera rotation angle along z-axis when project onto the map
    camera_heading_in_degrees: float


class TopDownRGB(NamedTuple):
    metadata: GridMapMetadata
    data: np.ndarray


class OccupancyGridMap(NamedTuple):
    metadata: GridMapMetadata
    data: np.ndarray


class DrivableAreaGridMap(NamedTuple):
    metadata: GridMapMetadata
    data: np.ndarray


@dataclass
class ViaPoint:
    position: Tuple[float, float]
    lane_index: float
    road_id: str
    required_speed: float


@dataclass(frozen=True)
class Vias:
    near_via_points: List[ViaPoint]
    """Ordered list of nearby points that have not been hit"""
    hit_via_points: List[ViaPoint]
    """List of points that were hit in the previous step"""


@dataclass
class Observation:
    # dt is the amount of sim_time the last step took .
    # step_count is the number of steps take by SMARTS so far.
    # elapsed_sim_time is the amout of simulation time that's passed so far.
    # note: to get the average step_time, elapsed_sim_time can be divided by step_count
    dt: float
    step_count: int
    elapsed_sim_time: float
    events: Events
    ego_vehicle_state: EgoVehicleObservation
    neighborhood_vehicle_states: List[VehicleObservation]
    waypoint_paths: List[List[Waypoint]]
    distance_travelled: float

    # TODO: Convert to `namedtuple` or only return point cloud
    # [points], [hits], [(ray_origin, ray_directino)]
    lidar_point_cloud: Tuple[
        List[np.ndarray], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]
    ]
    drivable_area_grid_map: DrivableAreaGridMap
    occupancy_grid_map: OccupancyGridMap
    top_down_rgb: TopDownRGB
    road_waypoints: RoadWaypoints = None
    via_data: Vias = None


@dataclass
class Collision:
    collidee_id: str


class Sensors:
    _log = logging.getLogger("Sensors")

    @staticmethod
    def observe_batch(sim, agent_id, sensor_states, vehicles):
        """Operates on a batch of vehicles for a single agent."""
        # TODO: Replace this with a more efficient implementation that _actually_
        #       does batching
        assert sensor_states.keys() == vehicles.keys()

        observations, dones = {}, {}
        for vehicle_id, vehicle in vehicles.items():
            sensor_state = sensor_states[vehicle_id]
            observations[vehicle_id], dones[vehicle_id] = Sensors.observe(
                sim, agent_id, sensor_state, vehicle
            )

        return observations, dones

    @staticmethod
    def observe(sim, agent_id, sensor_state, vehicle):
        neighborhood_vehicles = None
        if vehicle.subscribed_to_neighborhood_vehicles_sensor:
            neighborhood_vehicles = []
            for nv in vehicle.neighborhood_vehicles_sensor():
                nv_lane = sim.road_map.nearest_lane(
                    nv.pose.point, radius=vehicle.length
                )
                if nv_lane:
                    nv_road_id = nv_lane.road.road_id
                    nv_lane_id = nv_lane.lane_id
                    nv_lane_index = nv_lane.index
                else:
                    nv_road_id = None
                    nv_lane_id = None
                    nv_lane_index = None
                neighborhood_vehicles.append(
                    VehicleObservation(
                        id=nv.vehicle_id,
                        position=nv.pose.position,
                        bounding_box=nv.dimensions,
                        heading=nv.pose.heading,
                        speed=nv.speed,
                        road_id=nv_road_id,
                        lane_id=nv_lane_id,
                        lane_index=nv_lane_index,
                    )
                )

        if vehicle.subscribed_to_waypoints_sensor:
            waypoint_paths = vehicle.waypoints_sensor()
        else:
            waypoint_paths = sim.road_map.waypoint_paths(
                vehicle.pose,
                lookahead=1,
                within_radius=vehicle.length,
            )

        closest_lane = sim.road_map.nearest_lane(vehicle.pose.point)
        if closest_lane:
            ego_lane_id = closest_lane.lane_id
            ego_lane_index = closest_lane.index
            ego_road_id = closest_lane.road.road_id
        else:
            ego_lane_id = None
            ego_lane_index = None
            ego_road_id = None
        ego_vehicle_state = vehicle.state

        acceleration_params = {
            "linear_acceleration": None,
            "angular_acceleration": None,
            "linear_jerk": None,
            "angular_jerk": None,
        }
        if vehicle.subscribed_to_accelerometer_sensor:
            acceleration_values = vehicle.accelerometer_sensor(
                ego_vehicle_state.linear_velocity,
                ego_vehicle_state.angular_velocity,
                sim.last_dt,
            )
            acceleration_params.update(
                dict(
                    zip(
                        [
                            "linear_acceleration",
                            "angular_acceleration",
                            "linear_jerk",
                            "angular_jerk",
                        ],
                        acceleration_values,
                    )
                )
            )

        ego_vehicle_observation = EgoVehicleObservation(
            id=ego_vehicle_state.vehicle_id,
            position=ego_vehicle_state.pose.position,
            bounding_box=ego_vehicle_state.dimensions,
            heading=ego_vehicle_state.pose.heading,
            speed=ego_vehicle_state.speed,
            steering=ego_vehicle_state.steering,
            yaw_rate=ego_vehicle_state.yaw_rate,
            road_id=ego_road_id,
            lane_id=ego_lane_id,
            lane_index=ego_lane_index,
            mission=sensor_state.plan.mission,
            linear_velocity=ego_vehicle_state.linear_velocity,
            angular_velocity=ego_vehicle_state.angular_velocity,
            **acceleration_params,
        )

        road_waypoints = (
            vehicle.road_waypoints_sensor()
            if vehicle.subscribed_to_road_waypoints_sensor
            else None
        )

        near_via_points = []
        hit_via_points = []
        if vehicle.subscribed_to_via_sensor:
            (
                near_via_points,
                hit_via_points,
            ) = vehicle.via_sensor()
        via_data = Vias(
            near_via_points=near_via_points,
            hit_via_points=hit_via_points,
        )

        if waypoint_paths:
            vehicle.trip_meter_sensor.append_waypoint_if_new(waypoint_paths[0][0])
        distance_travelled = vehicle.trip_meter_sensor(sim)

        vehicle.driven_path_sensor.track_latest_driven_path(sim)

        if not vehicle.subscribed_to_waypoints_sensor:
            waypoint_paths = None

        drivable_area_grid_map = (
            vehicle.drivable_area_grid_map_sensor()
            if vehicle.subscribed_to_drivable_area_grid_map_sensor
            else None
        )
        ogm = vehicle.ogm_sensor() if vehicle.subscribed_to_ogm_sensor else None
        rgb = vehicle.rgb_sensor() if vehicle.subscribed_to_rgb_sensor else None
        lidar = vehicle.lidar_sensor() if vehicle.subscribed_to_lidar_sensor else None

        done, events = Sensors._is_done_with_events(
            sim, agent_id, vehicle, sensor_state
        )

        if (
            done
            and sensor_state.steps_completed == 1
            and agent_id in sim.agent_manager.ego_agent_ids
        ):
            logger.warning(f"Agent Id: {agent_id} is done on the first step")

        return (
            Observation(
                dt=sim.last_dt,
                step_count=sim.step_count,
                elapsed_sim_time=sim.elapsed_sim_time,
                events=events,
                ego_vehicle_state=ego_vehicle_observation,
                neighborhood_vehicle_states=neighborhood_vehicles,
                waypoint_paths=waypoint_paths,
                distance_travelled=distance_travelled,
                top_down_rgb=rgb,
                occupancy_grid_map=ogm,
                drivable_area_grid_map=drivable_area_grid_map,
                lidar_point_cloud=lidar,
                road_waypoints=road_waypoints,
                via_data=via_data,
            ),
            done,
        )

    @staticmethod
    def step(sim, sensor_state):
        return sensor_state.step()

    @classmethod
    def _agents_alive_done_check(
        cls, agent_manager, agents_alive: AgentsAliveDoneCriteria
    ):
        if not agents_alive:
            return False

        if (
            agents_alive.minimum_ego_agents_alive
            and len(agent_manager.ego_agent_ids) < agents_alive.minimum_ego_agents_alive
        ):
            return True
        if (
            agents_alive.minimum_total_agents_alive
            and len(agent_manager.agent_ids) < agents_alive.minimum_total_agents_alive
        ):
            return True
        if agents_alive.agent_lists_alive:
            for agents_list_alive in agents_alive.agent_lists_alive:
                assert isinstance(
                    agents_list_alive.agents_list, (List, Set, Tuple)
                ), "Please specify a list of agent ids to watch"
                assert isinstance(
                    agents_list_alive.minimum_agents_alive_in_list, int
                ), "Please specify an int for minimum number of alive agents in the list"
                assert (
                    agents_list_alive.minimum_agents_alive_in_list >= 0
                ), "minimum_agents_alive_in_list should not be negative"
                agents_alive_check = [
                    1 if id in agent_manager.agent_ids else 0
                    for id in agents_list_alive.agents_list
                ]
                if (
                    agents_alive_check.count(1)
                    < agents_list_alive.minimum_agents_alive_in_list
                ):
                    return True

        return False

    @classmethod
    def _is_done_with_events(cls, sim, agent_id, vehicle, sensor_state):
        interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        done_criteria = interface.done_criteria

        # TODO:  the following calls nearest_lanes (expensive) 6 times
        reached_goal = cls._agent_reached_goal(sim, vehicle)
        collided = sim.vehicle_did_collide(vehicle.id)
        is_off_road = cls._vehicle_is_off_road(sim, vehicle)
        is_on_shoulder = cls._vehicle_is_on_shoulder(sim, vehicle)
        is_not_moving = cls._vehicle_is_not_moving(sim, vehicle)
        reached_max_episode_steps = sensor_state.reached_max_episode_steps
        is_off_route, is_wrong_way = cls._vehicle_is_off_route_and_wrong_way(
            sim, vehicle
        )
        agents_alive_done = cls._agents_alive_done_check(
            sim.agent_manager, done_criteria.agents_alive
        )

        done = (
            (is_off_road and done_criteria.off_road)
            or reached_goal
            or reached_max_episode_steps
            or (is_on_shoulder and done_criteria.on_shoulder)
            or (collided and done_criteria.collision)
            or (is_not_moving and done_criteria.not_moving)
            or (is_off_route and done_criteria.off_route)
            or (is_wrong_way and done_criteria.wrong_way)
            or agents_alive_done
        )

        events = Events(
            collisions=sim.vehicle_collisions(vehicle.id),
            off_road=is_off_road,
            reached_goal=reached_goal,
            reached_max_episode_steps=reached_max_episode_steps,
            off_route=is_off_route,
            on_shoulder=is_on_shoulder,
            wrong_way=is_wrong_way,
            not_moving=is_not_moving,
            agents_alive_done=agents_alive_done,
        )

        return done, events

    @classmethod
    def _agent_reached_goal(cls, sim, vehicle):
        sensor_state = sim.vehicle_index.sensor_state_for_vehicle_id(vehicle.id)
        distance_travelled = vehicle.trip_meter_sensor()
        mission = sensor_state.plan.mission
        return mission.is_complete(vehicle, distance_travelled)

    @classmethod
    def _vehicle_is_off_road(cls, sim, vehicle):
        return not sim.scenario.road_map.road_with_point(Point(*vehicle.position))

    @classmethod
    def _vehicle_is_on_shoulder(cls, sim, vehicle):
        # XXX: this isn't technically right as this would also return True
        #      for vehicles that are completely off road.
        for corner_coordinate in vehicle.bounding_box:
            if not sim.scenario.road_map.road_with_point(Point(*corner_coordinate)):
                return True
        return False

    @classmethod
    def _vehicle_is_not_moving(cls, sim, vehicle):
        last_n_seconds_considered = 60

        # Flag if the vehicle has been immobile for the past 60 seconds
        if sim.elapsed_sim_time < last_n_seconds_considered:
            return False

        distance = vehicle.driven_path_sensor.distance_travelled(
            sim, last_n_seconds=last_n_seconds_considered
        )

        # Due to controller instabilities there may be some movement even when a
        # vehicle is "stopped". Here we allow 1m of total distance in 60 seconds.
        return distance < 1

    @classmethod
    def _vehicle_is_off_route_and_wrong_way(cls, sim, vehicle):
        """Determines if the agent is on route and on the correct side of the road.

        Args:
            sim: An instance of the simulator.
            agent_id: The id of the agent to check.

        Returns:
            A tuple (is_off_route, is_wrong_way)
            is_off_route:
                Actor's vehicle is not on its route or an oncoming traffic lane.
            is_wrong_way:
                Actor's vehicle is going against the lane travel direction.
        """

        sensor_state = sim.vehicle_index.sensor_state_for_vehicle_id(vehicle.id)
        route_roads = sensor_state.plan.route.roads

        vehicle_pos = Point(*vehicle.position)
        vehicle_minimum_radius_bounds = (
            np.linalg.norm(vehicle.chassis.dimensions.as_lwh[:2]) * 0.5
        )
        # Check that center of vehicle is still close to route
        radius = vehicle_minimum_radius_bounds + 5
        nearest_lane = sim.scenario.road_map.nearest_lane(vehicle_pos, radius=radius)

        # No road nearby, so we're not on route!
        if not nearest_lane:
            return (True, False)

        # Check whether vehicle is in wrong-way
        is_wrong_way = cls._check_wrong_way_event(nearest_lane, vehicle)

        # Check whether vehicle has no-route or is on-route
        if (
            not route_roads  # Vehicle has no-route. E.g., endless mission with a random route
            or nearest_lane.road in route_roads  # Vehicle is on-route
            or nearest_lane.in_junction
        ):
            return (False, is_wrong_way)

        veh_offset = nearest_lane.offset_along_lane(vehicle_pos)

        # so we're obviously not on the route, but we might have just gone
        # over the center line into an oncoming lane...
        for on_lane in nearest_lane.oncoming_lanes_at_offset(veh_offset):
            if on_lane.road in route_roads:
                return (False, is_wrong_way)

        # Vehicle is completely off-route
        return (True, is_wrong_way)

    @staticmethod
    def _vehicle_is_wrong_way(vehicle, closest_lane):
        target_pose = closest_lane.center_pose_at_point(Point(*vehicle.pose.position))
        # Check if the vehicle heading is oriented away from the lane heading.
        return (
            np.fabs(vehicle.pose.heading.relative_to(target_pose.heading)) > 0.5 * np.pi
        )

    @classmethod
    def _check_wrong_way_event(cls, lane_to_check, vehicle):
        # When the vehicle is in an intersection, turn off the `wrong way` check to avoid
        # false positive `wrong way` events.
        if lane_to_check.in_junction:
            return False
        return cls._vehicle_is_wrong_way(vehicle, lane_to_check)


class Sensor:
    def step(self):
        pass


class SensorState:
    def __init__(self, max_episode_steps, plan):
        self._max_episode_steps = max_episode_steps
        self._plan = plan
        self._step = 0

    def step(self):
        self._step += 1

    @property
    def reached_max_episode_steps(self):
        if self._max_episode_steps is None:
            return False

        return self._step >= self._max_episode_steps

    @property
    def plan(self):
        return self._plan

    @property
    def steps_completed(self):
        return self._step


class CameraSensor(Sensor):
    def __init__(
        self,
        vehicle,
        renderer,  # type Renderer or None
        name: str,
        mask: int,
        width: int,
        height: int,
        resolution: float,
    ):
        assert renderer
        self._log = logging.getLogger(self.__class__.__name__)
        self._vehicle = vehicle
        self._camera = renderer.build_offscreen_camera(
            name,
            mask,
            width,
            height,
            resolution,
        )

    def teardown(self):
        self._camera.teardown()

    def step(self):
        self._follow_vehicle()

    def _follow_vehicle(self):
        largest_dim = max(self._vehicle._chassis.dimensions.as_lwh)
        self._camera.update(self._vehicle.pose, 20 * largest_dim)


class DrivableAreaGridMapSensor(CameraSensor):
    def __init__(
        self,
        vehicle,
        width: int,
        height: int,
        resolution: float,
        renderer,  # type Renderer or None
    ):
        super().__init__(
            vehicle,
            renderer,
            "drivable_area_grid_map",
            RenderMasks.DRIVABLE_AREA_HIDE,
            width,
            height,
            resolution,
        )
        self._resolution = resolution

    def __call__(self) -> DrivableAreaGridMap:
        assert (
            self._camera is not None
        ), "Drivable area grid map has not been initialized"

        ram_image = self._camera.wait_for_ram_image(img_format="A")
        mem_view = memoryview(ram_image)
        image = np.frombuffer(mem_view, np.uint8)
        image.shape = (self._camera.tex.getYSize(), self._camera.tex.getXSize(), 1)
        image = np.flipud(image)

        metadata = GridMapMetadata(
            created_at=int(time.time()),
            resolution=self._resolution,
            height=image.shape[0],
            width=image.shape[1],
            camera_pos=self._camera.camera_np.getPos(),
            camera_heading_in_degrees=self._camera.camera_np.getH(),
        )
        return DrivableAreaGridMap(data=image, metadata=metadata)


class OGMSensor(CameraSensor):
    def __init__(
        self,
        vehicle,
        width: int,
        height: int,
        resolution: float,
        renderer,  # type Renderer or None
    ):
        super().__init__(
            vehicle,
            renderer,
            "ogm",
            RenderMasks.OCCUPANCY_HIDE,
            width,
            height,
            resolution,
        )
        self._resolution = resolution

    def __call__(self) -> OccupancyGridMap:
        assert self._camera is not None, "OGM has not been initialized"

        ram_image = self._camera.wait_for_ram_image(img_format="A")
        mem_view = memoryview(ram_image)
        grid = np.frombuffer(mem_view, np.uint8)
        grid.shape = (self._camera.tex.getYSize(), self._camera.tex.getXSize(), 1)
        grid = np.flipud(grid)
        grid = grid.clip(min=0, max=1).astype(np.int8)
        grid *= 100  # full confidence on known cells

        metadata = GridMapMetadata(
            created_at=int(time.time()),
            resolution=self._resolution,
            height=grid.shape[0],
            width=grid.shape[1],
            camera_pos=self._camera.camera_np.getPos(),
            camera_heading_in_degrees=self._camera.camera_np.getH(),
        )
        return OccupancyGridMap(data=grid, metadata=metadata)


class RGBSensor(CameraSensor):
    def __init__(
        self,
        vehicle,
        width: int,
        height: int,
        resolution: float,
        renderer,  # type Renderer or None
    ):
        super().__init__(
            vehicle, renderer, "rgb", RenderMasks.RGB_HIDE, width, height, resolution
        )
        self._resolution = resolution

    def __call__(self) -> TopDownRGB:
        assert self._camera is not None, "RGB has not been initialized"

        ram_image = self._camera.wait_for_ram_image(img_format="RGB")
        mem_view = memoryview(ram_image)
        image = np.frombuffer(mem_view, np.uint8)
        image.shape = (self._camera.tex.getYSize(), self._camera.tex.getXSize(), 3)
        image = np.flipud(image)

        metadata = GridMapMetadata(
            created_at=int(time.time()),
            resolution=self._resolution,
            height=image.shape[0],
            width=image.shape[1],
            camera_pos=self._camera.camera_np.getPos(),
            camera_heading_in_degrees=self._camera.camera_np.getH(),
        )
        return TopDownRGB(data=image, metadata=metadata)


class LidarSensor(Sensor):
    def __init__(
        self,
        vehicle,
        bullet_client,
        sensor_params: SensorParams = None,
        lidar_offset=(0, 0, 1),
    ):
        self._vehicle = vehicle
        self._bullet_client = bullet_client
        self._lidar_offset = np.array(lidar_offset)

        self._lidar = Lidar(
            self._vehicle.position + self._lidar_offset,
            sensor_params,
            self._bullet_client,
        )

    def step(self):
        self._follow_vehicle()

    def _follow_vehicle(self):
        self._lidar.origin = self._vehicle.position + self._lidar_offset

    def __call__(self):
        return self._lidar.compute_point_cloud()

    def teardown(self):
        pass


class DrivenPathSensor(Sensor):
    """Tracks the driven path as a series of positions (regardless if the vehicle is
    following the route or not). For performance reasons it only keeps the last
    N=max_path_length path segments.
    """

    Entry = namedtuple("TimeAndPos", ["timestamp", "position"])

    def __init__(self, vehicle, max_path_length: float = 500):
        self._vehicle = vehicle
        self._driven_path = deque(maxlen=max_path_length)

    def track_latest_driven_path(self, sim):
        pos = self._vehicle.position[:2]
        self._driven_path.append(
            DrivenPathSensor.Entry(timestamp=sim.elapsed_sim_time, position=pos)
        )

    def __call__(self):
        return [x.position for x in self._driven_path]  # only return the positions

    def teardown(self):
        pass

    def distance_travelled(
        self, sim, last_n_seconds: float = None, last_n_steps: int = None
    ):
        if last_n_seconds is None and last_n_steps is None:
            raise ValueError("Either last N seconds or last N steps must be provided")

        if last_n_steps is not None:
            n = last_n_steps + 1  # to factor in the current step we're on
            filtered_pos = [x.position for x in self._driven_path][-n:]
        else:  # last_n_seconds
            threshold = sim.elapsed_sim_time - last_n_seconds
            filtered_pos = [
                x.position for x in self._driven_path if x.timestamp >= threshold
            ]

        xs = np.array([p[0] for p in filtered_pos])
        ys = np.array([p[1] for p in filtered_pos])
        dist_array = (xs[:-1] - xs[1:]) ** 2 + (ys[:-1] - ys[1:]) ** 2
        return np.sum(np.sqrt(dist_array))


class TripMeterSensor(Sensor):
    """Tracks distance travelled along the route (in KM). Kilometeres driven while
    off-route are not counted as part of the total.
    """

    def __init__(self, vehicle, sim, plan):
        self._vehicle = vehicle
        self._sim = sim
        self._plan = plan
        self._wps_for_distance = []
        self._dist_travelled = 0.0
        self._last_dist_travelled = 0.0
        waypoint_paths = sim.road_map.waypoint_paths(
            vehicle.pose,
            lookahead=1,
            within_radius=vehicle.length,
        )
        if waypoint_paths:
            self._wps_for_distance.append(waypoint_paths[0][0])

    def append_waypoint_if_new(self, new_wp):
        # Distance calculation. Intention is the shortest trip travelled at the lane
        # level the agent has travelled. This is to prevent lateral movement from
        # increasing the total distance travelled.
        self._last_dist_travelled = self._dist_travelled

        wp_road = self._sim.road_map.lane_by_id(new_wp.lane_id).road.road_id

        should_count_wp = (
            # if we do not have a fixed route, we count all waypoints we accumulate
            not self._plan.mission.has_fixed_route
            # if we have a route to follow, only count wps on route
            or wp_road in self._plan.route.roads
        )

        if not self._wps_for_distance:
            if should_count_wp:
                self._wps_for_distance.append(new_wp)
            return
        most_recent_wp = self._wps_for_distance[-1]

        threshold_for_counting_wp = 0.5  # meters from last tracked waypoint
        if (
            np.linalg.norm(new_wp.pos - most_recent_wp.pos) > threshold_for_counting_wp
            and should_count_wp
        ):
            self._dist_travelled += TripMeterSensor._compute_additional_dist_travelled(
                most_recent_wp, new_wp
            )
            self._wps_for_distance.append(new_wp)

    @staticmethod
    def _compute_additional_dist_travelled(recent_wp, waypoint):
        heading_vec = recent_wp.heading.direction_vector()
        disp_vec = waypoint.pos - recent_wp.pos
        direction = np.sign(np.dot(heading_vec, disp_vec))
        distance = np.linalg.norm(disp_vec)
        return direction * distance

    def __call__(self, increment=False):
        if increment:
            return self._dist_travelled - self._last_dist_travelled

        return self._dist_travelled

    def teardown(self):
        pass


class NeighborhoodVehiclesSensor(Sensor):
    def __init__(self, vehicle, sim, radius=None):
        self._vehicle = vehicle
        self._sim = sim
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    def __call__(self):
        return self._sim.neighborhood_vehicles_around_vehicle(
            self._vehicle, radius=self._radius
        )

    def teardown(self):
        pass


class WaypointsSensor(Sensor):
    def __init__(self, vehicle, plan: Plan, lookahead=32):
        self._vehicle = vehicle
        self._plan = plan
        self._lookahead = lookahead

    def __call__(self):
        return self._plan.road_map.waypoint_paths(
            self._vehicle.pose,
            lookahead=self._lookahead,
            route=self._plan.route,
        )

    def teardown(self):
        pass


class RoadWaypointsSensor(Sensor):
    def __init__(self, vehicle, sim, plan, horizon=32):
        self._vehicle = vehicle
        self._road_map = sim.road_map
        self._plan = plan
        self._horizon = horizon

    def __call__(self):
        veh_pt = self._vehicle.pose.point
        lane = self._road_map.nearest_lane(veh_pt)
        if not lane:
            return RoadWaypoints(lanes={}, route_waypoints=[])
        road = lane.road
        lane_paths = {}
        for croad in (
            [road] + road.parallel_roads + road.oncoming_roads_at_point(veh_pt)
        ):
            for lane in croad.lanes:
                lane_paths[lane.lane_id] = self.paths_for_lane(lane)

        route_waypoints = self.route_waypoints()

        return RoadWaypoints(lanes=lane_paths, route_waypoints=route_waypoints)

    def route_waypoints(self):
        return self._road_map.waypoint_paths(
            self._vehicle.pose,
            lookahead=self._horizon,
            route=self._plan.route,
        )

    def paths_for_lane(self, lane, overflow_offset=None):
        # XXX: the following assumes waypoint spacing is 1m
        if overflow_offset is None:
            offset = lane.offset_along_lane(Point(*self._vehicle.position))
            start_offset = offset - self._horizon
        else:
            start_offset = lane.length + overflow_offset

        incoming_lanes = lane.incoming_lanes  # XXX: was "getIncoming(onlyDirect=True)"
        if start_offset < 0 and len(incoming_lanes) > 0:
            paths = []
            for lane in incoming_lanes:
                paths += self.paths_for_lane(lane, start_offset)
            return paths
        else:
            start_offset = max(0, start_offset)
            wp_start = lane.from_lane_coord(RefLinePoint(start_offset))
            adj_pose = Pose.from_center(wp_start, self._vehicle.heading)
            wps_to_lookahead = self._horizon * 2
            paths = lane.waypoint_paths_for_pose(
                pose=adj_pose,
                lookahead=wps_to_lookahead,
                route=self._plan.route,
            )
            return paths

    def teardown(self):
        pass


class AccelerometerSensor(Sensor):
    def __init__(self, vehicle):
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

    def teardown(self):
        pass


class ViaSensor(Sensor):
    def __init__(self, vehicle, plan, lane_acquisition_range, speed_accuracy):
        self._consumed_via_points = set()
        self._plan: Plan = plan
        self._acquisition_range = lane_acquisition_range
        self._vehicle = vehicle
        self._speed_accuracy = speed_accuracy

    @property
    def _vias(self) -> Iterable[Via]:
        return self._plan.mission.via

    def __call__(self):
        near_points: List[ViaPoint] = list()
        hit_points: List[ViaPoint] = list()
        vehicle_position = self._vehicle.position[:2]

        @lru_cache()
        def closest_point_on_lane(position, lane_id):
            lane = self._plan.road_map.lane_by_id(lane_id)
            return lane.center_at_point(position)

        for via in self._vias:
            closest_position_on_lane = closest_point_on_lane(
                tuple(vehicle_position), via.lane_id
            )
            closest_position_on_lane = closest_position_on_lane[:2]

            dist_from_lane_sq = squared_dist(vehicle_position, closest_position_on_lane)
            if dist_from_lane_sq > self._acquisition_range ** 2:
                continue

            point = ViaPoint(
                tuple(via.position),
                lane_index=via.lane_index,
                road_id=via.road_id,
                required_speed=via.required_speed,
            )

            near_points.append(point)
            dist_from_point_sq = squared_dist(vehicle_position, via.position)
            if (
                dist_from_point_sq <= via.hit_distance ** 2
                and via not in self._consumed_via_points
                and np.isclose(
                    self._vehicle.speed, via.required_speed, atol=self._speed_accuracy
                )
            ):
                self._consumed_via_points.add(via)
                hit_points.append(point)

        return (
            sorted(
                near_points,
                key=lambda point: squared_dist(point.position, vehicle_position),
            ),
            hit_points,
        )

    def teardown(self):
        pass
