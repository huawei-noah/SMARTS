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
import logging
import sys
import time
import weakref
from collections import deque, namedtuple
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from smarts.core.agent_interface import AgentsAliveDoneCriteria
from smarts.core.plan import Plan
from smarts.core.road_map import RoadMap, Waypoint
from smarts.core.signals import SignalState
from smarts.core.utils.math import squared_dist

from .coordinates import Heading, Point, Pose, RefLinePoint
from .events import Events
from .lidar import Lidar
from .lidar_sensor_params import SensorParams
from .masks import RenderMasks
from .observations import (
    Collision,
    DrivableAreaGridMap,
    EgoVehicleObservation,
    GridMapMetadata,
    Observation,
    OccupancyGridMap,
    RoadWaypoints,
    SignalObservation,
    TopDownRGB,
    VehicleObservation,
    ViaPoint,
    Vias,
)
from .plan import Via

logger = logging.getLogger(__name__)

LANE_ID_CONSTANT = "off_lane"
ROAD_ID_CONSTANT = "off_road"
LANE_INDEX_CONSTANT = -1


def _make_vehicle_observation(road_map, neighborhood_vehicle):
    nv_lane = road_map.nearest_lane(neighborhood_vehicle.pose.point, radius=3)
    if nv_lane:
        nv_road_id = nv_lane.road.road_id
        nv_lane_id = nv_lane.lane_id
        nv_lane_index = nv_lane.index
    else:
        nv_road_id = ROAD_ID_CONSTANT
        nv_lane_id = LANE_ID_CONSTANT
        nv_lane_index = LANE_INDEX_CONSTANT

    return VehicleObservation(
        id=neighborhood_vehicle.actor_id,
        position=neighborhood_vehicle.pose.position,
        bounding_box=neighborhood_vehicle.dimensions,
        heading=neighborhood_vehicle.pose.heading,
        speed=neighborhood_vehicle.speed,
        road_id=nv_road_id,
        lane_id=nv_lane_id,
        lane_index=nv_lane_index,
        lane_position=None,
    )


class Sensors:
    """Sensor utility"""

    _log = logging.getLogger("Sensors")

    @staticmethod
    def observe_batch(
        sim, agent_id, sensor_states, vehicles
    ) -> Tuple[Dict[str, Observation], Dict[str, bool]]:
        """Operates all sensors on a batch of vehicles for a single agent."""
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
    def observe(sim, agent_id, sensor_state, vehicle) -> Tuple[Observation, bool]:
        """Generate observations for the given agent around the given vehicle."""
        neighborhood_vehicle_states = None
        if vehicle.subscribed_to_neighborhood_vehicle_states_sensor:
            neighborhood_vehicle_states = []
            for nv in vehicle.neighborhood_vehicle_states_sensor():
                veh_obs = _make_vehicle_observation(sim.road_map, nv)
                nv_lane_pos = None
                if (
                    veh_obs.lane_id is not LANE_ID_CONSTANT
                    and vehicle.subscribed_to_lane_position_sensor
                ):
                    nv_lane_pos = vehicle.lane_position_sensor(
                        sim.road_map.lane_by_id(veh_obs.lane_id), nv
                    )
                neighborhood_vehicle_states.append(
                    veh_obs._replace(lane_position=nv_lane_pos)
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
        ego_lane_pos = None
        if closest_lane:
            ego_lane_id = closest_lane.lane_id
            ego_lane_index = closest_lane.index
            ego_road_id = closest_lane.road.road_id
            if vehicle.subscribed_to_lane_position_sensor:
                ego_lane_pos = vehicle.lane_position_sensor(closest_lane, vehicle)
        else:
            ego_lane_id = LANE_ID_CONSTANT
            ego_lane_index = LANE_INDEX_CONSTANT
            ego_road_id = ROAD_ID_CONSTANT
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

        ego_vehicle = EgoVehicleObservation(
            id=ego_vehicle_state.actor_id,
            position=ego_vehicle_state.pose.point.as_np_array,
            bounding_box=ego_vehicle_state.dimensions,
            heading=Heading(ego_vehicle_state.pose.heading),
            speed=ego_vehicle_state.speed,
            steering=ego_vehicle_state.steering,
            yaw_rate=ego_vehicle_state.yaw_rate,
            road_id=ego_road_id,
            lane_id=ego_lane_id,
            lane_index=ego_lane_index,
            mission=sensor_state.plan.mission,
            linear_velocity=ego_vehicle_state.linear_velocity,
            angular_velocity=ego_vehicle_state.angular_velocity,
            lane_position=ego_lane_pos,
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
            vehicle.trip_meter_sensor.update_distance_wps_record(waypoint_paths)
        distance_travelled = vehicle.trip_meter_sensor(sim)

        vehicle.driven_path_sensor.track_latest_driven_path(sim)

        if not vehicle.subscribed_to_waypoints_sensor:
            waypoint_paths = None

        drivable_area_grid_map = (
            vehicle.drivable_area_grid_map_sensor()
            if vehicle.subscribed_to_drivable_area_grid_map_sensor
            else None
        )
        occupancy_grid_map = (
            vehicle.ogm_sensor() if vehicle.subscribed_to_ogm_sensor else None
        )
        top_down_rgb = (
            vehicle.rgb_sensor() if vehicle.subscribed_to_rgb_sensor else None
        )
        lidar_point_cloud = (
            vehicle.lidar_sensor() if vehicle.subscribed_to_lidar_sensor else None
        )

        done, events = Sensors._is_done_with_events(
            sim, agent_id, vehicle, sensor_state
        )

        if done and sensor_state.steps_completed == 1:
            agent_type = "Social agent"
            if agent_id in sim.agent_manager.ego_agent_ids:
                agent_type = "Ego agent"
            logger.warning(
                f"{agent_type} with Agent ID: {agent_id} is done on the first step"
            )

        signals = None
        if vehicle.subscribed_to_signals_sensor:
            provider_state = sim._last_provider_state
            signals = vehicle.signals_sensor(
                closest_lane,
                ego_lane_pos,
                ego_vehicle_state,
                sensor_state.plan,
                provider_state,
            )

        agent_controls = agent_id == sim.vehicle_index.actor_id_from_vehicle_id(
            ego_vehicle_state.actor_id
        )

        return (
            Observation(
                dt=sim.last_dt,
                step_count=sim.step_count,
                steps_completed=sensor_state.steps_completed,
                elapsed_sim_time=sim.elapsed_sim_time,
                events=events,
                ego_vehicle_state=ego_vehicle,
                under_this_agent_control=agent_controls,
                neighborhood_vehicle_states=neighborhood_vehicle_states,
                waypoint_paths=waypoint_paths,
                distance_travelled=distance_travelled,
                top_down_rgb=top_down_rgb,
                occupancy_grid_map=occupancy_grid_map,
                drivable_area_grid_map=drivable_area_grid_map,
                lidar_point_cloud=lidar_point_cloud,
                road_waypoints=road_waypoints,
                via_data=via_data,
                signals=signals,
            ),
            done,
        )

    @staticmethod
    def step(sim, sensor_state):
        """Step the sensor state."""
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
        event_config = interface.event_configuration

        # TODO:  the following calls nearest_lanes (expensive) 6 times
        reached_goal = cls._agent_reached_goal(sim, vehicle)
        collided = sim.vehicle_did_collide(vehicle.id)
        is_off_road = cls._vehicle_is_off_road(sim, vehicle)
        is_on_shoulder = cls._vehicle_is_on_shoulder(sim, vehicle)
        is_not_moving = cls._vehicle_is_not_moving(
            sim, vehicle, event_config.not_moving_time, event_config.not_moving_distance
        )
        reached_max_episode_steps = sensor_state.reached_max_episode_steps
        is_off_route, is_wrong_way = cls._vehicle_is_off_route_and_wrong_way(
            sim, vehicle
        )
        agents_alive_done = cls._agents_alive_done_check(
            sim.agent_manager, done_criteria.agents_alive
        )

        done = not sim.resetting and (
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
        return not sim.scenario.road_map.road_with_point(vehicle.pose.point)

    @classmethod
    def _vehicle_is_on_shoulder(cls, sim, vehicle):
        # XXX: this isn't technically right as this would also return True
        #      for vehicles that are completely off road.
        for corner_coordinate in vehicle.bounding_box:
            if not sim.scenario.road_map.road_with_point(Point(*corner_coordinate)):
                return True
        return False

    @classmethod
    def _vehicle_is_not_moving(
        cls, sim, vehicle, last_n_seconds_considered, min_distance_moved
    ):
        # Flag if the vehicle has been immobile for the past 'last_n_seconds_considered' seconds
        if sim.elapsed_sim_time < last_n_seconds_considered:
            return False

        distance = vehicle.driven_path_sensor.distance_travelled(
            sim, last_n_seconds=last_n_seconds_considered
        )

        # Due to controller instabilities there may be some movement even when a
        # vehicle is "stopped".
        return distance < min_distance_moved

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

        vehicle_pos = vehicle.pose.point
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

        # Check for case if there was an early merge into another incoming lane. This means the
        # vehicle should still be following the lane direction to be valid as still on route.
        if not is_wrong_way:
            # See if the lane leads into the current route
            for lane in nearest_lane.outgoing_lanes:
                if lane.road in route_roads:
                    return (False, is_wrong_way)
                # If outgoing lane is in a junction see if the junction lane leads into current route.
                if lane.in_junction:
                    for out_lane in lane.outgoing_lanes:
                        if out_lane.road in route_roads:
                            return (False, is_wrong_way)

        # Vehicle is completely off-route
        return (True, is_wrong_way)

    @staticmethod
    def _vehicle_is_wrong_way(vehicle, closest_lane):
        target_pose = closest_lane.center_pose_at_point(vehicle.pose.point)
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
    """The sensor base class."""

    def step(self):
        """Update sensor state."""
        pass

    def teardown(self):
        """Clean up internal resources"""
        raise NotImplementedError


class SensorState:
    """Sensor state information"""

    def __init__(self, max_episode_steps: int, plan):
        self._max_episode_steps = max_episode_steps
        self._plan = plan
        self._step = 0

    def step(self):
        """Update internal state."""
        self._step += 1

    @property
    def reached_max_episode_steps(self) -> bool:
        """Inbuilt sensor information that describes if episode step limit has been reached."""
        if self._max_episode_steps is None:
            return False

        return self._step >= self._max_episode_steps

    @property
    def plan(self):
        """Get the current plan for the actor."""
        return self._plan

    @property
    def steps_completed(self) -> int:
        """Get the number of steps where this sensor has been updated."""
        return self._step


class CameraSensor(Sensor):
    """The base for a sensor that renders images."""

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
        self._follow_vehicle()  # ensure we have a correct initial camera position

    def teardown(self):
        self._camera.teardown()

    def step(self):
        self._follow_vehicle()

    def _follow_vehicle(self):
        largest_dim = max(self._vehicle._chassis.dimensions.as_lwh)
        self._camera.update(self._vehicle.pose, 20 * largest_dim)


class DrivableAreaGridMapSensor(CameraSensor):
    """A sensor that renders drivable area from around its target actor."""

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
            camera_position=self._camera.camera_np.getPos(),
            camera_heading_in_degrees=self._camera.camera_np.getH(),
        )
        return DrivableAreaGridMap(data=image, metadata=metadata)


class OGMSensor(CameraSensor):
    """A sensor that renders occupancy information from around its target actor."""

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

        metadata = GridMapMetadata(
            created_at=int(time.time()),
            resolution=self._resolution,
            height=grid.shape[0],
            width=grid.shape[1],
            camera_position=self._camera.camera_np.getPos(),
            camera_heading_in_degrees=self._camera.camera_np.getH(),
        )
        return OccupancyGridMap(data=grid, metadata=metadata)


class RGBSensor(CameraSensor):
    """A sensor that renders color values from around its target actor."""

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
            "top_down_rgb",
            RenderMasks.RGB_HIDE,
            width,
            height,
            resolution,
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
            camera_position=self._camera.camera_np.getPos(),
            camera_heading_in_degrees=self._camera.camera_np.getH(),
        )
        return TopDownRGB(data=image, metadata=metadata)


class LidarSensor(Sensor):
    """A lidar sensor."""

    def __init__(
        self,
        vehicle,
        bullet_client,
        sensor_params: Optional[SensorParams] = None,
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

    def __init__(self, vehicle, max_path_length: int = 500):
        self._vehicle = vehicle
        self._driven_path = deque(maxlen=max_path_length)

    def track_latest_driven_path(self, sim):
        """Records the current location of the tracked vehicle."""
        pos = self._vehicle.position[:2]
        self._driven_path.append(
            DrivenPathSensor.Entry(timestamp=sim.elapsed_sim_time, position=pos)
        )

    def __call__(self, count=sys.maxsize):
        return [x.position for x in self._driven_path][-count:]

    def teardown(self):
        pass

    def distance_travelled(
        self,
        sim,
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
            threshold = sim.elapsed_sim_time - last_n_seconds
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

    def __init__(self, vehicle, road_map, plan):
        self._vehicle = vehicle
        self._road_map: RoadMap = road_map
        self._plan = plan
        self._wps_for_distance: List[Waypoint] = []
        self._dist_travelled = 0.0
        self._last_dist_travelled = 0.0
        self._last_actor_position = None

    def update_distance_wps_record(self, waypoint_paths):
        """Append a waypoint to the history if it is not already counted."""
        # Distance calculation. Intention is the shortest trip travelled at the lane
        # level the agent has travelled. This is to prevent lateral movement from
        # increasing the total distance travelled.
        self._last_dist_travelled = self._dist_travelled

        new_wp = waypoint_paths[0][0]
        wp_road = self._road_map.lane_by_id(new_wp.lane_id).road.road_id

        should_count_wp = (
            # if we do not have a fixed route, we count all waypoints we accumulate
            not self._plan.mission.requires_route
            # if we have a route to follow, only count wps on route
            or wp_road in [road.road_id for road in self._plan.route.roads]
        )

        if not self._wps_for_distance:
            self._last_actor_position = self._vehicle.pose.position
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
            self._vehicle.pose.position,
            self._last_actor_position,
        )
        self._dist_travelled += additional_distance
        self._last_actor_position = self._vehicle.pose.position

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

    def __call__(self, increment=False):
        if increment:
            return self._dist_travelled - self._last_dist_travelled

        return self._dist_travelled

    def teardown(self):
        pass


class NeighborhoodVehiclesSensor(Sensor):
    """Detects other vehicles around the sensor equipped vehicle."""

    def __init__(self, vehicle, sim, radius=None):
        self._vehicle = vehicle
        self._sim = weakref.ref(sim)
        self._radius = radius

    @property
    def radius(self):
        """Radius to check for nearby vehicles."""
        return self._radius

    def __call__(self):
        sim = self._sim()
        assert sim
        return sim.neighborhood_vehicles_around_vehicle(
            self._vehicle, radius=self._radius
        )

    def teardown(self):
        pass


class WaypointsSensor(Sensor):
    """Detects waypoints leading forward along the vehicle plan."""

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
    """Detects waypoints from all paths nearby the vehicle."""

    def __init__(self, vehicle, sim, plan, horizon=32):
        self._vehicle = vehicle
        self._road_map = sim.road_map
        self._plan = plan
        self._horizon = horizon

    def __call__(self) -> RoadWaypoints:
        veh_pt = self._vehicle.pose.point
        lane = self._road_map.nearest_lane(veh_pt)
        if not lane:
            return RoadWaypoints(lanes={})
        road = lane.road
        lane_paths = {}
        for croad in (
            [road] + road.parallel_roads + road.oncoming_roads_at_point(veh_pt)
        ):
            for lane in croad.lanes:
                lane_paths[lane.lane_id] = self.paths_for_lane(lane)

        return RoadWaypoints(lanes=lane_paths)

    def paths_for_lane(self, lane, overflow_offset=None):
        """Gets waypoint paths along the given lane."""
        # XXX: the following assumes waypoint spacing is 1m
        if overflow_offset is None:
            offset = lane.offset_along_lane(self._vehicle.pose.point)
            start_offset = offset - self._horizon
        else:
            start_offset = lane.length + overflow_offset

        incoming_lanes = lane.incoming_lanes
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
    """Tracks motion changes within the vehicle equipped with this sensor."""

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


class LanePositionSensor(Sensor):
    """Tracks lane-relative RefLine (Frenet) coordinates."""

    def __init__(self, vehicle):
        pass

    def __call__(self, lane: RoadMap.Lane, vehicle):
        return lane.to_lane_coord(vehicle.pose.point)

    def teardown(self):
        pass


class ViaSensor(Sensor):
    """Tracks collection of ViaPoint collectables"""

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
            if dist_from_lane_sq > self._acquisition_range**2:
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
                dist_from_point_sq <= via.hit_distance**2
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


class SignalsSensor(Sensor):
    """Reports state of traffic signals (lights) in the lanes ahead of vehicle."""

    def __init__(self, vehicle, road_map: RoadMap, lookahead: float):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._vehicle = vehicle
        self._road_map = road_map
        self._lookahead = lookahead

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
        state,  # VehicleState
        plan: Plan,
        provider_state,  # ProviderState
    ) -> List[SignalObservation]:
        result = []
        if not lane:
            return result
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
                self._logger.warning(
                    "could not find signal state corresponding with feature_id={signal.feature_id}"
                )
                continue
            assert isinstance(signal_state, SignalState)
            controlled_lanes = None
            if signal_state.controlled_lanes:
                controlled_lanes = [cl.lane_id for cl in signal_state.controlled_lanes]
            result.append(
                SignalObservation(
                    state=signal_state.state,
                    stop_point=signal_state.stopping_pos,
                    controlled_lanes=controlled_lanes,
                    last_changed=signal_state.last_changed,
                )
            )

        return result

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

    def teardown(self):
        pass
