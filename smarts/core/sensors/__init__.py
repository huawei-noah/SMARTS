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
from __future__ import annotations

import logging
import math
import re
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

import numpy as np

from smarts.core.coordinates import Heading, Point
from smarts.core.events import Events
from smarts.core.observations import (
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
from smarts.core.plan import Plan, PlanFrame
from smarts.core.renderer_base import RendererBase
from smarts.core.sensor import (
    AccelerometerSensor,
    CameraSensor,
    CustomRenderSensor,
    DrivableAreaGridMapSensor,
    DrivenPathSensor,
    LanePositionSensor,
    LidarSensor,
    NeighborhoodVehiclesSensor,
    OGMSensor,
    RGBSensor,
    RoadWaypointsSensor,
    Sensor,
    SignalsSensor,
    TripMeterSensor,
    ViaSensor,
    WaypointsSensor,
)
from smarts.core.utils.core_logging import timeit

if TYPE_CHECKING:
    from smarts.core.agent_interface import (
        AgentInterface,
        AgentsAliveDoneCriteria,
        InterestDoneCriteria,
    )
    from smarts.core.road_map import RoadMap
    from smarts.core.simulation_frame import SimulationFrame
    from smarts.core.simulation_local_constants import SimulationLocalConstants
    from smarts.core.utils.pybullet import bullet_client as bc
    from smarts.core.vehicle import Vehicle
    from smarts.core.vehicle_state import VehicleState

logger = logging.getLogger(__name__)

LANE_ID_CONSTANT = "off_lane"
ROAD_ID_CONSTANT = "off_road"
LANE_INDEX_CONSTANT = -1


def _make_vehicle_observation(
    road_map: RoadMap,
    neighborhood_vehicle: VehicleState,
    sim_frame: SimulationFrame,
    interest_extension: Optional[re.Pattern],
):
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
        position=neighborhood_vehicle.pose.position_tuple,
        bounding_box=neighborhood_vehicle.dimensions,
        heading=neighborhood_vehicle.pose.heading,
        speed=neighborhood_vehicle.speed,
        road_id=nv_road_id,
        lane_id=nv_lane_id,
        lane_index=nv_lane_index,
        lane_position=None,
        interest=sim_frame.actor_is_interest(
            neighborhood_vehicle.actor_id, extension=interest_extension
        ),
    )


class SensorState:
    """Sensor state information"""

    def __init__(self, max_episode_steps: Optional[int], plan_frame: PlanFrame):
        self._max_episode_steps = max_episode_steps
        self._plan_frame = plan_frame
        self._step = 0
        self._seen_interest_actors = False
        self._seen_alive_actors = False

    def step(self):
        """Update internal state."""
        self._step += 1

    @property
    def seen_alive_actors(self) -> bool:
        """If an agents alive actor has been spotted before."""
        return self._seen_alive_actors

    @seen_alive_actors.setter
    def seen_alive_actors(self, value: bool):
        self._seen_alive_actors = value

    @property
    def seen_interest_actors(self) -> bool:
        """If an interest actor has been spotted before."""
        return self._seen_interest_actors

    @seen_interest_actors.setter
    def seen_interest_actors(self, value: bool):
        self._seen_interest_actors = value

    @property
    def reached_max_episode_steps(self) -> bool:
        """Inbuilt sensor information that describes if episode step limit has been reached."""
        if self._max_episode_steps is None:
            return False

        return self._step >= self._max_episode_steps

    def get_plan(self, road_map: RoadMap):
        """Get the current plan for the actor."""
        return Plan.from_frame(self._plan_frame, road_map)

    @property
    def steps_completed(self) -> int:
        """Get the number of steps where this sensor has been updated."""
        return self._step

    @classmethod
    def invalid(cls: Type[SensorState]) -> SensorState:
        """Generate a invalid default frame."""
        return cls(None, PlanFrame.empty())


class SensorResolver:
    """An interface describing sensor observation and update systems."""

    # TODO: Remove renderer and bullet client from the arguments
    # TODO: Remove updated sensors from the return. No sensors should be modified in observe!!!
    def observe(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer: RendererBase,
        bullet_client: bc.BulletClient,
    ) -> Tuple[Dict[str, Observation], Dict[str, bool], Dict[str, Dict[str, Sensor]]]:
        """Generate observations

        Args:
            sim_frame (SimulationFrame): The simulation frame.
            sim_local_constants (SimulationLocalConstants): Constraints defined by the local simulator.
            agent_ids (Set[str]): The agents to run.
            renderer (RendererBase): The renderer to use.
            bullet_client (Any): The bullet client. This parameter is likely to be removed.
        """
        raise NotImplementedError()

    def step(self, sim_frame: SimulationFrame, sensor_states: Iterable[SensorState]):
        """Step the sensor state."""
        raise NotImplementedError()

    def _gen_phys_observations(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        bullet_client: bc.BulletClient,
        updated_sensors: Dict[str, Dict[str, Sensor]],
    ) -> Dict[str, Dict[str, Any]]:
        with timeit("physics sensors", logger.debug):
            phys_observations: Dict[str, Dict[str, Any]] = {}
            for agent_id in agent_ids:
                for vehicle_id in sim_frame.vehicles_for_agents[agent_id]:
                    (
                        phys_observations[agent_id],
                        updated_phys_sensors,
                    ) = Sensors.process_physics_sensors(
                        sim_frame,
                        sim_local_constants,
                        sim_frame.sensor_states[vehicle_id],
                        vehicle_id,
                        bullet_client,
                    )
                    updated_sensors[vehicle_id].update(updated_phys_sensors)

        return phys_observations

    def _gen_rendered_observations(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer: RendererBase,
        updated_sensors: Dict[str, Dict[str, Sensor]],
    ):
        with timeit("rendered observations", logger.debug):
            rendering_observations = {}
            for agent_id in agent_ids:
                for vehicle_id in sim_frame.vehicles_for_agents[agent_id]:
                    (
                        rendering_observations[agent_id],
                        updated_unsafe_sensors,
                    ) = Sensors.process_rendering_sensors(
                        sim_frame,
                        sim_local_constants,
                        sim_frame.agent_interfaces[agent_id],
                        sim_frame.sensor_states[vehicle_id],
                        vehicle_id,
                        renderer,
                    )
                    updated_sensors[vehicle_id].update(updated_unsafe_sensors)
        return rendering_observations

    def _sync_custom_camera_sensors(
        self, sim_frame: SimulationFrame, renderer: RendererBase, observations
    ):
        for v_id, sensors in sim_frame.vehicle_sensors.items():
            for s_id, sensor in sensors.items():
                if sensor.serializable or not isinstance(sensor, CustomRenderSensor):
                    continue
                sensor.step(
                    sim_frame=sim_frame, renderer=renderer, observations=observations
                )


class Sensors:
    """Sensor related utilities"""

    @classmethod
    def observe_serializable_sensor_batch(
        cls,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids_for_group: Iterable[str],
    ) -> Tuple[Dict[str, Observation], Dict[str, bool], Dict[str, Dict[str, Sensor]]]:
        """Run the serializable sensors in a batch."""
        observations, dones, updated_sensors = {}, {}, {}
        for agent_id in agent_ids_for_group:
            vehicle_ids = sim_frame.vehicles_for_agents.get(agent_id)
            interface = sim_frame.agent_interfaces.get(agent_id)
            if not vehicle_ids:
                continue
            for vehicle_id in vehicle_ids:
                (
                    observations[agent_id],
                    dones[agent_id],
                    updated_sensors[vehicle_id],
                ) = cls.process_serialization_safe_sensors(
                    sim_frame,
                    sim_local_constants,
                    interface,
                    sim_frame.sensor_states[vehicle_id],
                    vehicle_id,
                    agent_id,
                )
        return observations, dones, updated_sensors

    @staticmethod
    def process_rendering_sensors(
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        interface: AgentInterface,
        sensor_state: SensorState,
        vehicle_id: str,
        renderer: RendererBase,
    ) -> Tuple[Dict[str, Any], Dict[str, Sensor]]:
        """Run observations that can only be done on the main thread."""
        vehicle_sensors = sim_frame.vehicle_sensors[vehicle_id]

        def get_camera_sensor_result(
            sensors: Dict[str, Sensor], sensor_name: str, renderer: RendererBase
        ):
            if (sensor := sensors.get(sensor_name)) is not None:
                return sensor(renderer=renderer)
            return None

        updated_sensors = {
            sensor_name: sensor
            for sensor_name, sensor in vehicle_sensors.items()
            if isinstance(sensor, CameraSensor)
        }

        return (
            dict(
                drivable_area_grid_map=get_camera_sensor_result(
                    vehicle_sensors, "drivable_area_grid_map_sensor", renderer
                ),
                occupancy_grid_map=get_camera_sensor_result(
                    vehicle_sensors, "ogm_sensor", renderer
                ),
                top_down_rgb=get_camera_sensor_result(
                    vehicle_sensors, "rgb_sensor", renderer
                ),
                occlusion_map=get_camera_sensor_result(
                    vehicle_sensors, "occlusion_map_sensor", renderer
                ),
                custom_renders=tuple(
                    get_camera_sensor_result(
                        vehicle_sensors, f"custom_render{i}_sensor", renderer
                    )
                    for i, _ in enumerate(interface.custom_renders)
                ),
            ),
            updated_sensors,
        )

    @staticmethod
    def process_physics_sensors(
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        sensor_state: SensorState,
        vehicle_id: str,
        bullet_client: bc.BulletClient,
    ) -> Tuple[Dict[str, Any], Dict[str, Sensor]]:
        """Run observations that can only be done on the main thread."""
        vehicle_sensors = sim_frame.vehicle_sensors[vehicle_id]

        vehicle_state = sim_frame.vehicle_states[vehicle_id]
        lidar = None
        updated_sensors = {}
        lidar_sensor = vehicle_sensors.get("lidar_sensor")
        if lidar_sensor:
            lidar_sensor.follow_vehicle(vehicle_state)
            lidar = lidar_sensor(bullet_client)
            updated_sensors["lidar_sensor"] = lidar_sensor

        return (
            dict(
                lidar_point_cloud=lidar,
            ),
            updated_sensors,
        )

    @classmethod
    def process_serialization_unsafe_sensors(
        cls,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        interface: AgentInterface,
        sensor_state: SensorState,
        vehicle_id: str,
        renderer: RendererBase,
        bullet_client: bc.BulletClient,
    ) -> Tuple[Dict[str, Any], Dict[str, Sensor]]:
        """Run observations that can only be done on the main thread."""
        p_sensors, p_updated_sensors = cls.process_physics_sensors(
            sim_frame=sim_frame,
            sim_local_constants=sim_local_constants,
            sensor_state=sensor_state,
            vehicle_id=vehicle_id,
            bullet_client=bullet_client,
        )
        r_sensors, r_updated_sensors = cls.process_rendering_sensors(
            sim_frame=sim_frame,
            sim_local_constants=sim_local_constants,
            interface=interface,
            sensor_state=sensor_state,
            vehicle_id=vehicle_id,
            renderer=renderer,
        )

        return {**p_sensors, **r_sensors}, {**p_updated_sensors, **r_updated_sensors}

    @staticmethod
    def process_serialization_safe_sensors(
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        interface: AgentInterface,
        sensor_state: SensorState,
        vehicle_id: str,
        agent_id: Optional[str] = None,
    ) -> Tuple[Observation, bool, Dict[str, Sensor]]:
        """Observations that can be done on any thread."""
        vehicle_sensors = sim_frame.vehicle_sensors[vehicle_id]
        vehicle_state = sim_frame.vehicle_states[vehicle_id]
        plan = sensor_state.get_plan(sim_local_constants.road_map)
        neighborhood_vehicle_states = None
        neighborhood_vehicle_states_sensor = vehicle_sensors.get(
            "neighborhood_vehicle_states_sensor"
        )
        if neighborhood_vehicle_states_sensor:
            neighborhood_vehicle_states = []
            interest_pattern = (
                interface.done_criteria.interest.actors_pattern
                if interface is not None
                and interface.done_criteria.interest is not None
                else None
            )
            for nv in neighborhood_vehicle_states_sensor(
                vehicle_state, sim_frame.vehicle_states.values()
            ):
                veh_obs = _make_vehicle_observation(
                    sim_local_constants.road_map, nv, sim_frame, interest_pattern
                )
                lane_position_sensor = vehicle_sensors.get("lane_position_sensor")
                nv_lane_pos = None
                if veh_obs.lane_id is not LANE_ID_CONSTANT and lane_position_sensor:
                    nv_lane_pos = lane_position_sensor(
                        sim_local_constants.road_map.lane_by_id(veh_obs.lane_id), nv
                    )
                neighborhood_vehicle_states.append(
                    veh_obs._replace(lane_position=nv_lane_pos)
                )

        waypoints_sensor = vehicle_sensors.get("waypoints_sensor")
        if waypoints_sensor:
            waypoint_paths = waypoints_sensor(
                vehicle_state, plan, sim_local_constants.road_map
            )
        else:
            waypoint_paths = sim_local_constants.road_map.waypoint_paths(
                vehicle_state.pose,
                lookahead=1,
                within_radius=vehicle_state.dimensions.length,
            )

        closest_lane = sim_local_constants.road_map.nearest_lane(
            vehicle_state.pose.point
        )
        ego_lane_pos = None
        if closest_lane:
            ego_lane_id = closest_lane.lane_id
            ego_lane_index = closest_lane.index
            ego_road_id = closest_lane.road.road_id
            lane_position_sensor = vehicle_sensors.get("lane_position_sensor")
            if lane_position_sensor:
                ego_lane_pos = lane_position_sensor(closest_lane, vehicle_state)
        else:
            ego_lane_id = LANE_ID_CONSTANT
            ego_lane_index = LANE_INDEX_CONSTANT
            ego_road_id = ROAD_ID_CONSTANT

        acceleration_params = {
            "linear_acceleration": None,
            "angular_acceleration": None,
            "linear_jerk": None,
            "angular_jerk": None,
        }

        accelerometer_sensor = vehicle_sensors.get("accelerometer_sensor")
        if accelerometer_sensor:
            acceleration_values = accelerometer_sensor(
                vehicle_state.linear_velocity,
                vehicle_state.angular_velocity,
                sim_frame.last_dt,
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
                        (
                            (None if av is None else tuple(float(f) for f in av))
                            for av in acceleration_values
                        ),
                    )
                )
            )

        ego_vehicle = EgoVehicleObservation(
            id=vehicle_state.actor_id,
            position=vehicle_state.pose.position_tuple,
            bounding_box=vehicle_state.dimensions,
            heading=vehicle_state.pose.heading,
            speed=vehicle_state.speed,
            steering=vehicle_state.steering,
            yaw_rate=vehicle_state.yaw_rate,
            road_id=ego_road_id,
            lane_id=ego_lane_id,
            lane_index=ego_lane_index,
            mission=plan.mission,
            linear_velocity=vehicle_state.linear_velocity_tuple(),
            angular_velocity=vehicle_state.angular_velocity_tuple(),
            lane_position=ego_lane_pos,
            **acceleration_params,
        )

        road_waypoints_sensor = vehicle_sensors.get("road_waypoints_sensor")
        road_waypoints = (
            road_waypoints_sensor(vehicle_state, plan, sim_local_constants.road_map)
            if road_waypoints_sensor
            else None
        )

        near_via_points = ()

        via_sensor = vehicle_sensors.get("via_sensor")
        if via_sensor:
            near_via_points = via_sensor(
                vehicle_state, plan, sim_local_constants.road_map
            )
        via_data = Vias(
            near_via_points=near_via_points,
        )

        distance_travelled = 0
        trip_meter_sensor: Optional[TripMeterSensor] = vehicle_sensors.get(
            "trip_meter_sensor"
        )
        if trip_meter_sensor:
            if waypoint_paths:
                trip_meter_sensor.update_distance_wps_record(
                    waypoint_paths, vehicle_state, plan, sim_local_constants.road_map
                )
            distance_travelled = trip_meter_sensor(increment=True)

        driven_path_sensor = vehicle_sensors.get("driven_path_sensor")
        if driven_path_sensor:
            driven_path_sensor.track_latest_driven_path(
                sim_frame.elapsed_sim_time, vehicle_state
            )

        if not waypoints_sensor:
            waypoint_paths = None

        done, events = Sensors._is_done_with_events(
            sim_frame,
            sim_local_constants,
            interface,
            vehicle_state,
            sensor_state,
            plan,
        )

        if done and sensor_state.steps_completed == 1:
            logger.warning("Vehicle with ID: %s is done on the first step", vehicle_id)

        signals = None
        signals_sensor = vehicle_sensors.get("signals_sensor")
        if signals_sensor:
            provider_state = sim_frame.last_provider_state
            signals = signals_sensor(
                closest_lane,
                ego_lane_pos,
                vehicle_state,
                plan,
                provider_state,
            )

        agent_controls = (
            agent_id is not None
            and agent_id == sim_frame.agent_vehicle_controls.get(vehicle_state.actor_id)
        )
        updated_sensors = {
            sensor_name: sensor
            for sensor_name, sensor in vehicle_sensors.items()
            if sensor.mutable and sensor.serializable
        }

        return (
            Observation(
                dt=sim_frame.last_dt,
                step_count=sim_frame.step_count,
                steps_completed=sensor_state.steps_completed,
                elapsed_sim_time=sim_frame.elapsed_sim_time,
                events=events,
                ego_vehicle_state=ego_vehicle,
                under_this_agent_control=agent_controls,
                neighborhood_vehicle_states=neighborhood_vehicle_states or (),
                waypoint_paths=waypoint_paths,
                distance_travelled=distance_travelled,
                road_waypoints=road_waypoints,
                via_data=via_data,
                signals=signals,
            ),
            done,
            updated_sensors,
        )

    @classmethod
    def observe_vehicle(
        cls,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        interface: AgentInterface,
        sensor_state: SensorState,
        vehicle: Vehicle,
        renderer: RendererBase,
        bullet_client: bc.BulletClient,
    ) -> Tuple[Observation, bool, Dict[str, Sensor]]:
        """Generate observations for the given agent around the given vehicle."""
        args = [sim_frame, sim_local_constants, interface, sensor_state, vehicle.id]
        safe_obs, dones, updated_safe_sensors = cls.process_serialization_safe_sensors(
            *args
        )
        unsafe_obs, updated_unsafe_sensors = cls.process_serialization_unsafe_sensors(
            *args, renderer, bullet_client
        )
        complete_obs = safe_obs._replace(
            **unsafe_obs,
        )
        return (complete_obs, dones, {**updated_safe_sensors, **updated_unsafe_sensors})

    @classmethod
    def _agents_alive_done_check(
        cls,
        ego_agent_ids: Collection[str],
        agent_ids: Collection[str],
        agents_alive: Optional[AgentsAliveDoneCriteria],
    ):
        if agents_alive is None:
            return False

        if (
            agents_alive.minimum_ego_agents_alive
            and len(ego_agent_ids) < agents_alive.minimum_ego_agents_alive
        ):
            return True
        if (
            agents_alive.minimum_total_agents_alive
            and len(agent_ids) < agents_alive.minimum_total_agents_alive
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
                    1 if id in agent_ids else 0 for id in agents_list_alive.agents_list
                ]
                if (
                    agents_alive_check.count(1)
                    < agents_list_alive.minimum_agents_alive_in_list
                ):
                    return True

        return False

    @classmethod
    def _interest_done_check(
        cls,
        interest_actors: Collection[str],
        sensor_state: SensorState,
        interest_criteria: Optional[InterestDoneCriteria],
    ):
        if interest_criteria is None or len(interest_actors) > 0:
            sensor_state.seen_interest_actors = True
            return False

        if interest_criteria.strict or sensor_state.seen_interest_actors:
            # if agent requires the actor to exist immediately
            # OR if previously seen relevant actors but no actors match anymore
            return True

        ## if never seen a relevant actor
        return False

    @classmethod
    def _is_done_with_events(
        cls,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        interface: AgentInterface,
        vehicle_state: VehicleState,
        sensor_state: SensorState,
        plan: Plan,
    ):
        vehicle_sensors = sim_frame.vehicle_sensors[vehicle_state.actor_id]
        done_criteria = interface.done_criteria
        event_config = interface.event_configuration
        interest = interface.done_criteria.interest

        # Optimization: avoid calling nearest_lanes 6 times
        vehicle_pos = vehicle_state.pose.point
        vehicle_minimum_radius_bounds = (
            np.linalg.norm(vehicle_state.dimensions.as_lwh[:2]) * 0.5
        )
        radius = vehicle_minimum_radius_bounds + 5
        nearest_lanes_and_dists = sim_local_constants.road_map.nearest_lanes(
            vehicle_pos, radius=radius
        )
        nearest_lanes = tuple(
            [nl for (nl, _) in nearest_lanes_and_dists]
        )  # Needs to be a tuple to be hashable

        reached_goal = cls._agent_reached_goal(
            sensor_state, plan, vehicle_state, vehicle_sensors.get("trip_meter_sensor")
        )
        collided = sim_frame.vehicle_did_collide(vehicle_state.actor_id)
        is_off_road = cls._vehicle_is_off_road(
            sim_local_constants.road_map, vehicle_state, nearest_lanes
        )
        is_on_shoulder = cls._vehicle_is_on_shoulder(
            sim_local_constants.road_map, vehicle_state, nearest_lanes
        )
        is_not_moving = cls._vehicle_is_not_moving(
            sim_frame,
            event_config.not_moving_time,
            event_config.not_moving_distance,
            vehicle_sensors.get("driven_path_sensor"),
        )
        reached_max_episode_steps = sensor_state.reached_max_episode_steps
        is_off_route, is_wrong_way = cls._vehicle_is_off_route_and_wrong_way(
            sim_frame, sim_local_constants, vehicle_state, plan, nearest_lanes_and_dists
        )
        agents_alive_done = cls._agents_alive_done_check(
            sim_frame.ego_ids, sim_frame.potential_agent_ids, done_criteria.agents_alive
        )
        interest_done = False
        if interest:
            interest_done = cls._interest_done_check(
                sim_frame.interest_actors(interest.actors_pattern),
                sensor_state,
                interest_criteria=interest,
            )

        done = not sim_frame.resetting and (
            (is_off_road and done_criteria.off_road)
            or reached_goal
            or reached_max_episode_steps
            or (is_on_shoulder and done_criteria.on_shoulder)
            or (collided and done_criteria.collision)
            or (is_not_moving and done_criteria.not_moving)
            or (is_off_route and done_criteria.off_route)
            or (is_wrong_way and done_criteria.wrong_way)
            or agents_alive_done
            or interest_done
        )

        events = Events(
            collisions=tuple(
                sim_frame.filtered_vehicle_collisions(vehicle_state.actor_id)
            ),
            off_road=is_off_road,
            reached_goal=reached_goal,
            reached_max_episode_steps=reached_max_episode_steps,
            off_route=is_off_route,
            on_shoulder=is_on_shoulder,
            wrong_way=is_wrong_way,
            not_moving=is_not_moving,
            agents_alive_done=agents_alive_done,
            interest_done=interest_done,
        )

        return done, events

    @classmethod
    def _agent_reached_goal(
        cls,
        sensor_state: SensorState,
        plan: Plan,
        vehicle_state: VehicleState,
        trip_meter_sensor: TripMeterSensor,
    ) -> bool:
        if not trip_meter_sensor or plan.mission is None:
            return False
        distance_travelled = trip_meter_sensor()
        mission = plan.mission
        return mission.is_complete(vehicle_state, distance_travelled)

    @classmethod
    def _vehicle_is_off_road(
        cls,
        road_map: RoadMap,
        vehicle_state: VehicleState,
        nearest_lanes: Optional[Sequence["RoadMap.Lane"]] = None,
    ) -> bool:
        return not road_map.road_with_point(
            vehicle_state.pose.point, lanes_to_search=nearest_lanes
        )

    @classmethod
    def _vehicle_is_on_shoulder(
        cls,
        road_map: RoadMap,
        vehicle_state: VehicleState,
        nearest_lanes: Optional[Sequence["RoadMap.Lane"]] = None,
    ) -> bool:
        # XXX: this isn't technically right as this would also return True
        #      for vehicles that are completely off road.
        for corner_coordinate in vehicle_state.bounding_box_points:
            if not road_map.road_with_point(
                Point(*corner_coordinate), lanes_to_search=nearest_lanes
            ):
                return True
        return False

    @classmethod
    def _vehicle_is_not_moving(
        cls, sim, last_n_seconds_considered, min_distance_moved, driven_path_sensor
    ) -> bool:
        # Flag if the vehicle has been immobile for the past 'last_n_seconds_considered' seconds
        if sim.elapsed_sim_time < last_n_seconds_considered:
            return False

        distance = sys.maxsize
        if driven_path_sensor is not None:
            distance = driven_path_sensor.distance_travelled(
                sim.elapsed_sim_time, last_n_seconds=last_n_seconds_considered
            )

        # Due to controller instabilities there may be some movement even when a
        # vehicle is "stopped".
        return distance < min_distance_moved

    @classmethod
    def _vehicle_is_off_route_and_wrong_way(
        cls,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        vehicle_state: VehicleState,
        plan: Plan,
        nearest_lanes: Optional[Sequence[Tuple[RoadMap.Lane, float]]] = None,
    ):
        """Determines if the agent is on route and on the correct side of the road.

        Args:
            sim_frame: An instance of the simulator.
            sim_local_constants: The current frozen state of the simulation for last reset.
            vehicle_state: The current state of the vehicle to check.
            plan: The current plan for the vehicle.
            nearest_lanes: Cached result of nearest lanes and distances for the vehicle position.

        Returns:
            A tuple (is_off_route, is_wrong_way)
            is_off_route:
                Actor's vehicle is not on its route or an oncoming traffic lane.
            is_wrong_way:
                Actor's vehicle is going against the lane travel direction.
        """

        route_roads = plan.route.roads

        # No road nearby, so we're not on route!
        if not nearest_lanes:
            return (True, False)

        # Handle case where there are multiple nearest lanes the same dist away
        min_dist = nearest_lanes[0][1]
        tied_nearest = [
            lane for (lane, d) in nearest_lanes if math.isclose(d, min_dist)
        ]
        nearest_on_route = [lane for lane in tied_nearest if lane.road in route_roads]
        nearest_lane = nearest_on_route[0] if nearest_on_route else nearest_lanes[0][0]

        # Check whether vehicle is in wrong-way
        is_wrong_way = cls._check_wrong_way_event(nearest_lane, vehicle_state)

        # Check whether vehicle has no-route or is on-route
        if (
            not route_roads  # Vehicle has no-route. E.g., endless mission with a random route
            or nearest_lane.road in route_roads  # Vehicle is on-route
            or nearest_lane.in_junction
        ):
            return (False, is_wrong_way)

        vehicle_pos = vehicle_state.pose.point
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
    def _vehicle_is_wrong_way(
        vehicle_state: VehicleState, closest_lane: RoadMap.Lane
    ) -> bool:
        target_pose = closest_lane.center_pose_at_point(vehicle_state.pose.point)
        # Check if the vehicle heading is oriented away from the lane heading.
        return (
            np.fabs(vehicle_state.pose.heading.relative_to(target_pose.heading))
            > 0.5 * np.pi
        )

    @classmethod
    def _check_wrong_way_event(
        cls, lane_to_check: RoadMap.Lane, vehicle_state: VehicleState
    ) -> bool:
        # When the vehicle is in an intersection, turn off the `wrong way` check to avoid
        # false positive `wrong way` events.
        if lane_to_check.in_junction:
            return False
        return cls._vehicle_is_wrong_way(vehicle_state, lane_to_check)
