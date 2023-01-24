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
import importlib.resources as pkg_resources
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from smarts.core.agent_interface import AgentInterface
from smarts.core.plan import Mission, Plan

from . import models
from .actor import ActorRole, ActorState
from .chassis import AckermannChassis, BoxChassis, Chassis
from .colors import SceneColors
from .coordinates import Dimensions, Heading, Pose
from .sensors import (
    AccelerometerSensor,
    DrivableAreaGridMapSensor,
    DrivenPathSensor,
    LanePositionSensor,
    LidarSensor,
    NeighborhoodVehiclesSensor,
    OGMSensor,
    RGBSensor,
    RoadWaypointsSensor,
    SignalsSensor,
    TripMeterSensor,
    ViaSensor,
    WaypointsSensor,
)
from .utils.custom_exceptions import RendererException
from .utils.math import rotate_cw_around_point


@dataclass
class VehicleState(ActorState):
    """Vehicle state information."""

    vehicle_config_type: Optional[str] = None  # key into VEHICLE_CONFIGS
    pose: Optional[Pose] = None
    dimensions: Optional[Dimensions] = None
    speed: float = 0.0
    steering: Optional[float] = None
    yaw_rate: Optional[float] = None
    linear_velocity: Optional[np.ndarray] = None
    angular_velocity: Optional[np.ndarray] = None
    linear_acceleration: Optional[np.ndarray] = None
    angular_acceleration: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.pose is not None and self.dimensions is not None

    @property
    def bbox(self) -> Polygon:
        """Returns a bounding box around the vehicle."""
        pos = self.pose.point
        half_len = 0.5 * self.dimensions.length
        half_width = 0.5 * self.dimensions.width
        poly = shapely_box(
            pos.x - half_width,
            pos.y - half_len,
            pos.x + half_width,
            pos.y + half_len,
        )
        return shapely_rotate(poly, self.pose.heading, use_radians=True)


@dataclass(frozen=True)
class VehicleConfig:
    """Vehicle configuration"""

    vehicle_type: str
    color: SceneColors
    dimensions: Dimensions
    glb_model: str


# A mapping between SUMO's vehicle types and our internal vehicle config.
# TODO: Don't leak SUMO's types here.
# XXX: The GLB's dimensions must match the specified dimensions here.
# TODO: for traffic histories, vehicle (and road) dimensions are set by the dataset
VEHICLE_CONFIGS = {
    "passenger": VehicleConfig(
        vehicle_type="car",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=3.68, width=1.47, height=1.4),
        glb_model="simple_car.glb",
    ),
    "bus": VehicleConfig(
        vehicle_type="bus",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=7, width=2.25, height=3),
        glb_model="bus.glb",
    ),
    "coach": VehicleConfig(
        vehicle_type="coach",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=8, width=2.4, height=3.5),
        glb_model="coach.glb",
    ),
    "truck": VehicleConfig(
        vehicle_type="truck",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=5, width=1.91, height=1.89),
        glb_model="truck.glb",
    ),
    "trailer": VehicleConfig(
        vehicle_type="trailer",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=10, width=2.5, height=4),
        glb_model="trailer.glb",
    ),
    "pedestrian": VehicleConfig(
        vehicle_type="pedestrian",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=0.5, width=0.5, height=1.6),
        glb_model="pedestrian.glb",
    ),
    "motorcycle": VehicleConfig(
        vehicle_type="motorcycle",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=2.5, width=1, height=1.4),
        glb_model="motorcycle.glb",
    ),
}


class Vehicle:
    """Represents a single vehicle."""

    _HAS_DYNAMIC_ATTRIBUTES = True  # dynamic pytype attribute

    def __init__(
        self,
        id: str,
        chassis: Chassis,
        vehicle_config_type: str = "passenger",
        color: Optional[SceneColors] = None,
        action_space=None,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._id = id

        self._chassis: Chassis = chassis
        self._vehicle_config_type = vehicle_config_type
        self._action_space = action_space

        self._meta_create_sensor_functions()
        self._sensors = {}

        # Color override
        self._color: Optional[SceneColors] = color
        if self._color is None:
            config = VEHICLE_CONFIGS[vehicle_config_type]
            self._color = config.color

        self._renderer = None

        self._initialized = True
        self._has_stepped = False

    def _assert_initialized(self):
        assert self._initialized, f"Vehicle({self.id}) is not initialized"

    def __repr__(self):
        return f"""Vehicle({self.id},
  pose={self.pose},
  speed={self.speed},
  type={self.vehicle_type},
  w={self.width},
  l={self.length},
  h={self.height}
)"""

    @property
    def id(self):
        """The id of this vehicle."""
        return self._id

    @property
    def length(self) -> float:
        """The length of this vehicle."""
        self._assert_initialized()
        return self._chassis.dimensions.length

    @property
    def max_steering_wheel(self) -> Optional[float]:
        """The max steering value the chassis steering wheel can turn to.

        Some chassis types do not support this.
        """
        self._assert_initialized()
        return getattr(self._chassis, "max_steering_wheel", None)

    @property
    def width(self) -> float:
        """The width of this vehicle."""
        self._assert_initialized()
        return self._chassis.dimensions.width

    @property
    def height(self) -> float:
        """The height of this vehicle."""
        self._assert_initialized()
        return self._chassis.dimensions.height

    @property
    def speed(self) -> float:
        """The current speed of this vehicle."""
        self._assert_initialized()
        return self._chassis.speed

    @property
    def sensors(self) -> dict:
        """The sensors attached to this vehicle."""
        self._assert_initialized()
        return self._sensors

    @property
    def renderer(self):  # type: ignore
        """The renderer this vehicle is rendered to."""
        # Returns: Optional[Renderer]
        return self._renderer

    # # TODO: See issue #898 This is a currently a no-op
    # @speed.setter
    # def speed(self, speed):
    #     self._chassis.speed = speed

    @property
    def vehicle_color(self) -> Union[SceneColors, Tuple, None]:
        """The color of this vehicle (generally used for rendering purposes.)"""
        self._assert_initialized()
        return self._color

    @property
    def state(self) -> VehicleState:
        """The current state of this vehicle."""
        self._assert_initialized()
        return VehicleState(
            actor_id=self.id,
            actor_type=self.vehicle_type,
            source="SMARTS",  # this is the "ground truth" state
            vehicle_config_type=self._vehicle_config_type,
            pose=self.pose,
            dimensions=self._chassis.dimensions,
            speed=self.speed,
            # pytype: disable=attribute-error
            steering=self._chassis.steering,
            # pytype: enable=attribute-error
            yaw_rate=self._chassis.yaw_rate,
            linear_velocity=self._chassis.velocity_vectors[0],
            angular_velocity=self._chassis.velocity_vectors[1],
        )

    @property
    def action_space(self):
        """The action space this vehicle uses."""
        self._assert_initialized()
        return self._action_space

    @property
    def pose(self) -> Pose:
        """The pose of this vehicle. Pose is defined as position and orientation."""
        self._assert_initialized()
        return self._chassis.pose

    @property
    def chassis(self) -> Chassis:
        """The underlying chassis of this vehicle."""
        self._assert_initialized()
        return self._chassis

    @property
    def heading(self) -> Heading:
        """The heading of this vehicle.

        Note: Heading rotates counterclockwise with north as 0.
        """
        self._assert_initialized()
        return self._chassis.pose.heading

    @property
    def position(self) -> np.ndarray:
        """The position of this vehicle."""
        self._assert_initialized()
        return self._chassis.pose.position

    @property
    def bounding_box(self) -> List[np.ndarray]:
        """The minimum fitting heading aligned bounding box. Four 2D points representing the minimum fitting box."""
        # XXX: this doesn't return a smarts.core.coordinates.BoundingBox!
        self._assert_initialized()
        # Assuming the position is the centre,
        # calculate the corner coordinates of the bounding_box
        origin = self.position[:2]
        dimensions = np.array([self.width, self.length])
        corners = np.array([(-1, 1), (1, 1), (1, -1), (-1, -1)]) / 2
        heading = self.heading
        return [
            rotate_cw_around_point(
                point=origin + corner * dimensions,
                radians=Heading.flip_clockwise(heading),
                origin=origin,
            )
            for corner in corners
        ]

    @property
    def vehicle_type(self) -> str:
        """Get the vehicle type identifier."""
        return VEHICLE_CONFIGS[self._vehicle_config_type].vehicle_type

    @property
    def valid(self) -> bool:
        """Check if the vehicle still `exists` and is still operable."""
        return self._initialized

    @staticmethod
    def agent_vehicle_dims(mission: Mission) -> Dimensions:
        """Get the vehicle dimensions from the mission requirements.
        Args:
            A mission for the agent.
        Returns:
            The mission vehicle spec dimensions XOR the default "passenger" vehicle dimensions.
        """
        if mission.vehicle_spec:
            # mission.vehicle_spec.veh_config_type will always be "passenger" for now,
            # but we use that value here in case we ever expand our history functionality.
            vehicle_config_type = mission.vehicle_spec.veh_config_type
            return Dimensions.copy_with_defaults(
                mission.vehicle_spec.dimensions,
                VEHICLE_CONFIGS[vehicle_config_type].dimensions,
            )
        # non-history agents can currently only control passenger vehicles.
        vehicle_config_type = "passenger"
        return VEHICLE_CONFIGS[vehicle_config_type].dimensions

    @classmethod
    def build_agent_vehicle(
        cls,
        sim,
        vehicle_id: str,
        agent_interface: AgentInterface,
        plan: Plan,
        vehicle_filepath: str,
        tire_filepath: str,
        trainable: bool,
        surface_patches: List[Dict[str, Any]],
        initial_speed: Optional[float] = None,
    ) -> "Vehicle":
        """Create a new vehicle and set up sensors and planning information as required by the
        ego agent.
        """
        mission = plan.mission

        chassis_dims = cls.agent_vehicle_dims(mission)

        start = mission.start
        if start.from_front_bumper:
            start_pose = Pose.from_front_bumper(
                front_bumper_position=np.array(start.position[:2]),
                heading=start.heading,
                length=chassis_dims.length,
            )
        else:
            start_pose = Pose.from_center(start.position, start.heading)

        vehicle_color = (
            SceneColors.Agent.value if trainable else SceneColors.SocialAgent.value
        )

        if agent_interface.vehicle_type == "sedan":
            urdf_name = "vehicle"
        elif agent_interface.vehicle_type == "bus":
            urdf_name = "bus"
        else:
            raise Exception("Vehicle type does not exist!!!")

        if (vehicle_filepath is None) or not os.path.exists(vehicle_filepath):
            with pkg_resources.path(models, urdf_name + ".urdf") as path:
                vehicle_filepath = str(path.absolute())

        controller_parameters = sim.vehicle_index.controller_params_for_vehicle_type(
            agent_interface.vehicle_type
        )

        chassis = None
        if agent_interface and agent_interface.action in sim.dynamic_action_spaces:
            if mission.vehicle_spec:
                logger = logging.getLogger(cls.__name__)
                logger.warning(
                    "setting vehicle dimensions on a AckermannChassis not yet supported"
                )
            chassis = AckermannChassis(
                pose=start_pose,
                bullet_client=sim.bc,
                vehicle_filepath=vehicle_filepath,
                tire_parameters_filepath=tire_filepath,
                friction_map=surface_patches,
                controller_parameters=controller_parameters,
                initial_speed=initial_speed,
            )
        else:
            chassis = BoxChassis(
                pose=start_pose,
                speed=initial_speed,
                dimensions=chassis_dims,
                bullet_client=sim.bc,
            )

        vehicle = Vehicle(
            id=vehicle_id,
            chassis=chassis,
            color=vehicle_color,
        )

        return vehicle

    @staticmethod
    def build_social_vehicle(
        sim, vehicle_id, vehicle_state, vehicle_config_type
    ) -> "Vehicle":
        """Create a new unassociated vehicle."""
        dims = Dimensions.copy_with_defaults(
            vehicle_state.dimensions, VEHICLE_CONFIGS[vehicle_config_type].dimensions
        )
        chassis = BoxChassis(
            pose=vehicle_state.pose,
            speed=vehicle_state.speed,
            dimensions=dims,
            bullet_client=sim.bc,
        )
        return Vehicle(
            id=vehicle_id, chassis=chassis, vehicle_config_type=vehicle_config_type
        )

    @staticmethod
    def attach_sensors_to_vehicle(sim, vehicle, agent_interface, plan):
        """Attach sensors as required to satisfy the agent interface's requirements"""
        # The distance travelled sensor is not optional b/c it is used for the score
        # and reward calculation
        vehicle.attach_trip_meter_sensor(
            TripMeterSensor(
                vehicle=vehicle,
                road_map=sim.road_map,
                plan=plan,
            )
        )

        # The distance travelled sensor is not optional b/c it is used for visualization
        # done criteria
        vehicle.attach_driven_path_sensor(DrivenPathSensor(vehicle=vehicle))

        if agent_interface.neighborhood_vehicle_states:
            vehicle.attach_neighborhood_vehicle_states_sensor(
                NeighborhoodVehiclesSensor(
                    vehicle=vehicle,
                    sim=sim,
                    radius=agent_interface.neighborhood_vehicle_states.radius,
                )
            )

        if agent_interface.accelerometer:
            vehicle.attach_accelerometer_sensor(AccelerometerSensor(vehicle=vehicle))

        if agent_interface.lane_positions:
            vehicle.attach_lane_position_sensor(LanePositionSensor(vehicle=vehicle))

        if agent_interface.waypoint_paths:
            vehicle.attach_waypoints_sensor(
                WaypointsSensor(
                    vehicle=vehicle,
                    plan=plan,
                    lookahead=agent_interface.waypoint_paths.lookahead,
                )
            )

        if agent_interface.road_waypoints:
            vehicle.attach_road_waypoints_sensor(
                RoadWaypointsSensor(
                    vehicle=vehicle,
                    sim=sim,
                    plan=plan,
                    horizon=agent_interface.road_waypoints.horizon,
                )
            )

        if agent_interface.drivable_area_grid_map:
            if not sim.renderer:
                raise RendererException.required_to("add a drivable_area_grid_map")
            vehicle.attach_drivable_area_grid_map_sensor(
                DrivableAreaGridMapSensor(
                    vehicle=vehicle,
                    width=agent_interface.drivable_area_grid_map.width,
                    height=agent_interface.drivable_area_grid_map.height,
                    resolution=agent_interface.drivable_area_grid_map.resolution,
                    renderer=sim.renderer,
                )
            )
        if agent_interface.occupancy_grid_map:
            if not sim.renderer:
                raise RendererException.required_to("add an OGM")
            vehicle.attach_ogm_sensor(
                OGMSensor(
                    vehicle=vehicle,
                    width=agent_interface.occupancy_grid_map.width,
                    height=agent_interface.occupancy_grid_map.height,
                    resolution=agent_interface.occupancy_grid_map.resolution,
                    renderer=sim.renderer,
                )
            )
        if agent_interface.top_down_rgb:
            if not sim.renderer:
                raise RendererException.required_to("add an RGB camera")
            vehicle.attach_rgb_sensor(
                RGBSensor(
                    vehicle=vehicle,
                    width=agent_interface.top_down_rgb.width,
                    height=agent_interface.top_down_rgb.height,
                    resolution=agent_interface.top_down_rgb.resolution,
                    renderer=sim.renderer,
                )
            )
        if agent_interface.lidar_point_cloud:
            vehicle.attach_lidar_sensor(
                LidarSensor(
                    vehicle=vehicle,
                    bullet_client=sim.bc,
                    sensor_params=agent_interface.lidar_point_cloud.sensor_params,
                )
            )

        vehicle.attach_via_sensor(
            ViaSensor(
                vehicle=vehicle,
                plan=plan,
                lane_acquisition_range=40,
                speed_accuracy=1.5,
            )
        )

        if agent_interface.signals:
            lookahead = agent_interface.signals.lookahead
            vehicle.attach_signals_sensor(
                SignalsSensor(
                    vehicle=vehicle, road_map=sim.road_map, lookahead=lookahead
                )
            )

    def step(self, current_simulation_time):
        """Update internal state."""
        self._has_stepped = True
        self._chassis.step(current_simulation_time)

    def control(self, *args, **kwargs):
        """Apply control values to this vehicle.

        Forwards control to the chassis.
        """
        self._chassis.control(*args, **kwargs)

    def update_state(self, state: VehicleState, dt: float):
        """Update the vehicle's state"""
        state.updated = True
        if state.role != ActorRole.External:
            assert isinstance(self._chassis, BoxChassis)
            self.control(pose=state.pose, speed=state.speed, dt=dt)
            return
        # External actors are "privileged", which means they work directly (bypass force application).
        # Conceptually, this is playing 'god' with physics and should only be used
        # to defer to a co-simulator's states.
        linear_velocity, angular_velocity = None, None
        if not np.allclose(
            self._chassis.velocity_vectors[0], state.linear_velocity
        ) or not np.allclose(self._chassis.velocity_vectors[1], state.angular_velocity):
            linear_velocity = state.linear_velocity
            angular_velocity = state.angular_velocity
        if not state.dimensions.equal_if_defined(self.length, self.width, self.height):
            self._log.warning(
                "Unable to change a vehicle's dimensions via external_state_update()."
            )
        # XXX:  any way to update acceleration in pybullet?
        self._chassis.state_override(dt, state.pose, linear_velocity, angular_velocity)

    def create_renderer_node(self, renderer):
        """Create the vehicle's rendering node in the renderer."""
        assert not self._renderer
        self._renderer = renderer
        config = VEHICLE_CONFIGS[self._vehicle_config_type]
        self._renderer.create_vehicle_node(
            config.glb_model, self._id, self.vehicle_color, self.pose
        )

    def sync_to_renderer(self):
        """Update the vehicle's rendering node."""
        assert self._renderer
        self._renderer.update_vehicle_node(self._id, self.pose)

    @lru_cache(maxsize=1)
    def _warn_AckermannChassis_set_pose(self):
        if self._has_stepped and isinstance(self._chassis, AckermannChassis):
            logging.warning(
                f"Agent `{self._id}` has called set pose after step."
                "This may cause collision problems"
            )

    # TODO: Merge this w/ speed setter as a set GCD call?
    def set_pose(self, pose: Pose):
        """Use with caution. This will directly set the pose of the chassis.

        This may disrupt physics simulation of the chassis physics body for a few steps after use.
        """
        self._warn_AckermannChassis_set_pose()
        self._chassis.set_pose(pose)

    def swap_chassis(self, chassis: Chassis):
        """Swap the current chassis with the given chassis. Apply the GCD of the previous chassis
        to the new chassis ("greatest common denominator state" from front-end to back-end)
        """
        chassis.inherit_physical_values(self._chassis)
        self._chassis.teardown()
        self._chassis = chassis

    def teardown(self, exclude_chassis=False):
        """Clean up internal resources"""
        for sensor in [
            sensor
            for sensor in [
                self._drivable_area_grid_map_sensor,
                self._ogm_sensor,
                self._rgb_sensor,
                self._lidar_sensor,
                self._driven_path_sensor,
            ]
            if sensor is not None
        ]:
            sensor.teardown()

        if not exclude_chassis:
            self._chassis.teardown()
        if self._renderer:
            self._renderer.remove_vehicle_node(self._id)
        self._initialized = False

    def _meta_create_sensor_functions(self):
        # Bit of metaprogramming to make sensor creation more DRY
        sensor_names = [
            "ogm_sensor",
            "rgb_sensor",
            "lidar_sensor",
            "driven_path_sensor",
            "trip_meter_sensor",
            "drivable_area_grid_map_sensor",
            "neighborhood_vehicle_states_sensor",
            "waypoints_sensor",
            "road_waypoints_sensor",
            "accelerometer_sensor",
            "lane_position_sensor",
            "via_sensor",
            "signals_sensor",
        ]
        for sensor_name in sensor_names:

            def attach_sensor(self, sensor, sensor_name=sensor_name):
                # replace previously-attached sensor with this one
                # (to allow updating its parameters).
                # Sensors might have been attached to a non-agent vehicle
                # (for example, for observation collection from history vehicles),
                # but if that vehicle gets hijacked, we want to use the sensors
                # specified by the hijacking agent's interface.
                detach = getattr(self, f"detach_{sensor_name}")
                if detach:
                    detach(sensor_name)
                    self._log.info(
                        f"replacing existing {sensor_name} on vehicle {self.id}"
                    )
                setattr(self, f"_{sensor_name}", sensor)
                self._sensors[sensor_name] = sensor

            def detach_sensor(self, sensor_name=sensor_name):
                sensor = getattr(self, f"_{sensor_name}", None)
                if sensor is not None:
                    sensor.teardown()
                    setattr(self, f"_{sensor_name}", None)
                    del self._sensors[sensor_name]

            def subscribed_to(self, sensor_name=sensor_name):
                sensor = getattr(self, f"_{sensor_name}", None)
                return sensor is not None

            def sensor_property(self, sensor_name=sensor_name):
                sensor = getattr(self, f"_{sensor_name}", None)
                assert sensor is not None, f"{sensor_name} is not attached to {self.id}"
                return sensor

            setattr(Vehicle, f"_{sensor_name}", None)
            setattr(Vehicle, f"attach_{sensor_name}", attach_sensor)
            setattr(Vehicle, f"detach_{sensor_name}", detach_sensor)
            setattr(Vehicle, f"subscribed_to_{sensor_name}", property(subscribed_to))
            setattr(Vehicle, f"{sensor_name}", property(sensor_property))

        def detach_all_sensors_from_vehicle(vehicle):
            for sensor_name in sensor_names:
                detach_sensor_func = getattr(vehicle, f"detach_{sensor_name}")
                detach_sensor_func()

        setattr(
            Vehicle,
            "detach_all_sensors_from_vehicle",
            staticmethod(detach_all_sensors_from_vehicle),
        )
