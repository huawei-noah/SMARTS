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
import importlib.resources as pkg_resources
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy
import yaml
from direct.showbase.ShowBase import ShowBase

from smarts.sstudio.types import UTurn

from . import models
from .chassis import AckermannChassis, BoxChassis, Chassis
from .colors import SceneColors
from .coordinates import BoundingBox, Heading, Pose
from .masks import RenderMasks
from .sensors import (
    AccelerometerSensor,
    DrivableAreaGridMapSensor,
    DrivenPathSensor,
    LidarSensor,
    NeighborhoodVehiclesSensor,
    OGMSensor,
    RGBSensor,
    RoadWaypointsSensor,
    TripMeterSensor,
    ViaSensor,
    WaypointsSensor,
)
from .utils.math import rotate_around_point


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: str
    vehicle_type: str
    pose: Pose
    dimensions: BoundingBox
    speed: float = 0
    steering: float = None
    yaw_rate: float = None
    source: str = None  # the source of truth for this vehicle state
    linear_velocity: numpy.ndarray = None
    angular_velocity: numpy.ndarray = None


@dataclass(frozen=True)
class VehicleConfig:
    vehicle_type: str
    color: tuple
    dimensions: BoundingBox
    glb_model: str


# A mapping between SUMO's vehicle types and our internal vehicle config.
# TODO: Don't leak SUMO's types here.
# XXX: The GLB's dimensions must match the specified dimensions here.
VEHICLE_CONFIGS = {
    "passenger": VehicleConfig(
        vehicle_type="car",
        color=SceneColors.SocialVehicle.value,
        dimensions=BoundingBox(length=3.68, width=1.47, height=1.4),
        glb_model="simple_car.glb",
    ),
    "bus": VehicleConfig(
        vehicle_type="bus",
        color=SceneColors.SocialVehicle.value,
        dimensions=BoundingBox(length=7, width=2.25, height=3),
        glb_model="bus.glb",
    ),
    "coach": VehicleConfig(
        vehicle_type="coach",
        color=SceneColors.SocialVehicle.value,
        dimensions=BoundingBox(length=8, width=2.4, height=3.5),
        glb_model="coach.glb",
    ),
    "truck": VehicleConfig(
        vehicle_type="truck",
        color=SceneColors.SocialVehicle.value,
        dimensions=BoundingBox(length=5, width=1.91, height=1.89),
        glb_model="truck.glb",
    ),
    "trailer": VehicleConfig(
        vehicle_type="trailer",
        color=SceneColors.SocialVehicle.value,
        dimensions=BoundingBox(length=10, width=2.5, height=4),
        glb_model="trailer.glb",
    ),
}

# TODO: Replace VehicleConfigs w/ the VehicleGeometry class
class VehicleGeometry:
    @classmethod
    def fromfile(cls, path, color):
        pass


class Vehicle:
    def __init__(
        self,
        id: str,
        pose: Pose,
        showbase: ShowBase,
        chassis: Chassis,
        # TODO: We should not be leaking SUMO here.
        sumo_vehicle_type="passenger",
        color=None,
        action_space=None,
    ):
        assert isinstance(pose, Pose)

        self._log = logging.getLogger(self.__class__.__name__)
        self._id = id

        self._chassis = chassis
        self._showbase = showbase
        self._sumo_vehicle_type = sumo_vehicle_type
        self._action_space = action_space
        self._speed = None

        self._meta_create_sensor_functions()
        self._sensors = {}

        config = VEHICLE_CONFIGS[sumo_vehicle_type]

        # Color override
        self._color = color
        if self._color is None:
            self._color = config.color

        # TODO: Move this into the VehicleGeometry class
        self._np = self._build_model(pose, config, showbase)
        self._initialized = True
        self._has_stepped = False

    def _assert_initialized(self):
        assert self._initialized, f"Vehicle({self.id}) is not initialized"

    def _build_model(self, pose: Pose, config: VehicleConfig, showbase):
        with pkg_resources.path(models, config.glb_model) as path:
            node_path = showbase.loader.loadModel(str(path.absolute()))

        node_path.setName("vehicle-%s" % self._id)
        node_path.setColor(self._color)
        pos, heading = pose.as_panda3d()
        node_path.setPosHpr(*pos, heading, 0, 0)
        node_path.hide(RenderMasks.DRIVABLE_AREA_HIDE)

        return node_path

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
        return self._id

    @property
    def length(self):
        self._assert_initialized()
        return self._chassis.dimensions.length

    @property
    def max_steering_wheel(self):
        self._assert_initialized()
        return self._chassis.max_steering_wheel

    @property
    def width(self):
        self._assert_initialized()
        return self._chassis.dimensions.width

    @property
    def height(self):
        self._assert_initialized()
        return self._chassis.dimensions.height

    @property
    def speed(self):
        self._assert_initialized()
        if self._speed is not None:
            return self._speed
        else:
            return self._chassis.speed

    def set_speed(self, speed):
        self._speed = speed

    @property
    def sensors(self):
        self._assert_initialized()
        return self._sensors

    # # TODO: See issue #898 This is a currently a no-op
    # @speed.setter
    # def speed(self, speed):
    #     self._chassis.speed = speed

    @property
    def np(self):
        self._assert_initialized()
        return self._np

    @property
    def vehicle_color(self):
        self._assert_initialized()
        return self._color

    @property
    def state(self):
        self._assert_initialized()
        return VehicleState(
            vehicle_id=self.id,
            vehicle_type=self.vehicle_type,
            pose=self.pose,
            dimensions=self._chassis.dimensions,
            speed=self.speed,
            steering=self._chassis.steering,
            yaw_rate=self._chassis.yaw_rate,
            source="SMARTS",
            linear_velocity=self._chassis.velocity_vectors[0],
            angular_velocity=self._chassis.velocity_vectors[1],
        )

    @property
    def action_space(self):
        self._assert_initialized()
        return self._action_space

    @property
    def pose(self) -> Pose:
        self._assert_initialized()
        return self._chassis.pose

    @property
    def chassis(self):
        self._assert_initialized()
        return self._chassis

    @property
    def heading(self) -> Heading:
        self._assert_initialized()
        return self._chassis.pose.heading

    @property
    def position(self):
        self._assert_initialized()
        pos, _ = self._chassis.pose.as_panda3d()
        return pos

    @property
    def bounding_box(self):
        self._assert_initialized()
        # Assuming the position is the centre,
        # calculate the corner coordinates of the bounding_box
        half_length = self.length / 2
        half_width = self.width / 2
        corners = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
        return [
            rotate_around_point(
                point=self.position[:2]
                + numpy.array(corner) * numpy.array([half_width, half_length]),
                radians=self.heading,
                origin=self.position[:2],
            )
            for corner in corners
        ]

    @property
    def vehicle_type(self):
        return VEHICLE_CONFIGS[self._sumo_vehicle_type].vehicle_type

    @staticmethod
    def build_agent_vehicle(
        sim,
        vehicle_id,
        agent_interface,
        mission_planner,
        vehicle_filepath,
        tire_filepath,
        trainable,
        surface_patches,
        controller_filepath,
        initial_speed=None,
    ):
        # Agents can currently only control passenger vehicles
        vehicle_type = "passenger"
        chassis_dims = VEHICLE_CONFIGS[vehicle_type].dimensions

        if isinstance(mission_planner.mission.task, UTurn):
            if mission_planner.mission.task.initial_speed:
                initial_speed = mission_planner.mission.task.initial_speed

        start = mission_planner.mission.start
        start_pose = Pose.from_front_bumper(
            front_bumper_position=numpy.array(start.position),
            heading=start.heading,
            length=chassis_dims.length,
        )

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

        if (controller_filepath is None) or not os.path.exists(controller_filepath):
            with pkg_resources.path(
                models, "controller_parameters.yaml"
            ) as controller_path:
                controller_filepath = str(controller_path.absolute())
        with open(controller_filepath, "r") as controller_file:
            controller_parameters = yaml.safe_load(controller_file)[
                agent_interface.vehicle_type
            ]

        chassis = None
        # change this to dynamic_action_spaces later when pr merged
        if agent_interface and agent_interface.action in sim.dynamic_action_spaces:
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
            pose=start_pose,
            showbase=sim,
            chassis=chassis,
            color=vehicle_color,
        )

        return vehicle

    @staticmethod
    def build_social_vehicle(sim, vehicle_id, vehicle_state, vehicle_type):
        return Vehicle(
            id=vehicle_id,
            pose=vehicle_state.pose,
            showbase=sim,
            chassis=BoxChassis(
                pose=vehicle_state.pose,
                speed=vehicle_state.speed,
                dimensions=vehicle_state.dimensions,
                bullet_client=sim.bc,
            ),
            sumo_vehicle_type=vehicle_type,
        )

    @staticmethod
    def attach_sensors_to_vehicle(sim, vehicle, agent_interface, mission_planner):
        # The distance travelled sensor is not optional b/c it is used for the score
        # and reward calculation
        vehicle.attach_trip_meter_sensor(
            TripMeterSensor(
                vehicle=vehicle,
                sim=sim,
                mission_planner=mission_planner,
            )
        )

        # The distance travelled sensor is not optional b/c it is used for visualization
        # done criteria
        vehicle.attach_driven_path_sensor(DrivenPathSensor(vehicle=vehicle))

        if agent_interface.neighborhood_vehicles:
            vehicle.attach_neighborhood_vehicles_sensor(
                NeighborhoodVehiclesSensor(
                    vehicle=vehicle,
                    sim=sim,
                    radius=agent_interface.neighborhood_vehicles.radius,
                )
            )

        if agent_interface.accelerometer:
            vehicle.attach_accelerometer_sensor(
                AccelerometerSensor(
                    vehicle=vehicle,
                    sim=sim,
                )
            )

        if agent_interface.waypoints:
            vehicle.attach_waypoints_sensor(
                WaypointsSensor(
                    sim=sim,
                    vehicle=vehicle,
                    mission_planner=mission_planner,
                    lookahead=agent_interface.waypoints.lookahead,
                )
            )

        if agent_interface.road_waypoints:
            vehicle.attach_road_waypoints_sensor(
                RoadWaypointsSensor(
                    vehicle=vehicle,
                    sim=sim,
                    mission_planner=mission_planner,
                    horizon=agent_interface.road_waypoints.horizon,
                )
            )

        if agent_interface.drivable_area_grid_map:
            vehicle.attach_drivable_area_grid_map_sensor(
                DrivableAreaGridMapSensor(
                    vehicle=vehicle,
                    width=agent_interface.drivable_area_grid_map.width,
                    height=agent_interface.drivable_area_grid_map.height,
                    resolution=agent_interface.drivable_area_grid_map.resolution,
                    scene_np=sim.np,
                    showbase=sim,
                )
            )
        if agent_interface.ogm:
            vehicle.attach_ogm_sensor(
                OGMSensor(
                    vehicle=vehicle,
                    width=agent_interface.ogm.width,
                    height=agent_interface.ogm.height,
                    resolution=agent_interface.ogm.resolution,
                    scene_np=sim.np,
                    showbase=sim,
                )
            )
        if agent_interface.rgb:
            vehicle.attach_rgb_sensor(
                RGBSensor(
                    vehicle=vehicle,
                    width=agent_interface.rgb.width,
                    height=agent_interface.rgb.height,
                    resolution=agent_interface.rgb.resolution,
                    scene_np=sim.np,
                    showbase=sim,
                )
            )
        if agent_interface.lidar:
            vehicle.attach_lidar_sensor(
                LidarSensor(
                    vehicle=vehicle,
                    bullet_client=sim.bc,
                    showbase=sim,
                    sensor_params=agent_interface.lidar.sensor_params,
                )
            )

        vehicle.attach_via_sensor(
            ViaSensor(
                vehicle=vehicle,
                mission_planner=mission_planner,
                lane_acquisition_range=40,
                speed_accuracy=1.5,
            )
        )

    def step(self, current_simulation_time):
        self._has_stepped = True
        self._chassis.step(current_simulation_time)

    def control(self, *args, **kwargs):
        self._chassis.control(*args, **kwargs)

    def sync_to_panda3d(self):
        pos, heading = self._chassis.pose.as_panda3d()
        self._np.setPosHpr(*pos, heading, 0, 0)

    @lru_cache(maxsize=1)
    def _warn_AckermannChassis_set_pose(self):
        if self._has_stepped and isinstance(self._chassis, AckermannChassis):
            logging.warning(
                f"Agent `{self._id}` has called set pose after step."
                "This may cause collision problems"
            )

    # TODO: Merge this w/ speed setter as a set GCD call?
    def set_pose(self, pose: Pose):
        self._warn_AckermannChassis_set_pose()
        self._chassis.set_pose(pose)

    def swap_chassis(self, chassis: Chassis):
        # Apply GCD ("greatest common denominator state" from front-end to back-end)
        chassis.inherit_physical_values(self._chassis)
        self._chassis.teardown()
        self._chassis = chassis

    def teardown(self, exclude_chassis=False):
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
        self._np.removeNode()
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
            "neighborhood_vehicles_sensor",
            "mission_planner_sensor",
            "waypoints_sensor",
            "road_waypoints_sensor",
            "accelerometer_sensor",
            "via_sensor",
        ]
        for sensor_name in sensor_names:

            def attach_sensor(self, sensor, sensor_name=sensor_name):
                assert (
                    getattr(self, f"_{sensor_name}", None) is None
                ), f"{sensor_name} already added"
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
                assert sensor is not None, f"{sensor_name} is not attached"
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
