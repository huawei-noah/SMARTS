#!/usr/bin/env python3

import os
import json
import rospy
import sys
from smarts_ros.msg import (
    AgentReport,
    AgentSpec,
    AgentsStamped,
    EntitiesStamped,
    EntityState,
    SmartsControl,
)
from smarts_ros.srv import SmartsInfo, SmartsInfoResponse, SmartsInfoRequest

from collections import deque
import numpy as np
import time
from threading import Lock
from typing import Any, Dict, List, Sequence

from envision.client import Client as Envision
from smarts.core.agent import Agent
from smarts.core.coordinates import BoundingBox, Heading, Pose
from smarts.core.scenario import (
    default_entry_tactic,
    Mission,
    PositionalGoal,
    Scenario,
    Start,
    VehicleSpec,
)
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.vehicle import VehicleState
from smarts.zoo import registry


# Don't expect SMARTS to be able to reliably maintain rates faster than this!
SMARTS_MAX_FREQ = 60.0


class ROSDriver:
    """Wraps SMARTS as a ROS (v1) node.
    See the README.md in `examples/ros` for
    instructions on how to setup and run."""

    def __init__(self):
        self._state_lock = Lock()
        self._control_lock = Lock()
        self._smarts = None
        self._reset()

    def _reset(self):
        if self._smarts:
            self._smarts.destroy()
        self._smarts = None
        self._target_freq = None
        self._state_topic = None
        self._state_publisher = None
        self._agents_publisher = None
        self._most_recent_state_sent = None
        with self._state_lock:
            self._recent_state = deque(maxlen=3)
        with self._control_lock:
            self._scenario_path = None
            self._reset_smarts = False
            self._agents = {}
            self._agents_to_add = {}

    def setup_ros(
        self,
        node_name: str = "SMARTS",
        def_namespace: str = "SMARTS/",
        buffer_size: int = 3,
        target_freq: float = None,
        time_ratio: float = 1.0,
        pub_queue_size: int = 10,
    ):
        assert not self._state_publisher
        # enforce only one SMARTS instance per ROS network...
        # NOTE: The node name specified here may be overridden by ROS
        # remapping arguments from the command line invocation.
        rospy.init_node(node_name, anonymous=False)

        # If the namespace is aready set in the environment, we use it,
        # otherwise we use our default.
        namespace = def_namespace if not os.environ.get("ROS_NAMESPACE") else ""

        self._state_publisher = rospy.Publisher(
            f"{namespace}entities_out", EntitiesStamped, queue_size=pub_queue_size
        )
        self._agents_publisher = rospy.Publisher(
            f"{namespace}agents_out", AgentsStamped, queue_size=pub_queue_size
        )

        rospy.Service(f"{namespace}{node_name}_info", SmartsInfo, self._get_smarts_info)

        rospy.Subscriber(
            f"{namespace}control", SmartsControl, self._smarts_control_callback
        )

        self._state_topic = f"{namespace}entities_in"
        rospy.Subscriber(self._state_topic, EntitiesStamped, self._entities_callback)

        rospy.Subscriber(f"{namespace}agent_spec", AgentSpec, self._agent_spec_callback)

        buffer_size = rospy.get_param("~buffer_size", buffer_size)
        if buffer_size and buffer_size != self._recent_state.maxlen:
            assert buffer_size > 0
            self._recent_state = deque(maxlen=buffer_size)

        # If target_freq is not specified, SMARTS is allowed to
        # run as quickly as it can with no delay between steps.
        target_freq = rospy.get_param("~target_freq", target_freq)
        if target_freq:
            assert target_freq > 0.0
            if target_freq > SMARTS_MAX_FREQ:
                rospy.logwarn(
                    f"specified target frequency of {target_freq} Hz cannot be guaranteed by SMARTS."
                )
            self._target_freq = target_freq

    def setup_smarts(
        self, headless: bool = True, seed: int = 42, time_ratio: float = 1.0
    ):
        assert not self._smarts
        if not self._state_publisher:
            raise Exception("must call setup_ros() first.")

        headless = rospy.get_param("~headless", headless)
        seed = rospy.get_param("~seed", seed)
        time_ratio = rospy.get_param("~time_ratio", time_ratio)
        assert time_ratio > 0.0
        self._time_ratio = time_ratio

        self._smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            fixed_timestep_sec=None,
            envision=None if headless else Envision(),
            external_provider=True,
        )
        assert self._smarts.external_provider
        self._last_step_time = None
        with self._control_lock:
            self._scenario_path = None
            self._reset_smarts = False
            self._agents = {}
            self._agents_to_add = {}

    def _smarts_control_callback(self, control: SmartsControl):
        with self._control_lock:
            self._scenario_path = control.reset_with_scenario_path
            self._reset_smarts = True
            self._agents = {}
            self._agents_to_add = {}
        for ros_agent_spec in control.initial_agents:
            self._agent_spec_callback(ros_agent_spec)

    def _get_smarts_info(self, req: SmartsInfoRequest) -> SmartsInfoResponse:
        resp = SmartsInfoResponse()
        resp.header.stamp = rospy.Time.now()
        if not self._smarts:
            rospy.logwarn("get_smarts_info() called before SMARTS set up.")
            return resp
        resp.version = self._smarts.version
        resp.step_count = self._smarts.step_count
        resp.elapsed_sim_time = self._smarts.elapsed_sim_time
        if self._smarts.scenario:
            resp.current_scenario_path = self._smarts.scenario.root_filepath
        return resp

    def _entities_callback(self, entities: EntitiesStamped):
        # note: push/pop is thread safe on a deque but
        # in our smoothing we are accessing all elements
        # so we still need to protect it.
        with self._state_lock:
            self._recent_state.append(entities)

    def _decode_entity_type(self, entity_type: int) -> str:
        if entity_type == EntityState.ENTITY_TYPE_CAR:
            return "passenger"
        if entity_type == EntityState.ENTITY_TYPE_TRUCK:
            return "truck"
        if entity_type == EntityState.ENTITY_TYPE_TRAILER:
            return "trailer"
        if entity_type == EntityState.ENTITY_TYPE_BUS:
            return "bus"
        if entity_type == EntityState.ENTITY_TYPE_COACH:
            return "coach"
        if entity_type == EntityState.ENTITY_TYPE_PEDESTRIAN:
            return "pedestrian"
        if entity_type == EntityState.ENTITY_TYPE_MOTORCYCLE:
            return "motorcycle"
        if entity_type == EntityState.ENTITY_TYPE_UNSPECIFIED:
            return "passenger"
        rospy.logwarn(
            f"unsupported entity_type {entity_type}. defaulting to passenger car."
        )
        return "passenger"

    def _encode_entity_type(self, entity_type: str) -> int:
        if entity_type in ["passenger", "car"]:
            return EntityState.ENTITY_TYPE_CAR
        if entity_type == "truck":
            return EntityState.ENTITY_TYPE_TRUCK
        if entity_type == "trailer":
            return EntityState.ENTITY_TYPE_TRAILER
        if entity_type == "bus":
            return EntityState.ENTITY_TYPE_BUS
        if entity_type == "coach":
            return EntityState.ENTITY_TYPE_COACH
        if entity_type == "pedestrian":
            return EntityState.ENTITY_TYPE_PEDESTRIAN
        if entity_type == "motorcycle":
            return EntityState.ENTITY_TYPE_MOTORCYCLE
        if entity_type is None:
            return EntityState.ENTITY_TYPE_UNSPECIFIED
        rospy.logwarn(f"unsupported entity_type {entity_type}. defaulting to 'car'.")
        return EntityState.ENTITY_TYPE_CAR

    def _decode_vehicle_type(self, vehicle_type: int) -> str:
        if vehicle_type == AgentSpec.VEHICLE_TYPE_CAR:
            return "passenger"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_TRUCK:
            return "truck"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_TRAILER:
            return "trailer"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_BUS:
            return "bus"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_COACH:
            return "coach"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_PEDESTRIAN:
            return "pedestrian"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_MOTORCYCLE:
            return "motorcycle"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_UNSPECIFIED:
            return "passenger"
        rospy.logwarn(
            f"unsupported vehicle_type {vehicle_type}. defaulting to passenger car."
        )
        return "passenger"

    def _agent_spec_callback(self, ros_agent_spec: AgentSpec):
        agent_params = (
            json.loads(ros_agent_spec.params_json) if ros_agent_spec.params_json else {}
        )
        agent_spec = registry.make(ros_agent_spec.agent_type, **agent_params)
        if not agent_spec:
            rospy.logwarn(
                f"got unknown agent_type '{ros_agent_spec.agent_type}' in AgentSpec message with params='{ros_agent_spec.param_json}'.  ignoring."
            )
        mission = Mission(
            start=Start.from_pose(Pose.from_ros(ros_agent_spec.start_pose)),
            goal=PositionGoal(ros_agent_spec.end_pose[:2], ros_agent_spec.veh_length),
            # via=TODO(ros_agent_spec.vias),
            # TODO:  how to prevent them from spawning on top of another existing vehicle? (see how it's done in SUMO traffic)
            entry_tactic=default_entry_tactic(ros_agent_spec.start_speed),
            vehicle_spec=VehicleSpec(
                veh_id=f"veh_for_agent_{ros_agent_spec.agent_id}",
                veh_type=self._decode_vehicle_type(ros_agent_spec.veh_type),
                dimensions=BoundingBox(
                    ros_agent_spec.veh_length,
                    ros_agent_spec.veh_width,
                    ros_agent_spec.veh_height,
                ),
            ),
        )
        with self._control_lock:
            if (
                ros_agent_spec.agent_id in self._agents
                or ros_agent_spec.agent_id in self._agents_to_add
            ):
                rospy.logwarn(
                    f"trying to add new agent with existing agent_id '{ros_agent_spec.agent_id}'.  ignoring."
                )
                return
            self._agents_to_add[ros_agent_spec.agent_id] = (agent_spec, mission)

    @staticmethod
    def _get_nested_attr(obj: Any, dotname: str) -> Any:
        props = dotname.split(".")
        for prop in props:
            obj = getattr(obj, prop)
        return obj

    @staticmethod
    def _extrapolate_to_now(
        state_name: str, states: Sequence[float], veh_id: str, staleness: float
    ) -> np.ndarray:
        # Here, we just do some straightforward/basic smoothing using a moving average,
        # and then extrapolate to the current time.
        dt = 0.0
        last_time = None
        avg = np.array((0.0, 0.0, 0.0, 0.0))
        for state in states:
            for entity in state.entities:
                if entity.entity_id != veh_id:
                    continue
                vector = ROSDriver._get_nested_attr(entity, state_name)
                assert vector
                if hasattr(vector, "w"):
                    vector = np.array((vector.x, vector.y, vector.z, vector.w))
                else:
                    vector = np.array((vector.x, vector.y, vector.z))
                avg[: len(vector)] += vector
                stamp = state.header.stamp.to_sec()
                if last_time:
                    dt += stamp - last_time
                last_time = stamp
                last_vect = vector
                break
        avg /= len(states)
        return last_vect + staleness * 2 * (last_vect - avg[: len(last_vect)]) / dt

    def _update_smarts_state(self, step_delta: float) -> bool:
        with self._state_lock:
            if (
                not self._recent_state
                or self._most_recent_state_sent == self._recent_state[-1]
            ):
                rospy.logdebug(
                    f"No messages received on topic {self._state_topic} yet to send to SMARTS."
                )
                return False
            states = [s for s in self._recent_state]

        entities = []
        most_recent_state = states[-1]
        staleness = (rospy.get_rostime() - most_recent_state.header.stamp).to_sec()
        for entity in most_recent_state.entities:
            veh_id = entity.entity_id
            veh_type = self._decode_entity_type(entity.entity_type)
            veh_dims = BoundingBox(entity.length, entity.width, entity.height)

            pos = ROSDriver._extrapolate_to_now(
                "pose.position", states, veh_id, staleness
            )
            orientation = ROSDriver._extrapolate_to_now(
                "pose.orientation", states, veh_id, staleness
            )
            lin_vel = ROSDriver._extrapolate_to_now(
                "velocity.linear", states, veh_id, staleness
            )
            ang_vel = ROSDriver._extrapolate_to_now(
                "velocity.angular", states, veh_id, staleness
            )
            lin_acc = ROSDriver._extrapolate_to_now(
                "acceleration.linear", states, veh_id, staleness
            )
            ang_acc = ROSDriver._extrapolate_to_now(
                "acceleration.angular", states, veh_id, staleness
            )

            entities.append(
                VehicleState(
                    source="EXTERNAL",
                    vehicle_id=veh_id,
                    vehicle_config_type=veh_type,
                    privileged=True,
                    pose=Pose(pos, orientation),
                    dimensions=veh_dims,
                    speed=np.linalg.norm(lin_vel),
                    linear_velocity=lin_vel,
                    angular_velocity=ang_vel,
                    linear_acceleration=lin_acc,
                    angular_acceleration=ang_acc,
                )
            )

        rospy.logdebug(
            f"sending state to SMARTS w/ step_delta={step_delta}, staleness={staleness}..."
        )
        self._smarts.external_provider.state_update(entities, step_delta)
        self._most_recent_state_sent = most_recent_state
        return True

    @staticmethod
    def _vector_to_xyz(v, xyz):
        xyz.x, xyz.y, xyz.z = v[0], v[1], v[2]

    @staticmethod
    def _vector_to_xyzw(v, xyzw):
        xyzw.x, xyzw.y, xyzw.z, xyzw.w = v[0], v[1], v[2], v[3]

    def _publish_state(self):
        smarts_state = self._smarts.external_provider.all_vehicle_states
        entities = EntitiesStamped()
        entities.header.stamp = rospy.Time.now()
        for vehicle in smarts_state:
            entity = EntityState()
            entity.entity_id = vehicle.vehicle_id
            entity.entity_type = self._encode_entity_type(vehicle.vehicle_type)
            entity.length = vehicle.dimensions.length
            entity.width = vehicle.dimensions.width
            entity.height = vehicle.dimensions.height
            ROSDriver._vector_to_xyz(vehicle.pose.position, entity.pose.position)
            ROSDriver._vector_to_xyzw(vehicle.pose.orientation, entity.pose.orientation)
            ROSDriver._vector_to_xyz(vehicle.linear_velocity, entity.velocity.linear)
            ROSDriver._vector_to_xyz(vehicle.angular_velocity, entity.velocity.angular)
            if vehicle.linear_acceleration:
                ROSDriver._vector_to_xyz(
                    vehicle.linear_acceleration, entity.acceleration.linear
                )
            if vehicle.angular_acceleration:
                ROSDriver._vector_to_xyz(
                    vehicle.angular_acceleration, entity.acceleration.angular
                )
            entities.entities.append(entity)
        self._state_publisher.publish(entities)

    def _publish_agents(
        self, observations: Dict[str, Observation], dones: Dict[str, bool]
    ):
        agents = AgentsStamped()
        agents.header.stamp = rospy.Time.now()
        for agent_id, agent_obs in observations.items():
            veh_state = agent_ob.ego_vehicle_state
            agent = AgentReport()
            agent.agent_id = agent_id
            agent.pose = Pose.from_center(veh_state.position, veh_state.heading)
            agent.speed = veh_state.speed
            agent.distance_travelled = agent_obs.distance_travelled
            agent.is_done = dones[agent_id]
            agent.reached_goal = agent_obs.events.reached_goal
            agent.did_collide = bool(agent_obs.events.collisions)
            agent.is_wrong_way = agent_obs.events.wrong_way
            agent.is_off_route = agent_obs.events.off_route
            agent.is_off_road = agent_obs.events.off_road
            agent.is_on_shoulder = agent_obs.events.on_shoulder
            agent.is_not_moving = agent_obs.events.not_moving
            agents.append(agent)
        self._agents_publisher.publish(agents)

    def _do_agents(self, observations: Dict[str, Observation]) -> Dict[str, Any]:
        with self._control_lock:
            actions = {
                agent_id: self._agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            for agent_id, agent in self._agents_to_add.items():
                spec, mission = agent[0], agent[1]
                assert agent_id not in self._agents
                self._agents[agent_id] = spec.build_agent()
                self._smarts.add_agent_with_mission(
                    agent_id, spec.agent_interface, mission
                )
            self._agents_to_add = {}
            return actions

    def _check_reset(self) -> Dict[str, Observation]:
        with self._control_lock:
            if self._reset_smarts:
                rospy.loginfo(f"resetting SMARTS w/ scenario={self._scenario_path}")
                self._reset_smarts = False
                if self._scenario_path:
                    observations = self._smarts.reset(Scenario(self._scenario_path))
                    self._last_step_time = None
                    return observations
                return {}
        return None

    def run_forever(self):
        if not self._state_publisher:
            raise Exception("must call setup_ros() first.")
        if not self._smarts:
            raise Exception("must call setup_smarts() first.")
        warned_scenario = False
        observations = {}
        step_delta = None
        if self._target_freq:
            rate = rospy.Rate(self._target_freq)
        rospy.loginfo(f"starting to spin")
        try:
            while not rospy.is_shutdown():

                obs = self._check_reset()
                if not self._scenario_path:
                    if not warned_scenario:
                        rospy.loginfo("waiting for scenario on control channel...")
                        warned_scenario = True
                    elif self._last_step_time:
                        rospy.loginfo("no more scenarios.  exiting...")
                        break
                    continue
                elif obs is not None:
                    observations = obs

                actions = self._do_agents(observations)

                if self._last_step_time:
                    step_delta = rospy.get_time() - self._last_step_time
                self._last_step_time = rospy.get_time()

                self._update_smarts_state(step_delta)

                observations, _, dones, _ = self._smarts.step(actions, step_delta)

                self._publish_state()
                self._publish_agents(observations, dones)

                if self._target_freq:
                    rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo("ROS interrrupted.  exiting...")

        self._reset()  # cleans up the SMARTS instance...


if __name__ == "__main__":
    driver = ROSDriver()
    driver.setup_ros()
    driver.setup_smarts()
    driver.run_forever()
