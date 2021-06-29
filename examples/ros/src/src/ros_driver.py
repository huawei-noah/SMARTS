#!/usr/bin/env python3

import rospy
import std_msgs
import sys
from smarts_ros.msg import EntitiesStamped, EntityState, SmartsControl

from collections import deque
import numpy as np
import time
from threading import Lock
from typing import Any, List, Sequence

from envision.client import Client as Envision
from smarts.core.coordinates import BoundingBox, Heading, Pose
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.vehicle import VehicleState


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
        self._publisher = None
        self._most_recent_state_sent = None
        with self._state_lock:
            self._recent_state = deque(maxlen=3)
        with self._control_lock:
            self._scenario_path = None
            self._reset_smarts = False

    def setup_ros(
        self,
        node_name: str = "SMARTS",
        namespace: str = "SMARTS",
        target_freq: float = None,
        buffer_size: int = 3,
        pub_queue_size: int = 10,
    ):
        assert not self._publisher
        # enforce only one SMARTS instance per ROS network...
        rospy.init_node(node_name, anonymous=False)

        out_topic = f"{namespace}/entities_out"
        self._publisher = rospy.Publisher(
            out_topic, EntitiesStamped, queue_size=pub_queue_size
        )

        self._state_topic = f"{namespace}/entities_in"
        rospy.Subscriber(self._state_topic, EntitiesStamped, self._entities_callback)
        rospy.Subscriber(
            f"{namespace}/control", SmartsControl, self._smarts_control_callback
        )

        buffer_size = rospy.get_param(f"{namespace}/buffer_size", buffer_size)
        if buffer_size and buffer_size != self._recent_state.maxlen:
            assert buffer_size > 0
            self._recent_state = deque(maxlen=buffer_size)

        # If target_freq is not specified, SMARTS is allowed to
        # run as quickly as it can with no delay between steps.
        target_freq = rospy.get_param(f"{namespace}/target_freq", target_freq)
        if target_freq:
            assert target_freq > 0.0
            if target_freq > SMART_MAX_FREQ:
                rospy.logwarn(
                    f"specified target frequency of {target_freq} Hz cannot be guaranteed by SMARTS."
                )
            self._target_freq = target_freq

    def setup_smarts(self, headless: bool, seed: int):
        assert not self._smarts
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

    def _smarts_control_callback(self, control):
        with self._control_lock:
            self._scenario_path = control.reset_with_scenario_path
            self._reset_smarts = True

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
        avg = np.array((0.0, 0.0, 0.0))
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
    def _vector_to_xyzw(v, xyz):
        xyz.x, xyz.y, xyz.z, xyz.w = v[0], v[1], v[2], v[3]

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
        self._publisher.publish(entities)

    def _check_reset(self):
        with self._control_lock:
            if self._reset_smarts:
                rospy.loginfo(f"resetting SMARTS w/ scenario={self._scenario_path}")
                self._smarts.reset(Scenario(self._scenario_path))
                self._reset_smarts = False
                self._last_step_time = None

    def run_forever(self):
        if not self._publisher:
            raise Exception("must call setup_ros() first.")
        if not self._smarts:
            raise Exception("must call setup_smarts() first.")
        warned_scenario = False
        step_delta = None
        if self._target_freq:
            rate = rospy.Rate(self._target_freq)
        rospy.loginfo(f"starting to spin")
        while not rospy.is_shutdown():
            self._check_reset()
            if not self._scenario_path:
                if warned_scenario:
                    rospy.loginfo("waiting for scenario on control channel...")
                    warned_scenario = True
                continue
            if self._last_step_time:
                step_delta = rospy.get_time() - self._last_step_time
            self._last_step_time = rospy.get_time()
            self._update_smarts_state(step_delta)
            self._smarts.step({}, step_delta)
            self._publish_state()
            if self._target_freq:
                rate.sleep()
        self._reset()


def _parse_args():
    from examples.argument_parser import default_argument_parser

    parser = default_argument_parser()
    parser.add_argument(
        "--node-name",
        help="The name to use for this ROS node.",
        type=str,
        default="SMARTS",
    )
    parser.add_argument(
        "--namespace",
        help="The ROS namespace to use for published/subscribed toipc names.",
        type=str,
        default="SMARTS",
    )
    parser.add_argument(
        "--buffer_size",
        help="The number of entity messages to buffer to use for smoothing/extrapolation.",
        type=int,
        choices=range(1, 100),
        default=3,
    )
    parser.add_argument(
        "--target_freq",
        help="The target frequencey in Hz.  If not specified, go as quickly as SMARTS permits.",
        type=float,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.scenarios:
        print("scenarios should be passed in via the SMARTS/control ROS topic.")
        sys.exit(-1)
    driver = ROSDriver()
    driver.setup_ros(args.node_name, args.namespace, args.target_freq, args.buffer_size)
    driver.setup_smarts(args.headless, args.seed)
    driver.run_forever()
