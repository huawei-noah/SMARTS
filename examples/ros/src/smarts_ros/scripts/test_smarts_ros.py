#!/usr/bin/env python3

# PKG = 'smarts_ros'
# import roslib
# roslib.load_manifest(PKG)

import os
import json
import rospy
import sys
from unittest import TestCase

from smarts_ros.msg import (
    AgentTask,
    AgentReport,
    AgentSpec,
    AgentsStamped,
    EntitiesStamped,
    EntityState,
    SmartsControl,
)
from smarts_ros.srv import SmartsInfo
from smarts.core.coordinates import fast_quaternion_from_angle


class TestSmartsRos(TestCase):
    """Node to test the smarts_ros package."""

    def __init__(self):
        super().__init__()
        self._smarts_info_srv = None
        self._control_publisher = None
        self._agents = {}

    def setup_ros(
        self,
        node_name: str = "SMARTS",
        def_namespace: str = "SMARTS/",
        pub_queue_size: int = 10,
    ):
        rospy.init_node(node_name, anonymous=True)

        # If the namespace is aready set in the environment, we use it,
        # otherwise we use our default.
        namespace = def_namespace if not os.environ.get("ROS_NAMESPACE") else ""

        self._control_publisher = rospy.Publisher(
            f"{namespace}control", SmartsControl, queue_size=pub_queue_size
        )
        self._agents_publisher = rospy.Publisher(
            f"{namespace}agent_spec", AgentSpec, queue_size=pub_queue_size
        )
        self._entities_publisher = rospy.Publisher(
            f"{namespace}entities_in", EntitiesStamped, queue_size=pub_queue_size
        )

        rospy.Subscriber(f"{namespace}agents_out", AgentsStamped, self._agents_callback)
        rospy.Subscriber(
            f"{namespace}entities_out", EntitiesStamped, self._entities_callback
        )

        service_name = f"{namespace}{node_name}_info"
        rospy.wait_for_service(service_name)
        self._smarts_info_srv = rospy.ServiceProxy(service_name, SmartsInfo)
        smarts_info = self._smarts_info_srv()
        rospy.loginfo(f"Tester detected SMARTS version={smarts_info.version}.")

    @staticmethod
    def _vector_to_xyz(v, xyz):
        xyz.x, xyz.y, xyz.z = v[0], v[1], v[2]

    @staticmethod
    def _vector_to_xyzw(v, xyzw):
        xyzw.x, xyzw.y, xyzw.z, xyzw.w = v[0], v[1], v[2], v[3]

    def _init_scenario(self):
        scenario = rospy.get_param("~scenario")
        rospy.loginfo(f"Tester using scenario:  {scenario}.")

        control_msg = SmartsControl()
        control_msg.reset_with_scenario_path = scenario
        if rospy.get_param("~add_agent", False):
            agent_spec = AgentSpec()
            agent_spec.agent_id = "TestROSAgent"
            agent_spec.veh_type = AgentSpec.VEHICLE_TYPE_CAR
            agent_spec.start_speed = rospy.get_param("~agent_speed")
            pose = json.loads(rospy.get_param("~agent_start_pose"))
            TestSmartsRos._vector_to_xyz(pose[0], agent_spec.start_pose.position)
            TestSmartsRos._vector_to_xyzw(
                fast_quaternion_from_angle(pose[1]),
                agent_spec.start_pose.orientation,
            )
            task = AgentTask()
            task.task_ref = rospy.get_param("~task_ref")
            task.task_ver = rospy.get_param("~task_ver")
            task.params_json = rospy.get_param("~task_params")
            agent_spec.tasks = [task]
            self._agents[agent_spec.agent_id] = agent_spec
            control_msg.initial_agents = [agent_spec]
        self._control_publisher.publish(control_msg)
        rospy.loginfo(
            f"SMARTS control message sent with scenario_path=f{control_msg.reset_with_scenario_path} and initial_agents=f{control_msg.initial_agents}."
        )
        return scenario

    def _agents_callback(self, agents: AgentsStamped):
        rospy.logdebug(f"got report about {len(agents.agents)} agents")
        if len(agents.agents) != len(self._agents):
            rospy.logwarn(
                f"SMARTS reporting {len(agents.agents)} agents, but we've added {len(self._agents)}."
            )

    def _entities_callback(self, entities: EntitiesStamped):
        rospy.logdebug(f"got report about {len(entities.entities)} agents")

    def run_forever(self):
        if not self._smarts_info_srv:
            raise RuntimeError("must call setup_ros() first.")
        scenario = self._init_scenario()
        rospy.spin()


if __name__ == "__main__":
    # import rostest
    # rostest.rosrun(PKG, 'test_smarts_ros', TestSmartsRos)
    tester = TestSmartsRos()
    tester.setup_ros()
    tester.run_forever()
