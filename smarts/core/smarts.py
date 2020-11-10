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
import os
import math
import logging
from typing import List, Sequence, Dict
from collections import defaultdict, namedtuple
import importlib.resources as pkg_resources

import gltf
import numpy
import pybullet
import pybullet_utils.bullet_client as bc
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    ClockObject,
    GeomVertexReader,
    loadPrcFileData,
    NodePath,
    Shader,
)
from sklearn.metrics.pairwise import euclidean_distances

from envision import types as envision_types
from envision.client import Client as EnvisionClient
from .utils.visdom_client import VisdomClient

from . import glsl
from . import models
from .agent import Agent
from .agent_manager import AgentManager
from .bubble_manager import BubbleManager
from .colors import Colors, SceneColors
from .controllers import ActionSpaceType, Controllers
from .coordinates import Heading, Pose
from .trap_manager import TrapManager
from .masks import RenderMasks
from .motion_planner_provider import MotionPlannerProvider
from .provider import ProviderState
from .sensors import Collision
from .scenario import Scenario
from .sumo_traffic_simulation import SumoTrafficSimulation
from .sumo_road_network import SumoRoadNetwork
from .traffic_history_provider import TrafficHistoryProvider
from .vehicle import VEHICLE_CONFIGS, Vehicle, VehicleState
from .vehicle_index import VehicleIndex
from .waypoints import Waypoints


# disable vsync otherwise we are limited to refresh-rate of screen
loadPrcFileData("", "sync-video false")
loadPrcFileData("", "model-path %s" % os.getcwd())
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "gl-version 3 3")
loadPrcFileData("", "notify-level error")

# https://www.panda3d.org/manual/?title=Multithreaded_Render_Pipeline
# loadPrcFileData('', 'threading-model Cull/Draw')

# have makeTextureBuffer create a visible window
# loadPrcFileData('', 'show-buffers true')


class SMARTSNotSetupError(Exception):
    pass


class SMARTS(ShowBase):
    def __init__(
        self,
        agent_interfaces,
        traffic_sim,
        envision: EnvisionClient = None,
        visdom: VisdomClient = None,
        timestep_sec=0.1,
        reset_agents_only=False,
        should_teardown_done_social_agents=True,
    ):
        try:
            super().__init__(self, windowType="offscreen")
        except Exception as e:
            # Known reasons for this failing:
            raise Exception(
                "Display is not found. Try running with different configurations of "
                "`export Display=` using `:0`, `:1`... If this does not work please consult "
                "the documentation."
            ) from e

        gltf.patch_loader(self.loader)

        self._log = logging.getLogger(self.__class__.__name__)

        self._should_teardown_done_social_agents = should_teardown_done_social_agents
        self._is_setup = False
        self._scenario: Scenario = None
        self._envision: EnvisionClient = envision
        self._visdom: VisdomClient = visdom
        self._timestep_sec = timestep_sec
        self._traffic_sim = traffic_sim
        self._motion_planner_provider = MotionPlannerProvider()
        self._traffic_history_provider = TrafficHistoryProvider()
        self._providers = [
            self._traffic_sim,
            self._motion_planner_provider,
            self._traffic_history_provider,
        ]
        # We buffer provider state between steps to compensate for TRACI's timestep delay
        self._last_provider_state = None
        self._reset_agents_only = reset_agents_only  # a.k.a "teleportation"
        self._imitation_learning_mode = False

        # Global clock always proceeds by a fixed dt on each tick
        self.taskMgr.clock.setMode(ClockObject.M_non_real_time)
        self.taskMgr.clock.setDt(timestep_sec)
        self._elapsed_sim_time = 0

        self.setBackgroundColor(0, 0, 0, 1)

        # Displayed framerate is misleading since we are not using a realtime clock
        self.setFrameRateMeter(False)

        # For macOS GUI. See our `BulletClient` docstring for details.
        # from .utils.bullet import BulletClient
        # self._bullet_client = BulletClient(pybullet.GUI)
        self._bullet_client = bc.BulletClient(pybullet.DIRECT)
        self._pybullet_action_spaces = {
            ActionSpaceType.Continuous,
            ActionSpaceType.Lane,
            ActionSpaceType.ActuatorDynamic,
            ActionSpaceType.LaneWithContinuousSpeed,
            ActionSpaceType.Trajectory,
            ActionSpaceType.MPC,
        }

        # Set up indices
        self._agent_manager = AgentManager(agent_interfaces)
        self._vehicle_index = VehicleIndex()

        self._agent_collisions = defaultdict(list)  # list of `Collision` instances
        self._bubble_manager = None
        self._trap_manager: TrapManager = None

        # SceneGraph-related setup
        self._root_np = None
        self._vehicles_np = None
        self._road_network_np = None
        self._ground_bullet_id = None

    def step(self, agent_actions):
        if not self._is_setup:
            raise SMARTSNotSetupError("Must call reset() or setup() before stepping.")

        try:
            return self._step(agent_actions)
        except (KeyboardInterrupt, SystemExit):
            # ensure we clean-up if the user exits the simulation
            self._log.info("Simulation was interrupted by the user.")
            self.destroy()
            raise  # re-raise the KeyboardInterrupt
        except Exception as e:
            self._log.error(
                "Simulation crashed with exception. Attempting to cleanly shutdown."
            )
            self._log.exception(e)
            self.destroy()
            raise  # re-raise

    def _step(self, agent_actions):
        """Steps through the simulation while applying the given agent actions.
        Returns the observations, rewards, and done signals.
        """

        # Due to a limitation of our traffic simulator(SUMO) interface(TRACI), we can
        # only observe traffic state of the previous simulation step.
        #
        # To compensate for this, we:
        #
        # 1. Fetch social agent actions
        # 2. Step all providers and harmonize state
        # 3. Step bubble manager
        # 4. Calculate observation and reward
        # 5. Send observations to social agents
        # 6. Clear done agents
        # 7. Perform visualization
        #
        # In this way, observations and reward are computed with data that is
        # consistently with one step of latencey and the agent will observe consistent
        # data.
        dt = self.taskMgr.clock.get_dt()
        self._elapsed_sim_time = self.taskMgr.clock.get_frame_time()

        # 1. Fetch agent actions
        all_agent_actions = self._agent_manager.fetch_agent_actions(self, agent_actions)

        # 2. Step all providers and harmonize state
        provider_state = self._step_providers(all_agent_actions, dt)

        # 3. Step bubble manager and trap manager
        self.vehicle_index.sync()
        self._bubble_manager.step(self)
        self._trap_manager.step(self)

        # 4. Calculate observation and reward
        # We pre-compute vehicle_states here because we *think* the users will
        # want these during their observation/reward computations.
        # This is a hack to give us some short term perf wins. Longer term we
        # need to expose better support for batched computations
        self._vehicle_states = [v.state for v in self._vehicle_index.vehicles]

        # Agents
        self._agent_manager.step_agent_sensors(self)

        # Panda3D
        # runs through the render pipeline
        # MUST perform this after step_agent_sensors() above, and before observe() below,
        # so that all updates are ready before rendering happens per frame
        self.taskMgr.mgr.poll()

        observations, rewards, scores, dones = self._agent_manager.observe(self)

        response_for_ego = self._agent_manager.filter_response_for_ego(
            (observations, rewards, scores, dones)
        )

        # 5. Send observations to social agents
        self._agent_manager.send_observations_to_social_agents(observations)

        # 6. Clear done agents
        done_agents = {id_ for id_ in dones if dones[id_]}
        agents_to_cleanup = self._agent_manager.teardown_ego_agents(done_agents)
        if self._should_teardown_done_social_agents:
            agents_to_cleanup |= self._agent_manager.teardown_social_agents(done_agents)

        self._teardown_agent_vehicles(agents_to_cleanup)

        # 7. Perform visualization
        self._try_emit_envision_state(provider_state, observations, scores)
        self._try_emit_visdom_obs(observations)

        observations, rewards, scores, dones = response_for_ego
        extras = dict(scores=scores)
        return observations, rewards, dones, extras

    def reset(self, scenario: Scenario):
        if scenario == self._scenario and self._reset_agents_only:
            agent_ids = self._agent_manager.teardown_ego_agents()
            self._teardown_agent_vehicles(agent_ids)
            self._trap_manager.init_traps(scenario)
            self._agent_manager.init_ego_agents(self)
            self._sync_panda3d()
        else:
            self.teardown()
            self.setup(scenario)

        # Tell history provide to ignore vehicles if we have assigned mission to them
        self._traffic_history_provider.set_replaced_ids(scenario.missions.keys())

        self.taskMgr.clock.reset()
        self._elapsed_sim_time = 0

        self._vehicle_states = [v.state for v in self._vehicle_index.vehicles]
        observations, _, _, _ = self._agent_manager.observe(self)
        observations_for_ego = self._agent_manager.reset_agents(observations)

        # Visualization
        self._try_emit_visdom_obs(observations)

        if len(self._agent_manager.ego_agent_ids):
            while len(observations_for_ego) < 1:
                observations_for_ego, _, _, _ = self.step({})

        self._reset_providers()

        return observations_for_ego

    def setup(self, scenario: Scenario):
        self._scenario = scenario

        self._root_np = NodePath("sim")
        self._root_np.reparentTo(self.render)

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
        self._root_np.setShader(unlit_shader)
        self._setup_road_network()
        self._vehicles_np = self._root_np.attachNewNode("vehicles")

        bubbles = scenario.discover_bubbles()
        self._bubble_manager = BubbleManager(bubbles, self.scenario.road_network)
        self._trap_manager = TrapManager(scenario)

        self._setup_bullet_client(self._bullet_client)
        provider_state = self._setup_providers(self._scenario)
        self._agent_manager.setup_agents(self)
        self._harmonize_providers(provider_state)
        self._last_provider_state = provider_state

        self._is_setup = True

    def add_provider(self, provider):
        self._providers.append(provider)

    def _setup_road_network(self):
        glb_path = self.scenario.map_glb_filepath
        if self._road_network_np:
            self._log.debug(
                "road_network={} already exists. Removing and adding a new "
                "one from glb_path={}".format(self._road_network_np, glb_path)
            )
        model_np = self.loader.loadModel(glb_path, noCache=True)

        np = self._root_np.attachNewNode("road_network")
        model_np.reparent_to(np)
        np.hide(RenderMasks.OCCUPANCY_HIDE)
        np.setColor(SceneColors.Road.value)
        self._road_network_np = np

    def _setup_bullet_client(self, client: bc.BulletClient):
        client.resetSimulation()
        client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        # PyBullet defaults the timestep to 240Hz. Several parameters are tuned with
        # this value in mind. For example the number of solver iterations and the error
        # reduction parameters (erp) for contact, friction and non-contact joints.
        # Attempting to get around this we set the number of substeps so that
        # timestep * substeps = 240Hz. Bullet (C++) does something to this effect as
        # well (https://git.io/Jvf0M), but PyBullet does not expose it.
        client.setPhysicsEngineParameter(
            fixedTimeStep=self._timestep_sec,
            numSubSteps=math.ceil(self._timestep_sec / (1 / 240)),
        )

        client.setGravity(0, 0, -9.8)

        plane_path = self._scenario.plane_filepath

        # 1e6 is the default value for plane length and width.
        plane_scale = (
            max(self._scenario.map_bounding_box[0], self._scenario.map_bounding_box[1])
            / 1e6
        )
        if not os.path.exists(plane_path):
            with pkg_resources.path(models, "plane.urdf") as path:
                plane_path = str(path.absolute())

        self._ground_bullet_id = client.loadURDF(
            plane_path,
            useFixedBase=True,
            basePosition=self._scenario.map_bounding_box[2],
            globalScaling=1.1 * plane_scale,
        )

    def teardown(self):
        self._agent_manager.teardown()
        self._vehicle_index.teardown()

        self._bullet_client.resetSimulation()
        self._traffic_sim.teardown()
        self._teardown_providers()

        if self._root_np is not None:
            self._root_np.clearLight()
            self._root_np.removeNode()

        if self._bubble_manager is not None:
            self._bubble_manager.teardown()
            self._bubble_manager = None

        if self._trap_manager is not None:
            self._trap_manager.teardown()
            self._trap_manager = None

        self._vehicles_np = None
        self._ground_bullet_id = None
        self._road_network_np = None
        self._is_setup = False

    def destroy(self):
        self.teardown()

        if self._envision:
            self._envision.teardown()

        if self._visdom:
            self._visdom.teardown()

        self._agent_manager.destroy()
        self._bullet_client.disconnect()

        super().destroy()

    def _teardown_agent_vehicles(self, agent_ids):
        torndown_vehicles = self._vehicle_index.teardown_vehicles_by_actor_ids(
            agent_ids
        )
        self._clear_collisions(agent_ids)

    def attach_sensors_to_vehicles(self, agent_spec, vehicle_ids):
        self._agent_manager.attach_sensors_to_vehicles(
            self, agent_spec.interface, vehicle_ids
        )

    def observe_from(self, vehicle_ids):
        return self._agent_manager.observe_from(self, vehicle_ids)

    @property
    def road_stiffness(self):
        return self._bullet_client.getDynamicsInfo(self._ground_bullet_id, -1)[9]

    @property
    def traffic_sim(self) -> SumoTrafficSimulation:
        return self._traffic_sim

    @property
    def waypoints(self) -> Waypoints:
        return self.scenario.waypoints

    @property
    def road_network(self) -> SumoRoadNetwork:
        return self.scenario.road_network

    @property
    def np(self):
        return self._root_np

    @property
    def vehicles_np(self):
        return self._vehicles_np

    @property
    def bc(self):
        return self._bullet_client

    @property
    def envision(self):
        return self._envision

    @property
    def elapsed_sim_time(self):
        return self._elapsed_sim_time

    def teardown_agents_without_vehicles(self, agent_ids: Sequence):
        """
        Teardown agents in the given list that have no vehicles registered as
        controlled-by or shadowed-by

        Params:
            agent_ids: Sequence of agent ids
        """
        agents_to_teardown = {
            agent_id
            for agent_id in agent_ids
            # Only clean-up when there are no controlled agents left (e.g. boids)
            if len(
                self.vehicle_index.vehicles_by_actor_id(
                    agent_id, include_shadowers=True
                )
            )
            == 0
        }
        self.agent_manager.teardown_social_agents(filter_ids=agents_to_teardown)

    def _teardown_vehicles_and_agents(self, vehicle_ids):
        shadow_and_controlling_agents = set()
        for vehicle_id in vehicle_ids:
            agent_id = self.vehicle_index.actor_id_from_vehicle_id(vehicle_id)
            if agent_id:
                shadow_and_controlling_agents.add(agent_id)
            shadow_agent_id = self.vehicle_index.shadow_actor_id_from_vehicle_id(
                vehicle_id
            )
            if shadow_agent_id:
                shadow_and_controlling_agents.add(shadow_agent_id)

        self._vehicle_index.teardown_vehicles_by_vehicle_ids(vehicle_ids)

        self.teardown_agents_without_vehicles(shadow_and_controlling_agents)

    def _pybullet_provider_sync(self, provider_state: ProviderState):
        current_vehicle_ids = {v.vehicle_id for v in provider_state.vehicles}
        previous_sv_ids = self._vehicle_index.social_vehicle_ids
        exited_vehicles = previous_sv_ids - current_vehicle_ids
        self._teardown_vehicles_and_agents(exited_vehicles)

        # Update our pybullet world given this provider state
        for vehicle in provider_state.vehicles:
            vehicle_id = vehicle.vehicle_id
            # either this is a pybullet agent vehicle, or it is a social vehicle
            if vehicle_id in self._vehicle_index.agent_vehicle_ids:
                # this is an agent vehicle
                agent_id = self._vehicle_index.actor_id_from_vehicle_id(vehicle_id)
                agent_interface = self._agent_manager.agent_interface_for_agent_id(
                    agent_id
                )
                agent_action_space = agent_interface.action_space
                if agent_action_space in self._pybullet_action_spaces:
                    # This is a pybullet agent, we were the source of this vehicle state.
                    # No need to make any changes
                    continue
                else:
                    # This is not a pybullet agent, but it has an avatar in this world
                    # to make it's observations. Update the avatar to match the new
                    # state of this vehicle
                    pybullet_vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
                    pybullet_vehicle.set_pose(vehicle.pose)
                    pybullet_vehicle.set_speed(vehicle.speed)
            else:
                # This vehicle is a social vehicle
                if vehicle_id in self._vehicle_index.social_vehicle_ids:
                    social_vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
                else:
                    # It is a new social vehicle we have not seen yet.
                    # Create it's avatar.
                    social_vehicle = self._vehicle_index.build_social_vehicle(
                        sim=self,
                        vehicle_state=vehicle,
                        actor_id=vehicle_id,
                        vehicle_id=vehicle_id,
                        vehicle_type=vehicle.vehicle_type,
                    )
                # Update the social vehicle avatar to match the vehicle state
                social_vehicle.control(pose=vehicle.pose, speed=vehicle.speed)

    def _pybullet_provider_step(self, agent_actions) -> ProviderState:
        self._perform_agent_actions(agent_actions)

        self._bullet_client.stepSimulation()

        self._process_collisions()

        provider_state = ProviderState()
        pybullet_agent_ids = {
            agent_id
            for agent_id, interface in self._agent_manager.agent_interfaces.items()
            if interface.action_space in self._pybullet_action_spaces
        }

        for vehicle_id in self._vehicle_index.agent_vehicle_ids:
            agent_id = self._vehicle_index.actor_id_from_vehicle_id(vehicle_id)
            if agent_id not in pybullet_agent_ids:
                continue

            vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
            vehicle.step(self._elapsed_sim_time)
            provider_state.vehicles.append(
                VehicleState(
                    vehicle_id=vehicle.id,
                    vehicle_type="passenger",
                    pose=vehicle.pose,
                    dimensions=vehicle.chassis.dimensions,
                    speed=vehicle.speed,
                    source="PYBULLET",
                )
            )

        return provider_state

    @property
    def vehicle_index(self):
        return self._vehicle_index

    @property
    def agent_manager(self):
        return self._agent_manager

    @property
    def providers(self):
        # TODO: Add check to ensure that action spaces are disjoint between providers
        # TODO: It's inconsistent that pybullet is not here
        return self._providers

    def _setup_providers(self, scenario) -> ProviderState:
        provider_state = ProviderState()
        for provider in self.providers:
            provider_state.merge(provider.setup(scenario))

        return provider_state

    def _teardown_providers(self):
        for provider in self.providers:
            provider.teardown()

        self._last_provider_state = None

    def _harmonize_providers(self, provider_state: ProviderState):
        for provider in self.providers:
            provider.sync(provider_state)

        self._pybullet_provider_sync(provider_state)
        self._sync_panda3d()

    def _reset_providers(self):
        for provider in self.providers:
            provider.reset()

    def _step_providers(self, actions, dt) -> List[VehicleState]:
        accumulated_provider_state = ProviderState()

        def agent_controls_vehicles(agent_id):
            vehicles = self._vehicle_index.vehicles_by_actor_id(agent_id)
            return len(vehicles) > 0

        def matches_provider_action_spaces(agent_id, action_spaces):
            interface = self._agent_manager.agent_interface_for_agent_id(agent_id)
            return interface.action_space in action_spaces

        # PyBullet
        pybullet_actions = {
            agent_id: action
            for agent_id, action in actions.items()
            if agent_controls_vehicles(agent_id)
            and matches_provider_action_spaces(agent_id, self._pybullet_action_spaces)
        }
        accumulated_provider_state.merge(self._pybullet_provider_step(pybullet_actions))

        for provider in self.providers:
            provider_state = self._step_provider(
                provider, actions, dt, self._elapsed_sim_time
            )
            if provider == self._traffic_sim:
                # Remove agent vehicles from provider vehicles
                provider_state.filter(self._vehicle_index.agent_vehicle_ids)

            accumulated_provider_state.merge(provider_state)

        self._harmonize_providers(accumulated_provider_state)
        return accumulated_provider_state

    def _step_provider(self, provider, actions, dt, elapsed_sim_time):
        def agent_controls_vehicles(agent_id):
            vehicles = self._vehicle_index.vehicles_by_actor_id(agent_id)
            return len(vehicles) > 0

        provider_actions = {}
        for agent_id, action in actions.items():
            agent_interface = self._agent_manager.agent_interface_for_agent_id(agent_id)
            if (
                agent_interface
                and agent_controls_vehicles(agent_id)
                and agent_interface.action_space in provider.action_spaces
            ):
                vehicle_ids = [
                    v.id
                    for v in self._vehicle_index.vehicles_by_actor_id(
                        agent_id, include_shadowers=True
                    )
                ]

                if self._agent_manager.is_boid_agent(self, agent_id):
                    for vehicle_id, vehicle_action in action.items():
                        assert vehicle_id in vehicle_ids
                        provider_actions[vehicle_id] = vehicle_action
                else:
                    assert len(vehicle_ids) == 1
                    provider_actions[vehicle_ids[0]] = action

        provider_state = provider.step(provider_actions, dt, elapsed_sim_time)
        return provider_state

    @property
    def scenario(self):
        return self._scenario

    @property
    def traffic_sim(self):
        return self._traffic_sim

    @property
    def np(self):
        return self._root_np

    @property
    def timestep_sec(self):
        return self._timestep_sec

    @property
    def road_stiffness(self):
        return self._bullet_client.getDynamicsInfo(self._ground_bullet_id, -1)[9]

    def neighborhood_vehicles_around_vehicle(self, vehicle, radius=None):
        other_states = [v for v in self._vehicle_states if v.vehicle_id != vehicle.id]
        if radius is None:
            return other_states

        other_positions = [state.pose.position for state in other_states]
        if not other_positions:
            return []

        distances = euclidean_distances(other_positions, [vehicle.position]).reshape(
            -1,
        )
        indices = numpy.argwhere(distances <= radius).flatten()
        return [other_states[i] for i in indices]

    def agent_reached_goal(self, agent_id):
        # TODO: If any vehicle reached the goal we consider the agent having reached
        #       its goal. Is this the behaviour we want?
        vehicles = self._vehicle_index.vehicles_by_actor_id(agent_id)
        for vehicle in vehicles:
            sensor_state = self._vehicle_index.sensor_state_for_vehicle_id(vehicle.id)
            distance_travelled = vehicle.trip_meter_sensor()
            mission = sensor_state.mission_planner.mission
            reached_goal = mission.is_complete(vehicle, distance_travelled)
            if reached_goal:
                return True

        return False

    def agent_is_on_shoulder(self, agent_id):
        vehicles = self._vehicle_index.vehicles_by_actor_id(agent_id)
        for vehicle in vehicles:
            if any(
                [
                    not self.scenario.road_network.point_is_within_road(
                        corner_coordinate
                    )
                    for corner_coordinate in vehicle.bounding_box
                ]
            ):
                return True
        return False

    def agent_is_off_road(self, agent_id):
        vehicles = self._vehicle_index.vehicles_by_actor_id(agent_id)
        for vehicle in vehicles:
            if self.scenario.road_network.point_is_within_road(vehicle.position):
                return False

        return len(vehicles) > 0

    def agent_did_collide(self, agent_id):
        return (
            len(
                [
                    c
                    for c in self._agent_collisions[agent_id]
                    if c.collidee_id != self._ground_bullet_id
                ]
            )
            > 0
        )

    def agent_collisions(self, agent_id):
        # Filter out off-road collisions.
        return [
            c
            for c in self._agent_collisions[agent_id]
            if c.collidee_id != self._ground_bullet_id
        ]

    def _clear_collisions(self, agent_ids):
        for agent_id in agent_ids:
            self._agent_collisions.pop(agent_id, None)

    def _perform_agent_actions(self, agent_actions):
        for agent_id, action in agent_actions.items():
            agent_vehicles = self._vehicle_index.vehicles_by_actor_id(agent_id)
            if len(agent_vehicles) == 0:
                self._log.warning(
                    f"{agent_id} doesn't have a vehicle, is the agent done? (dropping action)"
                )
            else:
                agent_interface = self._agent_manager.agent_interface_for_agent_id(
                    agent_id
                )
                is_boid_agent = self._agent_manager.is_boid_agent(self, agent_id)

                for vehicle in agent_vehicles:
                    vehicle_action = action[vehicle.id] if is_boid_agent else action

                    controller_state = self._vehicle_index.controller_state_for_vehicle_id(
                        vehicle.id
                    )
                    sensor_state = self._vehicle_index.sensor_state_for_vehicle_id(
                        vehicle.id
                    )
                    # TODO: Support performing batched actions
                    Controllers.perform_action(
                        self,
                        agent_id,
                        vehicle,
                        vehicle_action,
                        controller_state,
                        sensor_state,
                        agent_interface.action_space,
                    )

    def _sync_panda3d(self):
        for vehicle in self._vehicle_index.vehicles:
            vehicle.sync_to_panda3d()

    def _process_collisions(self):
        self._agent_collisions = defaultdict(list)  # list of `Collision` instances

        for vehicle_id in self._vehicle_index.agent_vehicle_ids:
            vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
            # We are only concerned with vehicle-vehicle collisions
            collidee_bullet_ids = set(
                [p.bullet_id for p in vehicle.chassis.contact_points]
            )
            collidee_bullet_ids.discard(self._ground_bullet_id)

            if not collidee_bullet_ids:
                continue

            for bullet_id in collidee_bullet_ids:
                collidee = self._bullet_id_to_vehicle(bullet_id)
                collision = self._node_to_collision(collidee.np.node())
                agent_id = self._vehicle_index.actor_id_from_vehicle_id(vehicle_id)
                self._agent_collisions[agent_id].append(collision)

    def _bullet_id_to_vehicle(self, bullet_id):
        for vehicle in self._vehicle_index.vehicles:
            if bullet_id == vehicle.chassis.bullet_id:
                return vehicle

        assert False, "Only collisions with agent or social vehicles is supported"

    def _node_to_collision(self, node):
        for vehicle in self._vehicle_index.vehicles:
            if node == vehicle.np.node():
                actor_id = self._vehicle_index.actor_id_from_vehicle_id(vehicle.id)
                # TODO: Should we specify the collidee as the vehicle ID instead of
                #       the agent/social ID?
                return Collision(collidee_id=actor_id)

        assert False, "Only collisions with agent or social vehicles is supported"

    def _try_emit_envision_state(self, provider_state, obs, scores):
        if not self._envision:
            return

        traffic = {}
        for v in provider_state.vehicles:
            if v.vehicle_id in self._vehicle_index.agent_vehicle_ids:
                # this is an agent controlled vehicle
                agent_id = self._vehicle_index.actor_id_from_vehicle_id(v.vehicle_id)
                agent_obs = obs[agent_id]
                is_boid_agent = self._agent_manager.is_boid_agent(self, agent_id)
                vehicle_obs = agent_obs[v.vehicle_id] if is_boid_agent else agent_obs

                if self._agent_manager.is_ego(agent_id):
                    actor_type = envision_types.TrafficActorType.Agent
                    mission_route_geometry = self._vehicle_index.sensor_state_for_vehicle_id(
                        v.vehicle_id
                    ).mission_planner.route.geometry
                else:
                    actor_type = envision_types.TrafficActorType.SocialAgent
                    mission_route_geometry = None

                point_cloud = vehicle_obs.lidar_point_cloud or ([], [], [])
                point_cloud = point_cloud[0]  # (points, hits, rays), just want points

                # TODO: driven path should be read from vehicle_obs
                driven_path = self._vehicle_index.vehicle_by_id(
                    v.vehicle_id
                ).driven_path_sensor()

                road_waypoints = []
                if vehicle_obs.road_waypoints:
                    road_waypoints = [
                        path
                        for paths in vehicle_obs.road_waypoints.lanes.values()
                        for path in paths
                    ]
                traffic[v.vehicle_id] = envision_types.TrafficActorState(
                    name=self._agent_manager.name_for_agent(agent_id),
                    actor_type=actor_type,
                    vehicle_type=envision_types.VehicleType.Car,
                    position=v.pose.position,
                    heading=v.pose.heading,
                    speed=v.speed,
                    actor_id=envision_types.format_actor_id(
                        agent_id, v.vehicle_id, is_multi=is_boid_agent,
                    ),
                    events=vehicle_obs.events,
                    waypoint_paths=(vehicle_obs.waypoint_paths or []) + road_waypoints,
                    point_cloud=point_cloud,
                    driven_path=driven_path,
                    mission_route_geometry=mission_route_geometry,
                )
            elif v.vehicle_id in self._vehicle_index.social_vehicle_ids:
                # this is a social vehicle
                vehicle = self._vehicle_index.vehicle_by_id(v.vehicle_id)
                traffic[v.vehicle_id] = envision_types.TrafficActorState(
                    actor_type=envision_types.TrafficActorType.SocialVehicle,
                    vehicle_type=v.vehicle_type,
                    position=list(v.pose.position),
                    heading=v.pose.heading,
                    speed=v.speed,
                )

        bubble_geometry = [
            list(bubble.geometry.exterior.coords)
            for bubble in self._bubble_manager.bubbles
        ]

        state = envision_types.State(
            traffic=traffic,
            scenario_id=self.scenario.scenario_hash,
            bubbles=bubble_geometry,
            scene_colors=SceneColors.EnvisionColors.value,
            scores=scores,
        )
        self._envision.send(state)

    def _try_emit_visdom_obs(self, obs):
        if not self._visdom:
            return

        self._visdom.send(obs)
