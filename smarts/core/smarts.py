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
import math
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from envision import types as envision_types
from envision.client import Client as EnvisionClient

with warnings.catch_warnings():
    # XXX: Benign warning, seems no other way to "properly" fix
    warnings.filterwarnings("ignore", "numpy.ufunc size changed")
    from sklearn.metrics.pairwise import euclidean_distances

from smarts import VERSION
from smarts.core.chassis import AckermannChassis, BoxChassis
from smarts.core.plan import Plan

from . import models
from .agent_interface import AgentInterface
from .agent_manager import AgentManager
from .bubble_manager import BubbleManager
from .colors import SceneColors
from .controllers import ActionSpaceType, Controllers
from .coordinates import BoundingBox, Point
from .external_provider import ExternalProvider
from .motion_planner_provider import MotionPlannerProvider
from .provider import Provider, ProviderRecoveryFlags, ProviderState
from .road_map import RoadMap
from .scenario import Mission, Scenario
from .sensors import Collision, Observation
from .sumo_traffic_simulation import SumoTrafficSimulation
from .traffic_history_provider import TrafficHistoryProvider
from .trajectory_interpolation_provider import TrajectoryInterpolationProvider
from .trap_manager import TrapManager
from .utils import pybullet
from .utils.id import Id
from .utils.math import rounder_for_dt
from .utils.pybullet import bullet_client as bc
from .utils.visdom_client import VisdomClient
from .vehicle import Vehicle, VehicleState
from .vehicle_index import VehicleIndex

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s: {%(module)s} %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
    level=logging.ERROR,
)

MAX_PYBULLET_FREQ = 240


class SMARTSNotSetupError(Exception):
    """Represents a case where SMARTS cannot operate because it is not set up yet."""

    pass


class SMARTSDestroyedError(Exception):
    """Represents a case where SMARTS cannot operate because it is destroyed."""

    pass


class SMARTS:
    """The core SMARTS simulator."""

    def __init__(
        self,
        agent_interfaces,
        traffic_sim,  # SumoTrafficSimulation
        envision: Optional[EnvisionClient] = None,
        visdom: Optional[VisdomClient] = None,
        fixed_timestep_sec: float = 0.1,
        reset_agents_only: bool = False,
        zoo_addrs=None,
        external_provider: bool = False,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._sim_id = Id.new("smarts")
        self._is_setup = False
        self._is_destroyed = False
        self._scenario: Optional[Scenario] = None
        self._renderer = None
        self._envision: Optional[EnvisionClient] = envision
        self._visdom: Optional[VisdomClient] = visdom
        self._traffic_sim = traffic_sim
        self._external_provider = None
        self._resetting = False
        self._reset_required = False

        assert fixed_timestep_sec is None or fixed_timestep_sec > 0
        self.fixed_timestep_sec: float = fixed_timestep_sec
        self._last_dt = fixed_timestep_sec

        self._elapsed_sim_time = 0
        self._total_sim_time = 0
        self._step_count = 0

        self._motion_planner_provider = MotionPlannerProvider()
        self._traffic_history_provider = TrafficHistoryProvider()
        self._trajectory_interpolation_provider = TrajectoryInterpolationProvider()
        self._provider_recovery_flags: Dict[Provider, ProviderRecoveryFlags] = {}
        self._providers: List[Provider] = []

        self.add_provider(
            self._motion_planner_provider,
        )
        self.add_provider(
            self._traffic_history_provider,
        )
        self.add_provider(
            self._trajectory_interpolation_provider,
        )
        if self._traffic_sim:
            self._insert_provider(
                0,
                self._traffic_sim,
                recovery_flags=ProviderRecoveryFlags.EPISODE_REQUIRED
                | ProviderRecoveryFlags.ATTEMPT_RECOVERY,
            )
        if external_provider:
            self._external_provider = ExternalProvider(self)
            self._insert_provider(0, self._external_provider)

        # We buffer provider state between steps to compensate for TRACI's timestep delay
        self._last_provider_state = None
        self._reset_agents_only = reset_agents_only  # a.k.a "teleportation"
        self._imitation_learning_mode = False

        # For macOS GUI. See our `BulletClient` docstring for details.
        # from .utils.bullet import BulletClient
        # self._bullet_client = BulletClient(pybullet.GUI)
        self._bullet_client = bc.BulletClient(pybullet.DIRECT)
        self._dynamic_action_spaces = {
            ActionSpaceType.Continuous,
            ActionSpaceType.Lane,
            ActionSpaceType.ActuatorDynamic,
            ActionSpaceType.LaneWithContinuousSpeed,
            ActionSpaceType.Trajectory,
            ActionSpaceType.MPC,
            # ActionSpaceType.Imitation,
        }

        # Set up indices
        self._agent_manager = AgentManager(agent_interfaces, zoo_addrs)
        self._vehicle_index = VehicleIndex()

        # TODO: Should not be stored in SMARTS
        self._vehicle_collisions = defaultdict(list)  # list of `Collision` instances
        self._vehicle_states = []

        self._bubble_manager = None
        self._trap_manager: Optional[TrapManager] = None

        self._ground_bullet_id = None
        self._map_bb = None

    def step(
        self,
        agent_actions: Dict[str, Any],
        time_delta_since_last_step: Optional[float] = None,
    ) -> Tuple[
        Dict[str, Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, float]],
    ]:
        """Progress the simulation by a fixed or specified time.
        Args:
            agent_actions:
                Actions that the agents want to perform on their actors.
            time_delta_since_last_step:
                Overrides the simulation step length. Progress simulation time by the given amount.
                Note the time_delta_since_last_step param is in (nominal) seconds.
        Returns:
            observations, rewards, dones, infos
        """
        if not self._is_setup:
            raise SMARTSNotSetupError("Must call reset() or setup() before stepping.")
        self._check_valid()
        assert not (
            self._fixed_timestep_sec and time_delta_since_last_step
        ), "cannot switch from fixed- to variable-time steps mid-simulation"

        try:
            return self._step(agent_actions, time_delta_since_last_step)
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

    def _check_if_acting_on_active_agents(self, agent_actions):
        for agent_id in agent_actions.keys():
            if agent_id not in self._agent_manager.ego_agent_ids:
                self._log.warning(
                    f"Attempted to perform actions on non-existing agent, {agent_id} "
                )

    def _step(self, agent_actions, time_delta_since_last_step: Optional[float] = None):
        """Steps through the simulation while applying the given agent actions.
        Returns the observations, rewards, done, and infos signals.
        """

        # Due to a limitation of our traffic simulator(SUMO) interface(TRACI), we can
        # only observe traffic state of the previous simulation step.
        #
        # To compensate for this, we:
        #
        # 0. Advance the simulation clock
        # 1. Fetch social agent actions
        # 2. Step all providers and harmonize state
        # 3. Step bubble manager
        # 4. Calculate observation and reward
        # 5. Send observations to social agents
        # 6. Clear done agents
        # 7. Perform visualization
        #
        # In this way, observations and reward are computed with data that is
        # consistently with one step of latency and the agent will observe consistent
        # data.

        # 0. Advance the simulation clock.
        # It's been this long since our last step.
        self._last_dt = time_delta_since_last_step or self._fixed_timestep_sec or 0.1
        self._elapsed_sim_time = self._rounder(self._elapsed_sim_time + self._last_dt)

        # 1. Fetch agent actions
        self._log.info("Fetching agent actions")
        all_agent_actions = self._agent_manager.fetch_agent_actions(self, agent_actions)

        # 2. Step all providers and harmonize state
        self._log.info("Stepping all providers and harmonizing state")
        provider_state = self._step_providers(all_agent_actions)
        self._log.info("Checking if all agents are active")
        self._check_if_acting_on_active_agents(agent_actions)

        # 3. Step bubble manager and trap manager
        self._log.info("Syncing vehicle index")
        self._vehicle_index.sync()
        self._log.info("Stepping through bubble manager")
        self._bubble_manager.step(self)
        self._log.info("Stepping through trap manager")
        self._trap_manager.step(self)

        # 4. Calculate observation and reward
        # We pre-compute vehicle_states here because we *think* the users will
        # want these during their observation/reward computations.
        # This is a hack to give us some short term perf wins. Longer term we
        # need to expose better support for batched computations
        self._vehicle_states = [v.state for v in self._vehicle_index.vehicles]

        # Agents
        self._log.info("Stepping through sensors")
        self._agent_manager.step_sensors(self)

        if self._renderer:
            # runs through the render pipeline (for camera-based sensors)
            # MUST perform this after step_sensors() above, and before observe() below,
            # so that all updates are ready before rendering happens per
            self._log.info("Running through the render pipeline")
            self._renderer.render()

        self._log.info("Calculating observations and rewards")
        observations, rewards, scores, dones = self._agent_manager.observe(self)

        self._log.info("Filtering response for ego")
        response_for_ego = self._agent_manager.filter_response_for_ego(
            (observations, rewards, scores, dones)
        )

        # 5. Send observations to social agents
        self._log.info("Sending observations to social agents")
        self._agent_manager.send_observations_to_social_agents(observations)

        # 6. Clear done agents
        self._log.info("Clearing done agents")
        self._teardown_done_agents_and_vehicles(dones)

        # 7. Perform visualization
        self._log.info("Trying to emit the envision state")
        self._try_emit_envision_state(provider_state, observations, scores)
        self._log.info("Trying to emit the visdom observations")
        self._try_emit_visdom_obs(observations)

        observations, rewards, scores, dones = response_for_ego
        extras = dict(scores=scores)

        self._step_count += 1

        return observations, rewards, dones, extras

    def _teardown_done_agents_and_vehicles(self, dones):
        def done_vehicle_ids(dones):
            vehicle_ids = set()
            for agent_id, done in dones.items():
                if self._agent_manager.is_boid_agent(agent_id):
                    vehicle_ids.update(id_ for id_ in done if done[id_])
                elif done:
                    ids = self._vehicle_index.vehicle_ids_by_actor_id(agent_id)
                    # 0 if shadowing, 1 if active
                    assert len(ids) <= 1, f"{len(ids)} <= 1"
                    vehicle_ids.update(ids)

            return vehicle_ids

        def done_agent_ids(dones):
            agent_ids = set()
            for agent_id, done in dones.items():
                if self._agent_manager.is_boid_agent(agent_id):
                    if not self.agent_manager.is_boid_keep_alive_agent(
                        agent_id
                    ) and all(dones[agent_id].values()):
                        agent_ids.add(agent_id)
                elif done:
                    agent_ids.add(agent_id)

            return agent_ids

        # XXX: These can not be put inline because we do queries that must proceed
        #      the actual teardown.
        vehicles_to_teardown = done_vehicle_ids(dones)
        agents_to_teardown = done_agent_ids(dones)

        self._agent_manager.teardown_ego_agents(agents_to_teardown)
        self._agent_manager.teardown_social_agents(agents_to_teardown)
        self._teardown_vehicles(vehicles_to_teardown)

    def reset(self, scenario: Scenario) -> Dict[str, Observation]:
        """Reset the simulation, reinitialize with the specified scenario. Then progress the
         simulation up to the first time an agent returns an observation, or time 0 if there are no
         agents in the simulation.
        Args:
            scenario:
                The scenario to reset the simulation with.
        Returns:
            Agent observations. This observation is as follows:
                - If no agents: the initial simulation observation at time 0
                - If agents: the first step of the simulation with an agent observation
        """
        tries = 2
        first_exception = None
        for _ in range(tries):
            try:
                self._resetting = True
                return self._reset(scenario)
            except Exception as e:
                if not first_exception:
                    first_exception = e
            finally:
                self._resetting = False
        self._log.error(f"Failed to successfully reset after {tries} times.")
        raise first_exception

    def _reset(self, scenario: Scenario):
        self._check_valid()
        if (
            scenario == self._scenario
            and self._reset_agents_only
            and not self._reset_required
        ):
            vehicle_ids_to_teardown = []
            agent_ids = self._agent_manager.teardown_ego_agents()
            for agent_id in agent_ids:
                ids = self._vehicle_index.vehicle_ids_by_actor_id(agent_id)
                vehicle_ids_to_teardown.extend(ids)
            self._teardown_vehicles(set(vehicle_ids_to_teardown))
            assert self._trap_manager
            self._trap_manager.init_traps(scenario.road_map, scenario.missions)
            self._agent_manager.init_ego_agents(self)
            if self._renderer:
                self._sync_vehicles_to_renderer()
        else:
            self.teardown()
            self.setup(scenario)

        # Tell history provide to ignore vehicles if we have assigned mission to them
        self._traffic_history_provider.set_replaced_ids(
            m.vehicle_spec.veh_id
            for m in scenario.missions.values()
            if m and m.vehicle_spec
        )

        self._total_sim_time += self._elapsed_sim_time
        self._elapsed_sim_time = 0
        self._step_count = 0
        self._reset_required = False

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
        """Setup the next scenario."""
        self._check_valid()
        self._scenario = scenario

        self._bubble_manager = BubbleManager(scenario.bubbles, scenario.road_map)
        self._trap_manager = TrapManager(scenario)

        if self._renderer:
            self._renderer.setup(scenario)
        self._setup_bullet_client(self._bullet_client)
        provider_state = self._setup_providers(self._scenario)
        self._vehicle_index.load_controller_params(
            scenario.controller_parameters_filepath
        )
        self._agent_manager.setup_agents(self)

        self._harmonize_providers(provider_state)
        self._last_provider_state = provider_state

        self._is_setup = True

    def add_provider(
        self,
        provider: Provider,
        recovery_flags: ProviderRecoveryFlags = ProviderRecoveryFlags.EXPERIMENT_REQUIRED,
    ):
        """Add a provider to the simulation. A provider is a co-simulator conformed to a common
        interface.
        """
        self._check_valid()
        assert isinstance(provider, Provider)
        self._insert_provider(len(self._providers), provider, recovery_flags)

    def _insert_provider(
        self,
        index: int,
        provider: Provider,
        recovery_flags: ProviderRecoveryFlags = ProviderRecoveryFlags.EXPERIMENT_REQUIRED,
    ):
        assert isinstance(provider, Provider)
        self._providers.insert(index, provider)
        self._provider_recovery_flags[provider] = recovery_flags

    def switch_ego_agents(self, agent_interfaces: Dict[str, AgentInterface]):
        """Change the ego agents in the simulation. Effective on the next reset."""
        self._check_valid()
        self._agent_manager.switch_initial_agents(agent_interfaces)
        self._is_setup = False

    def add_agent_with_mission(
        self, agent_id: str, agent_interface: AgentInterface, mission: Mission
    ):
        """Add an agent to the simulation. The simulation will attempt to provide a vehicle for
        the agent.
        """
        self._check_valid()
        # TODO:  check that agent_id isn't already used...
        if self._trap_manager.add_trap_for_agent(agent_id, mission, self.road_map):
            self._agent_manager.add_ego_agent(agent_id, agent_interface)
        else:
            self._log.warning(
                f"Unable to add entry trap for new agent '{agent_id}' with mission."
            )

    def add_agent_and_switch_control(
        self,
        vehicle_id: str,
        agent_id: str,
        agent_interface: AgentInterface,
        mission: Mission,
    ) -> Vehicle:
        """Add the new specified ego agent and then take over control of the specified vehicle."""
        self._check_valid()
        self.agent_manager.add_ego_agent(agent_id, agent_interface, for_trap=False)
        vehicle = self.switch_control_to_agent(
            vehicle_id, agent_id, mission, recreate=False, is_hijacked=True
        )
        self.create_vehicle_in_providers(vehicle, agent_id)

        return vehicle

    def switch_control_to_agent(
        self,
        vehicle_id: str,
        agent_id: str,
        mission: Mission,
        recreate: bool,
        is_hijacked: bool,
    ) -> Vehicle:
        """Give control of the specified vehicle to the given agent.

        It is not possible to take over a vehicle already controlled by another agent.
        """
        self._check_valid()
        # Check if this is a history vehicle
        history_veh_id = self._traffic_history_provider.get_history_id(vehicle_id)
        canonical_veh_id = history_veh_id if history_veh_id else vehicle_id

        assert not self.vehicle_index.vehicle_is_hijacked(
            canonical_veh_id
        ), f"Vehicle has already been hijacked: {canonical_veh_id}"
        assert (
            not canonical_veh_id in self.vehicle_index.agent_vehicle_ids()
        ), f"Can't hijack vehicle that is already controlled by an agent: {canonical_veh_id}"

        # Remove vehicle from traffic history provider
        if history_veh_id:
            self._traffic_history_provider.set_replaced_ids([vehicle_id])

        # Switch control to agent
        plan = Plan(self.road_map, mission)
        interface = self.agent_manager.agent_interface_for_agent_id(agent_id)
        self.vehicle_index.start_agent_observation(
            self, canonical_veh_id, agent_id, interface, plan
        )
        vehicle = self.vehicle_index.switch_control_to_agent(
            self,
            canonical_veh_id,
            agent_id,
            boid=False,
            recreate=recreate,
            hijacking=is_hijacked,
            agent_interface=interface,
        )

        return vehicle

    def create_vehicle_in_providers(
        self,
        vehicle: Vehicle,
        agent_id: str,
    ):
        """Notify all providers of the existence of a vehicle."""
        self._check_valid()
        interface = self.agent_manager.agent_interface_for_agent_id(agent_id)
        for provider in self.providers:
            if interface.action_space in provider.action_spaces:
                provider.create_vehicle(
                    VehicleState(
                        vehicle_id=vehicle.id,
                        vehicle_config_type="passenger",
                        pose=vehicle.pose,
                        dimensions=vehicle.chassis.dimensions,
                        speed=vehicle.speed,
                        source="HIJACK",
                    )
                )

    def _setup_bullet_client(self, client: bc.BulletClient):
        client.resetSimulation()
        client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        # PyBullet defaults the timestep to 240Hz. Several parameters are tuned with
        # this value in mind. For example the number of solver iterations and the error
        # reduction parameters (erp) for contact, friction and non-contact joints.
        # Attempting to get around this we set the number of substeps so that
        # timestep * substeps = 240Hz. Bullet (C++) does something to this effect as
        # well (https://git.io/Jvf0M), but PyBullet does not expose it.
        # But if our timestep is variable (due to being externally driven)
        # then we will step pybullet multiple times ourselves as necessary
        # to account for the time delta on each SMARTS step.
        self._pybullet_period = (
            self._fixed_timestep_sec
            if self._fixed_timestep_sec
            else 1 / MAX_PYBULLET_FREQ
        )
        client.setPhysicsEngineParameter(
            fixedTimeStep=self._pybullet_period,
            numSubSteps=int(self._pybullet_period * MAX_PYBULLET_FREQ),
            numSolverIterations=10,
            solverResidualThreshold=0.001,
            # warmStartingFactor=0.99
        )

        client.setGravity(0, 0, -9.8)
        self._map_bb = None
        self._setup_pybullet_ground_plane(client)

    def _setup_pybullet_ground_plane(self, client: bc.BulletClient):
        plane_path = self._scenario.plane_filepath
        if not os.path.exists(plane_path):
            with pkg_resources.path(models, "plane.urdf") as path:
                plane_path = str(path.absolute())

        if not self._map_bb:
            self._map_bb = self.road_map.bounding_box

        if self._map_bb:
            # 1e6 is the default value for plane length and width in smarts/models/plane.urdf.
            DEFAULT_PLANE_DIM = 1e6
            ground_plane_scale = (
                2.2 * max(self._map_bb.length, self._map_bb.width) / DEFAULT_PLANE_DIM
            )
            ground_plane_center = self._map_bb.center
        else:
            # first step on undefined map, just use a big scale (1e6).
            # it should get updated as soon as vehicles are added...
            ground_plane_scale = 1.0
            ground_plane_center = (0, 0, 0)

        if self._ground_bullet_id is not None:
            client.removeBody(self._ground_bullet_id)
            self._ground_bullet_id = None

        self._ground_bullet_id = client.loadURDF(
            plane_path,
            useFixedBase=True,
            basePosition=ground_plane_center,
            globalScaling=ground_plane_scale,
        )

    def teardown(self):
        """Clean up episode resources."""
        if self._agent_manager is not None:
            self._agent_manager.teardown()
        if self._vehicle_index is not None:
            self._vehicle_index.teardown()

        if self._bullet_client is not None:
            self._bullet_client.resetSimulation()
        if self._renderer is not None:
            self._renderer.teardown()
        if self._traffic_sim is not None:
            self._traffic_sim.teardown()
        self._teardown_providers()

        if self._bubble_manager is not None:
            self._bubble_manager.teardown()
            self._bubble_manager = None
        if self._trap_manager is not None:
            self._trap_manager.teardown()
            self._trap_manager = None

        self._ground_bullet_id = None
        self._is_setup = False

    def destroy(self):
        """Destroy the simulation. Cleans up all remaining simulation resources."""
        if self._is_destroyed:
            return
        self.teardown()

        if self._envision:
            self._envision.teardown()

        if self._visdom:
            self._visdom.teardown()

        if self._agent_manager is not None:
            self._agent_manager.destroy()
            self._agent_manager = None
        if self._vehicle_index is not None:
            self._vehicle_index = None
        if self._traffic_sim is not None:
            self._traffic_sim.destroy()
            self._traffic_sim = None
        if self._renderer is not None:
            self._renderer.destroy()
            self._renderer = None
        if self._bullet_client is not None:
            self._bullet_client.disconnect()
            self._bullet_client = None
        self._is_destroyed = True

    def _check_valid(self):
        if self._is_destroyed:
            raise SMARTSDestroyedError(
                "The current SMARTS instance has already been destroyed."
            )

    def __del__(self):
        try:
            self.destroy()
        except (TypeError, AttributeError) as e:
            # This is a print statement because the logging module may be deleted at program exit.
            raise SMARTSDestroyedError(
                "ERROR: A SMARTS instance may have been deleted by gc before a call to destroy."
                " Please explicitly call `del obj` or `SMARTS.destroy()` to make this error"
                " go away.",
                e,
            )

    def _teardown_vehicles(self, vehicle_ids):
        self._vehicle_index.teardown_vehicles_by_vehicle_ids(vehicle_ids)
        self._clear_collisions(vehicle_ids)

    def attach_sensors_to_vehicles(self, agent_interface, vehicle_ids):
        """Set the specified vehicles with the sensors needed to satisfy the specified agent
        interface.
        """
        self._check_valid()
        self._agent_manager.attach_sensors_to_vehicles(
            self, agent_interface, vehicle_ids
        )

    def observe_from(
        self, vehicle_ids: Sequence[str]
    ) -> Tuple[
        Dict[str, Observation], Dict[str, float], Dict[str, float], Dict[str, bool]
    ]:
        """Generate observations from the specified vehicles."""
        self._check_valid()
        return self._agent_manager.observe_from(
            self, vehicle_ids, self._traffic_history_provider.done_this_step
        )

    @property
    def renderer(self):
        """The renderer singleton. On call, the sim will attempt to create it if it does not exist."""
        if not self._renderer:
            try:
                from .renderer import Renderer

                self._renderer = Renderer(self._sim_id)
                if self._scenario:
                    self._renderer.setup(self._scenario)
                    self._vehicle_index.begin_rendering_vehicles(self._renderer)
            except Exception as e:
                self._log.warning("unable to create Renderer:  " + repr(e))
                self._renderer = None
        return self._renderer

    @property
    def is_rendering(self) -> bool:
        """If the simulation has image rendering active."""
        return self._renderer is not None

    @property
    def road_stiffness(self) -> Any:
        """The stiffness of the road."""
        return self._bullet_client.getDynamicsInfo(self._ground_bullet_id, -1)[9]

    @property
    def dynamic_action_spaces(self) -> Set[ActionSpaceType]:
        """The set of vehicle action spaces that use dynamics."""
        return self._dynamic_action_spaces

    @property
    def traffic_sim(
        self,
    ) -> Optional[SumoTrafficSimulation]:
        """The underlying traffic simulation."""
        return self._traffic_sim

    @property
    def road_map(self) -> RoadMap:
        """The road map api which allows lookup of road features."""
        return self.scenario.road_map

    @property
    def external_provider(self) -> ExternalProvider:
        """The external provider that can be used to inject vehicle states directly."""
        return self._external_provider

    @property
    def bc(self):
        """The bullet physics client instance."""
        return self._bullet_client

    @property
    def envision(self) -> Optional[EnvisionClient]:
        """The envision instance"""
        return self._envision

    @property
    def step_count(self) -> int:
        """The number of steps since the last reset."""
        return self._step_count

    @property
    def elapsed_sim_time(self) -> float:
        """Elapsed time since simulation start."""
        return self._elapsed_sim_time

    @property
    def version(self) -> str:
        """SMARTS version."""
        return VERSION

    def teardown_agents_without_vehicles(self, agent_ids: Iterable[str]):
        """
        Teardown agents in the given list that have no vehicles registered as
        controlled-by or shadowed-by
        Params:
            agent_ids: Sequence of agent ids
        """
        self._check_valid()
        agents_to_teardown = {
            agent_id
            for agent_id in agent_ids
            # Only clean-up when there are no controlled agents left (e.g. boids)
            if len(
                self._vehicle_index.vehicles_by_actor_id(
                    agent_id, include_shadowers=True
                )
            )
            == 0
        }

        agents_to_teardown = {
            id_
            for id_ in agents_to_teardown
            if not self.agent_manager.is_boid_keep_alive_agent(id_)
        }
        self.agent_manager.teardown_social_agents(filter_ids=agents_to_teardown)

    def _teardown_vehicles_and_agents(self, vehicle_ids):
        shadow_and_controlling_agents = set()
        for vehicle_id in vehicle_ids:
            agent_id = self._vehicle_index.actor_id_from_vehicle_id(vehicle_id)
            if agent_id:
                shadow_and_controlling_agents.add(agent_id)

            shadow_agent_id = self._vehicle_index.shadow_actor_id_from_vehicle_id(
                vehicle_id
            )
            if shadow_agent_id:
                shadow_and_controlling_agents.add(shadow_agent_id)

        self._vehicle_index.teardown_vehicles_by_vehicle_ids(vehicle_ids)
        self.teardown_agents_without_vehicles(shadow_and_controlling_agents)

    def _pybullet_provider_sync(self, provider_state: ProviderState):
        current_vehicle_ids = {v.vehicle_id for v in provider_state.vehicles}
        previous_sv_ids = self._vehicle_index.social_vehicle_ids()
        exited_vehicles = previous_sv_ids - current_vehicle_ids
        self._teardown_vehicles_and_agents(exited_vehicles)

        # Update our pybullet world given this provider state
        dt = provider_state.dt or self._last_dt
        for vehicle in provider_state.vehicles:
            vehicle_id = vehicle.vehicle_id
            # either this is a pybullet agent vehicle, or it is a social vehicle
            if vehicle_id in self._vehicle_index.agent_vehicle_ids():
                if not vehicle.updated:
                    # this is an agent vehicle
                    agent_id = self._vehicle_index.actor_id_from_vehicle_id(vehicle_id)
                    agent_interface = self._agent_manager.agent_interface_for_agent_id(
                        agent_id
                    )
                    agent_action_space = agent_interface.action_space
                    if agent_action_space not in self._dynamic_action_spaces:
                        # This is not a pybullet agent, but it has an avatar in this world
                        # to make it's observations. Update the avatar to match the new
                        # state of this vehicle
                        # XXX: this needs to be disentangled from pybullet.
                        pybullet_vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
                        assert isinstance(pybullet_vehicle.chassis, BoxChassis)
                        pybullet_vehicle.update_state(vehicle, dt=dt)
            else:
                # This vehicle is a social vehicle
                if vehicle_id in self._vehicle_index.social_vehicle_ids():
                    social_vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
                else:
                    # It is a new social vehicle we have not seen yet.
                    # Create it's avatar.
                    # XXX: this needs to be disentangled from pybullet.
                    # XXX: (adding social vehicles to the vehicle index should not require pybullet to be present)
                    social_vehicle = self._vehicle_index.build_social_vehicle(
                        sim=self,
                        vehicle_state=vehicle,
                        actor_id=vehicle_id,
                        vehicle_id=vehicle_id,
                        vehicle_config_type=vehicle.vehicle_config_type,
                    )
                # Update the social vehicle avatar to match the vehicle state
                if not vehicle.updated:
                    # Note:  update_state() happens *after* pybullet has been stepped.
                    social_vehicle.update_state(vehicle, dt=dt)

    def _step_pybullet(self):
        self._bullet_client.stepSimulation()
        pybullet_substeps = max(1, round(self._last_dt / self._pybullet_period)) - 1
        for _ in range(pybullet_substeps):
            for vehicle in self._vehicle_index.vehicles:
                vehicle.chassis.reapply_last_control()
            self._bullet_client.stepSimulation()
        for vehicle in self._vehicle_index.vehicles:
            vehicle.step(self._elapsed_sim_time)

    def _get_provider_state(self, source: str, action_space_pred) -> ProviderState:
        agent_ids = {
            agent_id
            for agent_id, interface in self._agent_manager.agent_interfaces.items()
            if action_space_pred(interface.action_space)
        }
        provider_state = ProviderState()
        for vehicle_id in self._vehicle_index.agent_vehicle_ids():
            agent_id = self._vehicle_index.actor_id_from_vehicle_id(vehicle_id)
            if agent_id not in agent_ids:
                continue
            vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
            provider_state.vehicles.append(
                VehicleState(
                    vehicle_id=vehicle.id,
                    vehicle_config_type="passenger",
                    pose=vehicle.pose,
                    dimensions=vehicle.chassis.dimensions,
                    speed=vehicle.speed,
                    source=source,
                )
            )
        return provider_state

    @property
    def vehicle_index(self):
        """The vehicle index for direct vehicle manipulation."""
        return self._vehicle_index

    @property
    def agent_manager(self) -> AgentManager:
        """The agent manager for direct agent manipulation."""
        return self._agent_manager

    @property
    def providers(self) -> List[Provider]:
        """The current providers contributing to the simulation."""
        # TODO: Add check to ensure that action spaces are disjoint between providers
        # TODO: It's inconsistent that pybullet is not here
        return self._providers

    def get_provider_by_type(self, requested_type) -> Provider:
        """Get The first provider that matches the requested type."""
        self._check_valid()
        for provider in self._providers:
            if isinstance(provider, requested_type):
                return provider
        return None

    def _setup_providers(self, scenario) -> ProviderState:
        provider_state = ProviderState()
        for provider in self.providers:
            try:
                new_provider_state = provider.setup(scenario)
            except Exception as provider_error:
                new_provider_state = self._handle_provider(provider, provider_error)
            provider_state.merge(new_provider_state)
        return provider_state

    def _teardown_providers(self):
        for provider in self.providers:
            provider.teardown()
        self._last_provider_state = None

    def _harmonize_providers(self, provider_state: ProviderState):
        for provider in self.providers:
            try:
                provider.sync(provider_state)
            except Exception as provider_error:
                self._handle_provider(provider, provider_error)
        self._pybullet_provider_sync(provider_state)
        if self._renderer:
            self._sync_vehicles_to_renderer()

    def _reset_providers(self):
        for provider in self.providers:
            try:
                provider.reset()
            except Exception as provider_error:
                self._handle_provider(provider, provider_error)

    def _handle_provider(self, provider: Provider, provider_error) -> ProviderState:
        provider_problem = bool(provider_error or not provider.connected)
        if not provider_problem:
            return

        recovery_flags = self._provider_recovery_flags.get(
            provider, ProviderRecoveryFlags.EXPERIMENT_REQUIRED
        )
        recovered = False
        if recovery_flags & ProviderRecoveryFlags.ATTEMPT_RECOVERY:
            provider_state, recovered = provider.recover(
                self._scenario, self.elapsed_sim_time, provider_error
            )

        provider_state = provider_state or ProviderState()
        if recovered:
            return provider_state

        if recovery_flags & ProviderRecoveryFlags.EPISODE_REQUIRED:
            self._reset_required = True
            if self._resetting:
                self._log.error(
                    f"`Provider {provider.__class__.__name__} has crashed during reset`"
                )
                raise provider_error
            return provider_state
        elif recovery_flags & ProviderRecoveryFlags.EXPERIMENT_REQUIRED:
            raise provider_error

        # default to re-raise error
        raise provider_error

    def _step_providers(self, actions) -> ProviderState:
        accumulated_provider_state = ProviderState()

        def agent_controls_vehicles(agent_id):
            vehicles = self._vehicle_index.vehicles_by_actor_id(agent_id)
            return len(vehicles) > 0

        def matches_provider_action_spaces(agent_id, action_spaces):
            interface = self._agent_manager.agent_interface_for_agent_id(agent_id)
            return interface.action_space in action_spaces

        def matches_no_provider_action_space(agent_id):
            interface = self._agent_manager.agent_interface_for_agent_id(agent_id)
            for provider in self.providers:
                if interface.action_space in provider.action_spaces:
                    return False
            return True

        pybullet_actions = {}
        other_actions = {}
        for agent_id, action in actions.items():
            if not agent_controls_vehicles(agent_id):
                continue
            if matches_provider_action_spaces(agent_id, self._dynamic_action_spaces):
                pybullet_actions[agent_id] = action
            elif matches_no_provider_action_space(agent_id):
                other_actions[agent_id] = action

        if pybullet_actions or other_actions:
            self._perform_agent_actions(pybullet_actions)
            self._perform_agent_actions(other_actions)
            self._check_ground_plane()
            self._step_pybullet()
            self._process_collisions()
            if pybullet_actions:
                as_pred = (
                    lambda action_space: action_space in self._dynamic_action_spaces
                )
                accumulated_provider_state.merge(
                    self._get_provider_state("PYBULLET", as_pred)
                )
            if other_actions:
                as_pred = (
                    lambda action_space: action_space not in self._dynamic_action_spaces
                )
                accumulated_provider_state.merge(
                    self._get_provider_state("OTHER", as_pred)
                )

        for provider in self.providers:
            try:
                provider_state = self._step_provider(provider, actions)
            except Exception as provider_error:
                provider_state = self._handle_provider(provider, provider_error)

            if provider == self._traffic_sim:
                # Remove agent vehicles from provider vehicles
                provider_state.filter(self._vehicle_index.agent_vehicle_ids())
            accumulated_provider_state.merge(provider_state)

        self._harmonize_providers(accumulated_provider_state)
        return accumulated_provider_state

    def _step_provider(self, provider, actions):
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

                if self._agent_manager.is_boid_agent(agent_id):
                    for vehicle_id, vehicle_action in action.items():
                        assert vehicle_id in vehicle_ids
                        provider_actions[vehicle_id] = vehicle_action
                else:
                    assert len(vehicle_ids) == 1
                    provider_actions[vehicle_ids[0]] = action

        provider_state = provider.step(
            provider_actions, self._last_dt, self._elapsed_sim_time
        )
        return provider_state

    @property
    def should_reset(self):
        """If the simulation requires a reset."""
        return self._reset_required

    @property
    def scenario(self):
        """The current simulation scenario."""
        return self._scenario

    @property
    def timestep_sec(self) -> float:
        """Deprecated. Use `fixed_timestep_sec`."""
        warnings.warn(
            "SMARTS timestep_sec property has been deprecated in favor of fixed_timestep_sec.  Please update your code.",
            category=DeprecationWarning,
        )
        return self.fixed_timestep_sec

    @property
    def fixed_timestep_sec(self) -> float:
        """The simulation fixed timestep."""
        # May be None if time deltas are externally driven
        return self._fixed_timestep_sec

    @fixed_timestep_sec.setter
    def fixed_timestep_sec(self, fixed_timestep_sec: float):
        if not fixed_timestep_sec:
            # This is the fastest we could possibly run given constraints from pybullet
            self._rounder = rounder_for_dt(round(1 / MAX_PYBULLET_FREQ, 6))
        else:
            self._rounder = rounder_for_dt(fixed_timestep_sec)
        self._fixed_timestep_sec = fixed_timestep_sec
        self._is_setup = False  # need to re-setup pybullet

    @property
    def last_dt(self) -> float:
        """The last delta time."""
        assert not self._last_dt or self._last_dt > 0
        return self._last_dt

    def neighborhood_vehicles_around_vehicle(self, vehicle, radius=None):
        """Find vehicles in the vicinity of the target vehicle."""
        self._check_valid()
        other_states = [v for v in self._vehicle_states if v.vehicle_id != vehicle.id]
        if radius is None:
            return other_states

        other_positions = [state.pose.position for state in other_states]
        if not other_positions:
            return []

        distances = euclidean_distances(other_positions, [vehicle.position]).reshape(
            -1,
        )
        indices = np.argwhere(distances <= radius).flatten()
        return [other_states[i] for i in indices]

    def vehicle_did_collide(self, vehicle_id) -> bool:
        """Test if the given vehicle had any collisions in the last physics update."""
        self._check_valid()
        for c in self._vehicle_collisions[vehicle_id]:
            if c.collidee_id != self._ground_bullet_id:
                return True
        return False

    def vehicle_collisions(self, vehicle_id) -> List[Collision]:
        """Get a list of all collisions the given vehicle was involved in during the last
        physics update.
        """
        self._check_valid()
        return [
            c
            for c in self._vehicle_collisions[vehicle_id]
            if c.collidee_id != self._ground_bullet_id
        ]

    def _clear_collisions(self, vehicle_ids):
        for vehicle_id in vehicle_ids:
            self._vehicle_collisions.pop(vehicle_id, None)

    def _perform_agent_actions(self, agent_actions):
        for agent_id, action in agent_actions.items():
            agent_vehicles = self._vehicle_index.vehicles_by_actor_id(agent_id)
            if len(agent_vehicles) == 0:
                self._log.warning(
                    f"{agent_id} doesn't have a vehicle, is the agent done? (dropping action)"
                )
                continue
            agent_interface = self._agent_manager.agent_interface_for_agent_id(agent_id)
            is_boid_agent = self._agent_manager.is_boid_agent(agent_id)

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
                    agent_interface.vehicle_type,
                )

    def _sync_vehicles_to_renderer(self):
        assert self._renderer
        for vehicle in self._vehicle_index.vehicles:
            vehicle.sync_to_renderer()

    def _process_collisions(self):
        self._vehicle_collisions = defaultdict(list)  # list of `Collision` instances

        for vehicle_id in self._vehicle_index.agent_vehicle_ids():
            vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
            # We are only concerned with vehicle-vehicle collisions
            collidee_bullet_ids = set(
                [p.bullet_id for p in vehicle.chassis.contact_points]
            )
            if self._ground_bullet_id in collidee_bullet_ids:
                collidee_bullet_ids.remove(self._ground_bullet_id)

            if not collidee_bullet_ids:
                continue

            for bullet_id in collidee_bullet_ids:
                collidee = self._bullet_id_to_vehicle(bullet_id)
                actor_id = self._vehicle_index.actor_id_from_vehicle_id(collidee.id)
                # TODO: Should we specify the collidee as the vehicle ID instead of
                #       the agent/social ID?
                collision = Collision(collidee_id=actor_id)
                self._vehicle_collisions[vehicle_id].append(collision)

    def _bullet_id_to_vehicle(self, bullet_id):
        for vehicle in self._vehicle_index.vehicles:
            if bullet_id == vehicle.chassis.bullet_id:
                return vehicle
        assert (
            False
        ), f"Only collisions with agent or social vehicles is supported, hit {bullet_id}"

    def _check_ground_plane(self):
        rescale_plane = False
        map_min = np.array(self._map_bb.min_pt[:2]) if self._map_bb else None
        map_max = np.array(self._map_bb.max_pt[:2]) if self._map_bb else None
        for vehicle_id in self._vehicle_index.agent_vehicle_ids():
            vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
            map_spot = np.array(vehicle.pose.position[:2])
            if map_min is None:
                map_min = map_spot
                rescale_plane = True
            elif any(map_spot < map_min):
                map_min = np.minimum(map_spot, map_min)
                rescale_plane = True
            if map_max is None:
                map_max = map_spot
                rescale_plane = True
            elif any(map_spot > map_max):
                map_max = np.maximum(map_spot, map_max)
                rescale_plane = True
        if rescale_plane:
            MIN_DIM = 500.0
            if map_max[0] - map_min[0] < MIN_DIM:
                map_min[0] -= MIN_DIM
                map_max[0] += MIN_DIM
            if map_max[1] - map_min[1] < MIN_DIM:
                map_min[1] -= MIN_DIM
                map_max[1] += MIN_DIM
            self._map_bb = BoundingBox(Point(*map_min), Point(*map_max))
            self._log.info(
                f"rescaling pybullet ground plane to at least {map_min} and {map_max}"
            )
            self._setup_pybullet_ground_plane(self._bullet_client)

    def _try_emit_envision_state(self, provider_state: ProviderState, obs, scores):
        if not self._envision:
            return

        traffic = {}
        position = {}
        speed = {}
        heading = {}
        lane_ids = {}
        for v in provider_state.vehicles:
            if v.vehicle_id in self._vehicle_index.agent_vehicle_ids():
                # this is an agent controlled vehicle
                agent_id = self._vehicle_index.actor_id_from_vehicle_id(v.vehicle_id)
                agent_obs = obs[agent_id]
                is_boid_agent = self._agent_manager.is_boid_agent(agent_id)
                vehicle_obs = agent_obs[v.vehicle_id] if is_boid_agent else agent_obs

                if self._agent_manager.is_ego(agent_id):
                    actor_type = envision_types.TrafficActorType.Agent
                    mission_route_geometry = (
                        self._vehicle_index.sensor_state_for_vehicle_id(
                            v.vehicle_id
                        ).plan.route.geometry
                    )
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
                    name=self._agent_manager.agent_name(agent_id),
                    actor_type=actor_type,
                    vehicle_type=envision_types.VehicleType.Car,
                    position=tuple(v.pose.position),
                    heading=float(v.pose.heading),
                    speed=v.speed,
                    actor_id=envision_types.format_actor_id(
                        agent_id,
                        v.vehicle_id,
                        is_multi=is_boid_agent,
                    ),
                    events=vehicle_obs.events,
                    waypoint_paths=(vehicle_obs.waypoint_paths or []) + road_waypoints,
                    point_cloud=point_cloud,
                    driven_path=driven_path,
                    mission_route_geometry=mission_route_geometry,
                )
                speed[agent_id] = v.speed
                position[agent_id] = tuple(v.pose.position[:2])
                heading[agent_id] = float(v.pose.heading)
                if (
                    vehicle_obs.waypoint_paths
                    and len(vehicle_obs.waypoint_paths[0]) > 0
                ):
                    lane_ids[agent_id] = vehicle_obs.waypoint_paths[0][0].lane_id
            elif v.vehicle_id in self._vehicle_index.social_vehicle_ids():
                # this is a social vehicle
                veh_type = (
                    v.vehicle_config_type if v.vehicle_config_type else v.vehicle_type
                )
                traffic[v.vehicle_id] = envision_types.TrafficActorState(
                    actor_type=envision_types.TrafficActorType.SocialVehicle,
                    vehicle_type=veh_type,
                    position=tuple(v.pose.position),
                    heading=float(v.pose.heading),
                    speed=v.speed,
                )

        bubble_geometry = [
            list(bubble.geometry.exterior.coords)
            for bubble in self._bubble_manager.bubbles
        ]

        scenario_folder_path = self.scenario._root
        scenario_name = os.path.split((scenario_folder_path).rstrip("/"))[1]
        assert (
            scenario_name != ""
        ), f"Scenario name was not properly extracted from the scenario folder path: {scenario_folder_path}"

        state = envision_types.State(
            traffic=traffic,
            scenario_id=self.scenario.scenario_hash,
            scenario_name=scenario_name,
            bubbles=bubble_geometry,
            scene_colors=SceneColors.EnvisionColors.value,
            scores=scores,
            ego_agent_ids=list(self._agent_manager.ego_agent_ids),
            position=position,
            speed=speed,
            heading=heading,
            lane_ids=lane_ids,
            frame_time=self._rounder(self._elapsed_sim_time + self._total_sim_time),
        )
        self._envision.send(state)

    def _try_emit_visdom_obs(self, obs):
        if not self._visdom:
            return
        self._visdom.send(obs)
