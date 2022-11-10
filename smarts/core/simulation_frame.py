# MIT License
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from cached_property import cached_property

from smarts.core.actor import ActorState
from smarts.core.agent_interface import AgentInterface
from smarts.core.vehicle_state import Collision, VehicleState

logger = logging.getLogger(__name__)


# TODO MTA: Move this class to a new separate file for typehint purposes
@dataclass(frozen=True)
class SimulationFrame:
    """This is state that should change per step of the simulation."""

    actor_states: Dict[str, ActorState]
    agent_vehicle_controls: Dict[str, str]
    agent_interfaces: Dict[str, AgentInterface]
    ego_ids: str
    elapsed_sim_time: float
    fixed_timestep: float
    resetting: bool
    # road_map: RoadMap
    map_spec: Any
    last_dt: float
    last_provider_state: Any
    step_count: int
    vehicle_collisions: Dict[str, List[Collision]]
    # TODO MTA: this association should be between agents and sensors
    vehicles_for_agents: Dict[str, List[str]]
    vehicle_ids: Set[str]
    vehicle_states: Dict[str, VehicleState]
    # TODO MTA: Some sensors still cause issues with serialization
    vehicle_sensors: Dict[str, Dict[str, Any]]

    sensor_states: Any
    # TODO MTA: this can be allowed here as long as it is only type information
    # renderer_type: Any = None
    _ground_bullet_id: Optional[str] = None

    @cached_property
    def all_agent_ids(self):
        return set(self.agent_interfaces.keys())

    @cached_property
    def agent_ids(self) -> Set[str]:
        return set(self.vehicles_for_agents.keys())

    def vehicle_did_collide(self, vehicle_id) -> bool:
        """Test if the given vehicle had any collisions in the last physics update."""
        vehicle_collisions = self.vehicle_collisions.get(vehicle_id, [])
        for c in vehicle_collisions:
            if c.collidee_id != self._ground_bullet_id:
                return True
        return False

    def filtered_vehicle_collisions(self, vehicle_id) -> List[Collision]:
        """Get a list of all collisions the given vehicle was involved in during the last
        physics update.
        """
        vehicle_collisions = self.vehicle_collisions.get(vehicle_id, [])
        return [
            c for c in vehicle_collisions if c.collidee_id != self._ground_bullet_id
        ]

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            assert self.vehicle_ids.__contains__
            assert self.agent_ids.union(self.vehicles_for_agents) == self.agent_ids
            assert len(self.agent_ids - set(self.agent_interfaces)) == 0
            assert len(self.vehicle_ids.symmetric_difference(self.vehicle_states))

    @staticmethod
    def serialize(simulation_frame: "SimulationFrame") -> Any:
        import cloudpickle

        return cloudpickle.dumps(simulation_frame)

    @staticmethod
    def deserialize(serialized_simulation_frame) -> "SimulationFrame":
        import cloudpickle

        return cloudpickle.loads(serialized_simulation_frame)
