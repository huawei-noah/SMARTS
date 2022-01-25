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
from dataclasses import dataclass, field
from enum import IntFlag
from typing import List, Optional, Set, Tuple

from .controllers import ActionSpaceType
from .scenario import Scenario
from .vehicle import VehicleState


class ProviderRecoveryFlags(IntFlag):
    """This describes actions to be taken with a provider should it fail."""

    NOT_REQUIRED = 0x00000000
    """Not needed for the current step. Error causes skip."""
    EPISODE_REQUIRED = 0x00000010
    """Needed for the current episode. Results in episode ending."""
    EXPERIMENT_REQUIRED = 0x00000100
    """Needed for the experiment. Results in exception if an error is thrown."""
    ATTEMPT_RECOVERY = 0x00001000
    """Provider should attempt to recover from the exception or disconnection."""


@dataclass
class ProviderState:
    """State information from a provider."""

    vehicles: List[VehicleState] = field(default_factory=list)
    dt: Optional[float] = None  # most Providers can leave this blank

    def merge(self, other: "ProviderState"):
        """Merge state with another provider's state."""
        our_vehicles = {v.vehicle_id for v in self.vehicles}
        other_vehicles = {v.vehicle_id for v in other.vehicles}
        assert our_vehicles.isdisjoint(other_vehicles)

        self.vehicles += other.vehicles
        self.dt = max(self.dt, other.dt, key=lambda x: x if x else 0)

    def filter(self, vehicle_ids):
        """Filter vehicle states down to the given vehicles."""
        provider_vehicle_ids = [v.vehicle_id for v in self.vehicles]
        for v_id in vehicle_ids:
            try:
                index = provider_vehicle_ids.index(v_id)
                del provider_vehicle_ids[index]
                del self.vehicles[index]
            except ValueError:
                continue


class Provider:
    """A Provider tracks a (sub)set of vehicles that all share the same action space.
    This is a base class (interface) from which all Providers should inherit."""

    ## TAI: Consider renaming to VehicleSet or somesuch.

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        """The action spaces of the provider."""
        raise NotImplementedError

    def setup(self, scenario: Scenario) -> ProviderState:
        """Initialize the provider with a scenario."""
        raise NotImplementedError

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        """Progress the provider to generate new vehicle state."""
        raise NotImplementedError

    def sync(self, provider_state: ProviderState):
        """Synchronize with external state."""
        raise NotImplementedError

    def create_vehicle(self, provider_vehicle: VehicleState):
        """Create a vehicle within the provider."""
        raise NotImplementedError

    def reset(self):
        """Reset this provider to a pre-initialized state."""
        raise NotImplementedError

    def teardown(self):
        """Clean up provider resources."""
        raise NotImplementedError

    def recover(
        self, scenario, elapsed_sim_time: float, error: Optional[Exception] = None
    ) -> Tuple[ProviderState, bool]:
        """Attempt to reconnect the provider if an error or disconnection occurred.
        Implementations may choose to e-raise the passed in exception.
        Args:
            scenario (Scenario): The scenario of the current episode.
            elapsed_sim_time (float): The current elapsed simulation time.
            error (Optional[Exception]): An exception if an exception was thrown.
        Returns:
            bool: The success/failure of the attempt to reconnect.
        """
        if error:
            raise error
        return ProviderState(), False

    @property
    def connected(self) -> bool:
        """Determine if the provider is still responsive. (e.g. the case that the provider is
        sending provider state over the internet and has stopped responding)
        Returns:
            bool: The connection state of the provider.
        """
        return True
