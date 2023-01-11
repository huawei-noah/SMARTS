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
import logging
from dataclasses import dataclass, field
from enum import IntFlag
from typing import List, Optional, Sequence, Set, Tuple

from .actor import ActorRole, ActorState
from .controllers import ActionSpaceType
from .road_map import RoadMap
from .scenario import Scenario


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
    RELINQUISH_ACTORS = 0x00010000
    """Provider should relinquish its agents if it cannot / will not recover."""


@dataclass
class ProviderState:
    """State information from a provider."""

    actors: List[ActorState] = field(default_factory=list)
    dt: Optional[float] = None  # most Providers can leave this blank

    def merge(self, other: "ProviderState"):
        """Merge state with another provider's state."""
        our_actors = {a.actor_id for a in self.actors}
        other_actors = {a.actor_id for a in other.actors}
        if not our_actors.isdisjoint(other_actors):
            overlap = our_actors & other_actors
            logging.warning(
                f"multiple providers control the same actors: {overlap}. "
                "Later added providers will take priority. "
            )
            logging.info(
                "Conflicting actor states: \n"
                f"Previous: {[(a.actor_id, a.source) for a in self.actors if a.actor_id in overlap]}\n"
                f"Later: {[(a.actor_id, a.source) for a in other.actors if a.actor_id in overlap]}\n"
            )

        ## TODO: Properly harmonize these actor ids so that there is a priority and per actor source
        self.actors += filter(lambda a: a.actor_id not in our_actors, other.actors)

        self.dt = max(self.dt, other.dt, key=lambda x: x if x else 0)

    def filter(self, actor_ids):
        """Filter actor states down to the given actors."""
        provider_actor_ids = [a.actor_id for a in self.actors]
        for a_id in actor_ids:
            try:
                index = provider_actor_ids.index(a_id)
                del provider_actor_ids[index]
                del self.actors[index]
            except ValueError:
                continue

    def intersects(self, actor_ids: Set[str]) -> bool:
        """Returns True iff any of the actor_ids are contained in this ProviderState .
        Returns False for empty-set containment."""
        provider_actor_ids = {a.actor_id for a in self.actors}
        intersection = actor_ids & provider_actor_ids
        return bool(intersection)


class ProviderManager:
    """Interface to be implemented by a class that manages a set of Providers
    that jointly control a set of actors, such that they can hand these off to
    each other when necessary.  Actors can only be passed among Providers that
    are managed by the same ProviderManager.  Providers can call these methods
    on the manager to do so."""

    # TODO:  do this is in a way such that external providers do not require any
    # sort of "injection" call (like set_manager() below) to set a manager reference.
    # One possibility:  instead of calling "provider_relinquishing_actor()", they
    # could just set the "source" field in the ActorState object to None and
    # other Providers that are willing to accept new actors could watch for this.

    def provider_relinquishing_actor(
        self, provider: "Provider", state: ActorState
    ) -> Optional["Provider"]:
        """Find a new Provider for an actor from among the Providers managed
        by this ProviderManager.  Returns the new provider or None if a suitable
        one could not be found, in which case the actor is removed."""
        raise NotImplementedError

    def provider_removing_actor(self, provider: "Provider", actor_id: str):
        """Called by a Provider when it is removing an actor from the simulation.
        This was added for convenience, but it isn't always necessary to be called."""
        raise NotImplementedError


class Provider:
    """A Provider manages a (sub)set of actors (e.g., vehicles) that all share the same action space(s).
    This is a base class (interface) from which all Providers should inherit."""

    @property
    def recovery_flags(self) -> ProviderRecoveryFlags:
        """Flags specifying what this provider should do if it fails.
        (May be overridden by child classes.)"""
        return (
            ProviderRecoveryFlags.EXPERIMENT_REQUIRED
            | ProviderRecoveryFlags.RELINQUISH_ACTORS
        )

    @recovery_flags.setter
    def recovery_flags(self, flags: ProviderRecoveryFlags):
        """Setter to allow recovery flags to be changed."""
        raise NotImplementedError

    def set_manager(self, manager: ProviderManager):
        """Indicate the manager that this provider should inform of all actor handoffs."""
        raise NotImplementedError

    @property
    def actions(self) -> Set[ActionSpaceType]:
        """The action spaces of the provider."""
        raise NotImplementedError

    def setup(self, scenario: Scenario) -> ProviderState:
        """Initialize the provider with a scenario."""
        raise NotImplementedError

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        """Progress the provider to generate new actor state.
        Args:
            actions: one or more valid actions from the supported action_spaces of this provider
            dt: time (in seconds) to simulate during this simulation step
            elapsed_sim_time: amount of time (in seconds) that's elapsed so far in the simulation
        Returns:
            ProviderState representing the state of all actors this manages.
        """
        raise NotImplementedError

    def sync(self, provider_state: ProviderState):
        """Synchronize with state managed by other Providers."""
        raise NotImplementedError

    def can_accept_actor(self, state: ActorState) -> bool:
        """Whether this Provider can take control of an existing actor
        with state that was previously managed by another Provider.
        The state.role field should indicate the desired role, not the
        previous role."""
        return False

    def add_actor(
        self, provider_actor: ActorState, from_provider: Optional["Provider"] = None
    ):
        """Management of the actor with state is being assigned
        (or transferred if from_provider is not None) to this Provider.
        Will only be called if can_accept_actor() has returned True."""
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
        Implementations may choose to re-raise the passed in exception.
        Args:
            scenario (Scenario): The scenario of the current episode.
            elapsed_sim_time (float): The current elapsed simulation time.
            error (Optional[Exception]): An exception if an exception was thrown.
        Returns:
            ProviderState: the state of the provider upon recovery
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

    @property
    def source_str(self) -> str:
        """This property should be used to fill in the source field
        of all ActorState objects created/managed by this Provider."""
        return self.__class__.__name__

    def manages_actor(self, actor_id: str) -> bool:
        """Returns True iff the actor referenced by actor_id is managed by this Provider."""
        raise NotImplementedError

    def stop_managing(self, actor_id: str):
        """Tells the Provider to stop managing the specified actor;
        it will be managed by another Provider now."""
        raise NotImplementedError

    def remove_actor(self, actor_id: str):
        """The actor is being removed from the simulation."""
        if self.manages_actor(actor_id):
            self.stop_managing(actor_id)
        # can be overridden to do more cleanup as necessary
