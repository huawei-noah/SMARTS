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
from typing import Dict, Optional, Sequence, Set, Tuple

from .actor import ActorRole, ActorState
from .controllers import ActionSpaceType
from .provider import Provider, ProviderRecoveryFlags, ProviderState
from .road_map import RoadMap
from .scenario import Scenario
from .signals import SignalLightState, SignalState


class SignalProvider(Provider):
    """A SignalProvider manages traffic light signals."""

    def __init__(self):
        self._my_signals: Dict[str, SignalState] = dict()
        # start with the default recovery flags...
        self._recovery_flags = super().recovery_flags
        self._road_map = None

    @property
    def recovery_flags(self) -> ProviderRecoveryFlags:
        return self._recovery_flags

    @recovery_flags.setter
    def recovery_flags(self, flags: ProviderRecoveryFlags):
        self._recovery_flags = flags

    @property
    def actions(self) -> Set[ActionSpaceType]:
        # TAI: could allow for agents to hijack things like signals as well.
        # (Then the action spcae could just be the desired SignalLightState.)
        return set()

    @property
    def _provider_state(self) -> ProviderState:
        return ProviderState(actors=list(self._my_signals.values()))

    def setup(self, scenario: Scenario) -> ProviderState:
        self._road_map = scenario.road_map
        # we only need to control map signals that won't be controlled by other providers .
        # XXX:  we assume that if there is a history provider, it will provide the signal state,
        # XXX:  and if sumo traffic is supported then SumoTrafficSimulation is being used and it will provide state for Sumo signals.
        if scenario.traffic_history is None and not scenario.supports_sumo_traffic:
            for feature in self._road_map.dynamic_features:
                if feature.type == RoadMap.FeatureType.FIXED_LOC_SIGNAL:
                    controlled_lanes = [feature.type_specific_info]
                    self._my_signals[feature.feature_id] = SignalState(
                        actor_id=feature.feature_id,
                        actor_type="signal",
                        source=self.source_str,
                        role=ActorRole.Signal,
                        state=SignalLightState.UNKNOWN,
                        stopping_pos=feature.geometry[0],
                        controlled_lanes=controlled_lanes,
                        last_changed=None,
                    )
        return self._provider_state

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        # TODO:  update state of signals we control here based on some reasonable/simple default program
        return self._provider_state

    def sync(self, provider_state: ProviderState):
        for actor_state in provider_state.actors:
            if actor_state.actor_id in self._my_signals:
                assert isinstance(actor_state, SignalState)
                self._my_signals[actor_state.actor_id] = actor_state

    def can_accept_actor(self, state: ActorState) -> bool:
        return isinstance(state, SignalState)

    def add_actor(
        self, provider_actor: ActorState, from_provider: Optional[Provider] = None
    ):
        assert isinstance(provider_actor, SignalState)
        self._my_signals[provider_actor.actor_id] = provider_actor

    def reset(self):
        pass

    def teardown(self):
        self._my_signals = dict()

    def manages_actor(self, actor_id: str) -> bool:
        return actor_id in self._my_signals

    def stop_managing(self, actor_id: str):
        if actor_id in self._my_signals:
            del self._my_signals[actor_id]
