# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import enum
from collections import defaultdict
from enum import Enum, IntFlag, unique
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    SupportsFloat,
    Tuple,
    TypeVar,
)

import gymnasium as gym
import numpy as np

from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1

from .agent_communication import (
    Bands,
    Header,
    Message,
    MessagePasser,
    V2XReceiver,
    V2XTransmitter,
)

C = TypeVar("C", Callable[[str, SMARTS], Sequence[Tuple[Header, Message]]])


class VehicleWatch(gym.Wrapper):
    """A wrapper that augments the MessagePasser wrapper to allow programmable messages.

    These messages are configured through vehicle_watches.

    Args:
        env (MessagePasser): The base environment. This must be a MessagePasser.
        vehicle_watches (Dict[str, Tuple[V2XTransmitter, C]]):
            The configurable message generator. The first part is the transmitter config, the
            second is the transmission generator callable.
    """

    def __init__(
        self, env: MessagePasser, vehicle_watches: Dict[str, Tuple[V2XTransmitter, C]]
    ):
        super().__init__(env)
        assert isinstance(self.env, MessagePasser)

        # action space should remain the same
        self._vehicle_watches = vehicle_watches

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Resets the environment

        Args:
            seed (int, optional): The environment seed. Defaults to None.
            options (Dict[str, Any], optional): The options to the environment. Defaults to None.

        Returns:
            Tuple[Any, Dict[str, Any]]: The observations and infos.
        """
        return super().reset(seed=seed, options=options)

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """The gym step function."""
        observation, reward, term, trunc, info = super().step(action)
        env: MessagePasser = self.env

        smarts: SMARTS = env.smarts
        msgs = [
            watch(target, smarts)
            for target, (_, watch) in self._vehicle_watches.values()
        ]
        env.augment_observations(msgs, observation)

        return observation, reward, term, trunc, info

    def programmed_sumo_device(
        self, target: str, smarts: SMARTS
    ) -> Sequence[Tuple[Header, Message]]:
        """An example transmission method.

        Args:
            target (str): The intended sender of the transmission.
            smarts (SMARTS): The smarts instance to grab relevant information from.

        Returns:
            Sequence[Tuple[Header, Message]]: A new set of transmissions.
        """
        traffic_sim: SumoTrafficSimulation = smarts.get_provider_by_type(
            SumoTrafficSimulation
        )

        if not target in traffic_sim.actor_ids:
            return []

        return [
            (
                Header(
                    channel="position",
                    sender=target,
                    sender_type="leader",
                    cc={"__all__"},
                    bcc={},
                    format="position",
                ),
                Message(smarts.vehicle_index.vehicle_by_id(target).position),
            )
        ]
