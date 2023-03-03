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
    def __init__(self, env, vehicle_watches: Dict[str, Tuple[V2XTransmitter, C]]):
        super().__init__(env)
        assert isinstance(self.env, MessagePasser)

        # action space should remain the same
        self._vehicle_watches = vehicle_watches

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
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
