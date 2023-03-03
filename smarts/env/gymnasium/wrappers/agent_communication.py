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
from functools import lru_cache, partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import gymnasium as gym
import numpy as np

from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1


@unique
class Bands(IntFlag):
    """Communcation bands."""

    L0 = enum.auto()
    L1 = enum.auto()
    L2 = enum.auto()
    L3 = enum.auto()
    L4 = enum.auto()
    L5 = enum.auto()
    L6 = enum.auto()
    L7 = enum.auto()
    L8 = enum.auto()
    L9 = enum.auto()
    L10 = enum.auto()
    L11 = enum.auto()
    ALL = L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11


@unique
class Sensitivity(Enum):
    LOW = 0
    STANDARD = 1
    HIGH = 2


class Header(NamedTuple):
    """A header modeled loosely after an email.

    Args:
        channel (str): The channel this message is on.
        sender (str): The name of the sender.
        sender_type (str): The type of actor the sender is.
        cc (Set[str]):
            The actors and aliases this should be sent publicly to. All cc recipients see the cc.
        bcc (Set[str]):
            The actors and aliases this should be sent privately to. The recipient only sees self
            in bcc list.
        format (str): The format of this message.
    """

    channel: str
    sender: str
    sender_type: str
    cc: Set[str]
    bcc: Set[str]
    format: str


class Message(NamedTuple):
    """The message data.

    Args:
        content (Any): The content of this message.
    """

    content: Any
    # size: uint


class V2XTransmitter(NamedTuple):
    """A configuration utility to set up agents to transmit messages."""

    bands: Bands
    range: float
    # available_channels: List[str]
    # max_message_bytes: int = 125000


class V2XReceiver(NamedTuple):
    """A configuratoin utility to set up agent to receive messages."""

    bands: Bands
    aliases: Set[str]
    whitelist_channels: Optional[Set[str]] = None
    blacklist_channels: Set[str] = set()
    sensitivity: Sensitivity = Sensitivity.STANDARD


@lru_cache
def active_filter(a: frozenset, a_in_observations):
    return frozenset(a.intersection(a_in_observations))


# filter recipients by band
## compare transmitter
def band_filter(
    sender, recipients, message_config: Dict[str, Tuple[V2XTransmitter, V2XReceiver]]
):
    return frozenset(
        r
        for r in recipients
        if message_config[sender][0].bands | message_config[r][1].bands
    )


# filter recipients that do not listen to the channel
def accepts_channel(channel, receiver: V2XReceiver):
    return (
        (not receiver.whitelist_channels) or (channel in receiver.whitelist_channels)
    ) and channel not in receiver.blacklist_channels


def channel_filter(
    channel, recipients, message_config: Dict[str, Tuple[V2XTransmitter, V2XReceiver]]
):
    return frozenset(
        r for r in recipients if accepts_channel(channel, message_config[r][1])
    )


# pytype: enable=wrong-arg-types

## filter recipients by distance
## Includes all
## TODO ensure this works on all formatting types
# cached_dist_comp = lambda sender, receiver: obs[sender]["position"].dot(obs[receiver]["position"])
# cached_distance_filter = lru_cache(lambda sender, receivers: (
#     r for r in receivers if cached_distance_filter
# ))

# compress filters
def general_filter(
    header,
    initial_recipients,
    observations,
    message_config,
    alias_resolver: Callable[[frozenset], frozenset],
):
    return (
        cc
        for recipients in map(alias_resolver, initial_recipients)
        for cc in channel_filter(
            header.channel,
            band_filter(
                header.sender,
                active_filter(frozenset(recipients), frozenset(observations.keys())),
                message_config,
            ),
            message_config,
        )
    )


class MessagePasser(gym.Wrapper):
    """This wrapper augments the observations and actions to require passing messages from agents.

    It assumes that the underlying environment is :class:`HiWayEnvV1`"""

    def __init__(
        self,
        env: gym.Env,
        message_config: Dict[str, Tuple[V2XTransmitter, V2XReceiver]],
        max_message_bytes=125000,
    ):
        """"""
        assert isinstance(env.unwrapped, HiWayEnvV1)
        super().__init__(env)
        self._max_message_bytes = max_message_bytes
        self._message_config = message_config
        # map alias to agent ids (multiple agents can be under the same alias)
        self._alias_mapping = defaultdict(list)
        for a_id, (_, receiver) in message_config.items():
            for alias in receiver.aliases:
                self._alias_mapping[alias].append(a_id)
            self._alias_mapping[a_id].append(a_id)
            self._alias_mapping["__all__"].append(a_id)

        o_action_space: gym.spaces.Dict = self.env.action_space
        o_observation_space: gym.spaces.Dict = self.env.observation_space

        header_space = gym.spaces.Tuple(
            (
                gym.spaces.Text(20),  # channel
                gym.spaces.Text(30),  # sender
                gym.spaces.Text(10),  # sender_type
                gym.spaces.Sequence(gym.spaces.Text(30)),  # cc
                gym.spaces.Sequence(gym.spaces.Text(30)),  # bcc
                gym.spaces.Text(10),  # format
            )
        )

        def gen_msg_body_space(max_message_bytes: int):
            return gym.spaces.Tuple(
                (
                    gym.spaces.Box(
                        low=0, high=256, shape=(max_message_bytes,), dtype=np.uint8
                    ),
                )
            )

        def gen_msg_space(max_message_bytes: int):
            return gym.spaces.Tuple(
                (
                    header_space,
                    gen_msg_body_space(max_message_bytes),
                )
            )

        def gen_transmission_space(max_message_bytes: int):
            return gym.spaces.Sequence(gen_msg_space(max_message_bytes))

        self._transmission_space = gen_transmission_space(
            max_message_bytes=max_message_bytes
        )

        _action_space = {}
        for a_id, base_action_space in o_action_space.spaces.items():
            if a_id not in message_config:
                _action_space[a_id] = base_action_space
            else:
                _action_space[a_id] = gym.spaces.Tuple(
                    (
                        base_action_space,
                        self._transmission_space,
                    )
                )
        self.action_space = gym.spaces.Dict(_action_space)

        _observation_space = {}
        for a_id, obs in o_observation_space.spaces.items():
            if a_id not in message_config:
                _observation_space[a_id] = obs
            else:
                _observation_space[a_id] = gym.spaces.Dict(
                    dict(
                        **obs,
                        transmissions=self._transmission_space,
                    )
                )
        self.observation_space = gym.spaces.Dict(_observation_space)

    @property
    def message_config(self):
        """The current message config.

        Returns:
            Dict[str, Tuple[V2XTransmitter, V2XReceiver]]: The message configuration.
        """
        return self._message_config.copy()

    @property
    def max_message_bytes(self):
        """The max message bytes size.

        Returns:
            int: The max size of a message.
        """
        return self._max_message_bytes

    @lru_cache
    def resolve_alias(self, alias):
        """Resolve the alias to agent ids."""
        return frozenset(self._alias_mapping[alias])

    def step(self, action):
        """Steps the environment using the given action."""
        std_actions = {a_id: act for a_id, (act, _) in action.items()}
        observations, rewards, terms, truncs, infos = self.env.step(std_actions)

        obs_with_msgs = {
            a_id: dict(
                **obs,
                transmissions=[],
            )
            if a_id in self._message_config
            else obs
            for a_id, obs in observations.items()
        }

        self.augment_observations(
            messages=(msg for (_, msg) in action.values()), observations=obs_with_msgs
        )

        return obs_with_msgs, rewards, terms, truncs, infos

    def augment_observations(
        self,
        messages,
        observations,
    ):
        f = partial(
            general_filter,
            observations=observations,
            message_config=self._message_config,
            alias_resolver=self.resolve_alias,
        )
        for msg in messages:
            msg: List[Tuple[Header, Message]] = msg
            for header, message in msg:
                header: Header = header
                message: Message = message

                # expand the recipients
                cc_recipients = set(
                    f(header=header, initial_recipients=frozenset(header.cc))
                )
                bcc_recipients = set(
                    f(header=header, initial_recipients=frozenset(header.bcc))
                )
                cc_header = header._replace(cc=cc_recipients)

                # associate the messages to the recipients
                for recipient in (
                    cc_recipients - bcc_recipients
                ):  # privacy takes priority
                    observations[recipient]["transmissions"].append(
                        (cc_header._replace(bcc=set()), message)  # clear bcc
                    )
                for recipient in bcc_recipients:
                    observations[recipient]["transmissions"].append(
                        (
                            cc_header._replace(
                                bcc={recipient}
                            ),  # leave recipient in bcc
                            message,
                        )
                    )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Resets the environment."""
        observations, info = super().reset(seed=seed, options=options)
        obs_with_msgs = {
            a_id: dict(**obs, transmissions=self._transmission_space.sample((0, ())))
            for a_id, obs in observations.items()
        }
        return obs_with_msgs, info
