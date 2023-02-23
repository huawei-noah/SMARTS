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
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

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


class V2XReceiver(NamedTuple):
    """A configuratoin utility to set up agent to receive messages."""

    bands: Bands
    aliases: List[str]
    whitelist_channels: Optional[List[str]] = None
    blacklist_channels: Set[str] = set()
    sensitivity: Sensitivity = Sensitivity.STANDARD


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
        super().__init__(env)
        self._message_config = message_config
        # map alias to agent ids (multiple agents can be under the same alias)
        self._alias_mapping = defaultdict(list)
        for a_id, (_, receiver) in message_config.items():
            for alias in receiver.aliases:
                self._alias_mapping[alias].append(a_id)

        assert isinstance(env, HiWayEnvV1)
        o_action_space: gym.spaces.Dict = self.env.action_space
        msg_space = (
            gym.spaces.Box(low=0, high=256, shape=(max_message_bytes,), dtype=np.uint8),
        )
        self.action_space = gym.spaces.Dict(
            {
                a_id: gym.spaces.Tuple(
                    (
                        base_action_space,
                        msg_space,
                    )
                )
                for a_id, base_action_space in o_action_space.spaces.items()
            }
        )
        o_observation_space: gym.spaces.Dict = self.env.observation_space
        self._transmission_space = gym.spaces.Sequence(
            gym.spaces.Tuple(
                (
                    gym.spaces.Tuple(
                        (
                            gym.spaces.Text(20),  # channel
                            gym.spaces.Text(30),  # sender
                            gym.spaces.Text(10),  # sender_type
                            gym.spaces.Sequence(gym.spaces.Text(30)),  # cc
                            gym.spaces.Sequence(gym.spaces.Text(30)),  # bcc
                            gym.spaces.Text(10),  # format
                        )
                    ),
                    gym.spaces.Tuple((msg_space,)),
                )
            )
        )
        self.observation_space = gym.spaces.Dict(
            {
                a_id: gym.spaces.Dict(
                    dict(
                        **obs,
                        transmissions=self._transmission_space,
                    )
                )
                for a_id, obs in o_observation_space.items()
            }
        )

    @lru_cache
    def resolve_alias(self, alias):
        """Resolve the alias to agent ids."""
        return set(self._alias_mapping[alias])

    def step(self, action):
        """Steps the environment using the given action."""
        std_actions = {a_id: act for a_id, (act, _) in action}
        observations, rewards, terms, truncs, infos = self.env.step(std_actions)

        msgs = defaultdict(list)

        # pytype: disable=wrong-arg-types
        # filter recipients for active
        cached_active_filter = lru_cache(lambda a: a.intersection(observations.keys()))

        # filter recipients by band
        ## compare transmitter
        cached_band_filter = lru_cache(
            lambda sender, recipients: (
                r
                for r in recipients
                if self._message_config[sender][0].bands
                | self._message_config[r][1].bands
            )
        )

        # filter recipients that do not listen to the channel
        accepts_channel = lru_cache(
            lambda channel, recipient: (
                (not self._message_config[recipient][1].whitelist_channels)
                or (channel in self._message_config[recipient][1].whitelist_channels)
            )
            and channel not in self._message_config[recipient][1].blacklist_channels
        )
        cached_channel_filter = lru_cache(
            lambda channel, recipients: (
                r for r in recipients if accepts_channel(channel, r)
            )
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
        general_filter = lambda header, initial_recipients: (
            cc
            for recipients in map(self.resolve_alias, initial_recipients)
            for cc in cached_channel_filter(
                header.channel,
                cached_band_filter(header.sender, cached_active_filter(recipients)),
            )
        )

        # Organise the messages to their recipients
        for a_id, (_, msg) in action.items():
            msg: List[Tuple[Header, Message]] = msg
            for header, message in msg:
                header: Header = header
                message: Message = message

                # expand the recipients
                cc_recipients = set(general_filter(header, header.cc))
                bcc_recipients = set(general_filter(header, header.bcc))
                cc_header = header._replace(cc=cc_recipients)

                # associate the messages to the recipients
                for recipient in (
                    cc_recipients - bcc_recipients
                ):  # privacy takes priority
                    msgs[recipient].append(
                        (cc_header._replace(bcc=set()), message)  # clear bcc
                    )
                for recipient in bcc_recipients:
                    msgs[recipient].append(
                        (
                            cc_header._replace(
                                bcc={recipient}
                            ),  # leave recipient in bcc
                            message,
                        )
                    )

        obs_with_msgs = {
            a_id: dict(
                **obs,
                transmissions=msgs[a_id],
            )
            for a_id, obs in observations.items()
        }
        return obs_with_msgs, rewards, terms, truncs, infos

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Resets the environment."""
        observations, info = super().reset(seed=seed, options=options)
        obs_with_msgs = {
            a_id: dict(**obs, transmissions=self._transmission_space.sample(0))
            for a_id, obs in observations.items()
        }
        return obs_with_msgs, info
