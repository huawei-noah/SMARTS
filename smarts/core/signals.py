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
from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING, List, Optional

from smarts.core.colors import SceneColors

from .actor import ActorState

if TYPE_CHECKING:
    from smarts.core import coordinates


class SignalLightState(IntFlag):
    """States that a traffic signal light may take;
    note that these may be combined into a bit-mask."""

    UNKNOWN = 0
    OFF = 0
    STOP = 1
    CAUTION = 2
    GO = 4
    FLASHING = 8
    ARROW = 16


def signal_state_to_color(state: SignalLightState) -> SceneColors:
    """Maps a signal state to a color."""
    if state == SignalLightState.STOP:
        return SceneColors.SignalStop
    elif state == SignalLightState.CAUTION:
        return SceneColors.SignalCaution
    elif state == SignalLightState.GO:
        return SceneColors.SignalGo
    else:
        return SceneColors.SignalUnknown


@dataclass
class SignalState(ActorState):
    """Traffic signal state information."""

    state: Optional[SignalLightState] = None
    stopping_pos: Optional[coordinates.Point] = None
    controlled_lanes: Optional[List[str]] = None
    last_changed: Optional[float] = None  # will be None if not known

    def __post_init__(self):
        assert self.state is not None
